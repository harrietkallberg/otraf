import pandas as pd
import json as json 
from pathlib import Path
from hashlib import md5
from pathlib import Path
import json
from hashlib import md5

def sanitize_keys(obj):
    """Recursively convert tuple keys to strings to make the structure JSON-safe."""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                key = "__TUPLE__" + "::".join(map(str, k))
            else:
                key = k
            new_obj[key] = sanitize_keys(v)
        return new_obj
    elif isinstance(obj, list):
        return [sanitize_keys(item) for item in obj]
    else:
        return obj

def load_json_array(file_path):
    if Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json_array(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def append_unique(existing_list, new_entries):
    seen_hashes = {md5(json.dumps(e, sort_keys=True).encode()).hexdigest() for e in existing_list}
    additions = []
    for entry in new_entries:
        h = md5(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        if h not in seen_hashes:
            additions.append(entry)
    return existing_list + additions


class DataFormer:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.route_id = self.raw_data['route_id'].iloc[0]
        self.route_short_name = self.raw_data['route_short_name'].iloc[0]
        self.route_long_name = self.get_route_long_name()
        self.set_up_log_structure()

        # Process data through the pipeline
        self.df_before = self.prepare_columns(raw_data)
        
        # STEP 1: Stop topology validation (enhanced with hierarchies and labels)
        print("STEP 1: Stop topology validation...")
        self.create_and_validate_stop_topology(self.df_before)
        
        print("STEP 2: Direction topology validation...")
        self.create_and_validate_direction_topology(self.df_before)
        print("✅ DataFormer initialization complete!")

        self.export_logs_and_navigation(export_dir="exported_logs")
        self.export_classifications(export_dir="exported_logs")
        self.export_global_flat_logs()

#   ================================ Preparation =====================================

    def get_route_long_name(self):
        if 'route_long_name' in self.raw_data.columns and not self.raw_data['route_long_name'].empty:
            # Get first non-null value
            first_valid = self.raw_data['route_long_name'].dropna()
            if not first_valid.empty:
                route_long_name = first_valid.iloc[0]
            else:
                route_long_name = 'Unknown'
        else:
            route_long_name = 'Unknown'
        return route_long_name

    def prepare_columns(self, df):
        "prepare dataframe to have a good structure before manipulating it"
        df = df.copy()

        if 'date' in df.columns:
            df.rename(columns={'date': 'update_date'}, inplace=True)
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], format='%Y%m%d')

        columns_to_keep = ['trip_id', 'route_long_name', 'start_date', 'direction_id', 'stop_id', 'stop_name', 'stop_sequence', 
            'scheduled_departure_time', 'observed_departure_time', 'departure_delay', 'route_short_name','city', 'parent_station']

        available_columns = [col for col in columns_to_keep if col in df.columns]

        df = df[available_columns].copy()
        
        if 'observed_departure_time' in df.columns and 'scheduled_departure_time' in df.columns and 'departure_delay' in df.columns:
            df['observed_departure_time'] = df['scheduled_departure_time'] + pd.to_timedelta(df['departure_delay'], unit='s')
        
        df = df.drop_duplicates(subset=['trip_id', 'stop_id', 'stop_name','direction_id', 'start_date'])
        df['month'] = df['start_date'].dt.month
        df['month_type'] = df['start_date'].dt.month.apply(lambda x: 1 if x >= 6 and x <= 8 else 0)
        df['day_type'] = df['start_date'].dt.weekday.apply(lambda x: 'weekday' if x <= 4 else ('saturday' if x == 5 else 'sunday'))
        
        # Add trip-consistent start_time (first scheduled departure time for each trip)
        trip_start_times = df.groupby(['trip_id', 'direction_id', 'start_date']).apply(
            lambda trip: trip.sort_values('stop_sequence')['scheduled_departure_time'].iloc[0]
        )
        
        # Map start_time back to all rows
        df['start_time'] = df.set_index(['trip_id', 'direction_id', 'start_date']).index.map(trip_start_times)
        
        # Add time_type based on consistent trip start_time
        def categorize_time(row):
            if pd.isna(row['start_time']):
                return 'unknown'
                
            if row['day_type'] != 'weekday':
                return 'weekend'
                
            hour = pd.to_datetime(row['start_time']).hour  # Use start_time instead of scheduled_departure_time
            if 6 <= hour < 9: 
                return 'am_rush'
            elif 9 <= hour < 15: 
                return 'day'
            elif 15 <= hour < 17: 
                return 'pm_rush'
            else: 
                return 'night'
            
        df['time_type'] = df.apply(categorize_time, axis=1)

        if all(col in df.columns for col in ['trip_id', 'stop_id','stop_name', 'direction_id', 'start_date']):
            # Count duplicates before dropping
            duplicate_count = len(df) - len(df.drop_duplicates(subset=['trip_id', 'stop_id', 'stop_name', 'direction_id', 'start_date']))
            print(f'Removing {duplicate_count} duplicates.')
            df = df.drop_duplicates(subset=['trip_id', 'stop_id', 'stop_name', 'direction_id', 'start_date'])

        return df
    
#   ====================================== Handle Violation Logging ==========================================

    def set_up_log_structure(self):
        # ===== STOP TOPOLOGY VALIDATION LOGS =====
        self.stop_topology_logs = {
            'parent_station_classifications': {}, # Classification of Parent station if possible
            'stop_id_direction_assignment_violations': {},    # Individual stop_id assignment issues
            'metadata': {
                'total_violations': 0,
                'violation_counts_by_type': {
                    'parent_station': 0, 
                    'stop_id': 0
                },
                'route_summary': {
                    'total_stops': 0,
                    'total_parent_stations': 0,
                    'topology_types': {}  # Count of each topology type
                }
            }
        }
        
        # ===== DIRECTION TOPOLOGY VALIDATION LOGS =====
        self.direction_topology_logs = {
            'direction_classifications': {},     # Classification of Direction if possible
            'stop_id_pattern_assignment_violations': {},     # Individual stop_id assignment issues 
            'metadata': {
                'total_violations': 0,
                'violation_counts_by_type': {
                    'direction': 0,
                    'stop_id': 0
                },
                'route_summary': {
                    'total_directions': 0,
                    'topology_types': {} # Count of each topology type
                }
            }
        }
        
        # ===== PERFORMANCE & RELIABILITY ANALYSIS LOGS =====
        self.performance_logs = {
            'punctuality_barcharts': {},   
            'regulatory_stops': {}, 
            'histograms_stops': {},         
            'metadata': {
                'analysis_period': {
                    'start_date': None,
                    'end_date': None,
                    'total_days': 0
                },
                'performance_summary': {
                    'overall_on_time_rate': 0.0,
                    'average_delay': 0.0,
                    'reliability_score': 0.0
                }
            }
        }
        
        # ===== NAVIGATION STRUCTURES =====
        self.navigation_structures = {
            'stop_hierarchies': {
                'by_stop_name': {},    # stop_name -> parent_station -> stop_ids
                'by_stop_id': {},      # stop_id -> (parent_station, stop_name, label)
            },
            'direction_hierarchies': {
                'by_direction': {},    # direction_id -> pattern -> stop_ids
                'by_stop_id': {}       #  stop_ids -> pattern ->  direction_id
            }
        }
    
    def store_violations(self, domain: str, labels: dict, violations: dict):
        """
        Store both labels and violations cleanly.
        """
        logs = {
            'stop_topology': self.stop_topology_logs,
            'direction_topology': self.direction_topology_logs
        }[domain]

        for label_type, label_data in labels.items():
            logs[f"{label_type}_classifications"] = label_data

        for vtype, entries in violations.items():
            if f"{vtype}_violations" not in logs:
                logs[f"{vtype}_violations"] = {}
            logs[f"{vtype}_violations"].update({
                f"{vtype}_{i + len(logs[f'{vtype}_violations'])}": v for i, v in enumerate(entries)
            })


        logs['metadata']['violation_counts_by_type'] = {
            k: len(violations.get(k, [])) for k in violations
        }
        logs['metadata']['total_violations'] = sum(len(vs) for vs in violations.values())

    def create_violation_entry(self, violation_type, severity, description, **details):
        """Create standardized violation entry with searchable fields"""
        return {
            'violation_type': violation_type,
            'severity': severity,
            'description': description,
            
            # SEARCHABLE FIELDS - always included
            'route_id': str(self.route_id),
            'route_name': self.route_long_name,
            'route_short_name': self.route_short_name,
       
            # Original details
            **details
        }

    def add_violation_to_log(self, log_dict, key, violation_entry):
        """Add violation to specified log with consistent key format"""
        log_dict[key] = violation_entry
        return violation_entry

#   ================ CLEARLY SEPARATED VALIDATION METHODS =====================

    def create_and_validate_stop_topology(self, df):
        """Main topology validation method for 2-layer approach"""
        
        print("\n=== COMPLETE STOP VALIDATION WORKFLOW ===")
        
        # STEP 1: Validate parent station topology
        parent_station_violations, parent_labels = self. _classify_all_parent_stations_diversity(df)
        
        # STEP 2: Validate individual stop ID directions
        stop_id_violations, stop_id_labels = self._validate_stop_id_directions(df)
        
        print(f"Found {len(parent_station_violations)} parent station + {len(stop_id_violations)} stop ID violations")
        
        # Store labels for hierarchy creation
        self._stop_validation_labels = {
            'parent_labels': parent_labels,
            'stop_id_labels': stop_id_labels
        }
        
        # Collect all violations
        all_violations = parent_station_violations + stop_id_violations
        
        # Store violations in 2-tier structure
        self.store_violations(
        'stop_topology',
        labels={'parent_station': parent_labels, 'stop_id': stop_id_labels},
        violations={'parent_station': parent_station_violations, 'stop_id': stop_id_violations}
        )
        
        # Create hierarchies with all labels
        self._build_stop_hierarchies(df, all_violations)
        
        print(f"✅ Topology validation complete with {len(all_violations)} total violations")

    def _classify_all_parent_stations_diversity(self, df):
        """STEP 1: Validate topology for all parent_stations"""
        
        print("Validating parent station topology...")
        
        # Get all unique parent stations
        unique_parent_stations = df['parent_station'].unique()
        
        violations = []
        parent_labels = {}

        for parent_station in unique_parent_stations:
            parent_station_str = str(parent_station)
            
            # Analyze this parent station's topology
            label, violation = self._analyze_single_parent_station_topology(df, parent_station_str)
            
            parent_labels[parent_station_str] = label
            
            if violation:
                violations.append(violation)
                print(f"🚩 Parent station {parent_station_str} [{label}]: {violation['description']}")
            else:
                print(f"✅ Parent station {parent_station_str} [{label}]: Valid topology")
        
        print(f"Parent station validation complete: {len(violations)} violations found")
        return violations, parent_labels

    def _analyze_single_parent_station_topology(self, df, parent_station):
        """Analyze topology for a single parent_station"""
        
        # Get all data for this parent station
        parent_data = df[df['parent_station'].astype(str) == str(parent_station)]
        
        # Get stop_ids and stop_names for this parent station
        stop_ids = parent_data['stop_id'].unique().tolist()
        stop_names = parent_data['stop_name'].unique().tolist()
        
        # Get directional analysis for each stop_id
        stop_id_analysis = {}
        for stop_id in stop_ids:
            directions, direction_counts = self._get_stop_id_directions(df, stop_id, parent_station)
            stop_id_analysis[str(stop_id)] = {
                'directions': directions,
                'direction_counts': direction_counts,
                'is_multi_directional': len(directions) > 1
            }
        
        # Count totals
        total_stops = len(stop_ids)
        multi_count = sum(1 for analysis in stop_id_analysis.values() if analysis['is_multi_directional'])
        single_count = total_stops - multi_count
        
        # Validate topology and assign label
        if total_stops == 1:
            if multi_count == 1:
                return 'Shared', None
            elif single_count == 1:
                return 'Unidirectional', None
        
        elif total_stops == 2:
            if single_count == 2 and multi_count == 0:
                return 'Bidirectional', None
            else:
                violation = self.create_violation_entry(
                    'two_stop_misassigned_directions',
                    'high',
                    f'2-stop station with wrong configuration: {single_count} single + {multi_count} multi (expected: 2 single + 0 multi)',
                    parent_station=parent_station,
                    stop_names=stop_names,
                    details={
                        'expected':'2 single-directional, 0 multi-directional',
                        'actual':f'{single_count} single-directional, {multi_count} multi-directional',
                        'directional_analysis':stop_id_analysis
                    }
                )
                stop_violations = []
                for sid in stop_ids:
                    stop_violations.append(self.create_violation_entry(
                        'stop_id_from_invalid_parent_topology',
                        'medium',
                        'Stop is part of parent_station with invalid topology',
                        stop_id=sid,
                        parent_station=parent_station
                    ))
                return 'Undefined', [violation] + stop_violations
        
        else:
            if total_stops % 2 == 0:
                if single_count == total_stops and multi_count == 0:
                    return 'Bidirectional', None
                elif single_count % 2 == 0 and multi_count % 2 == 0:
                    return 'Hybrid', None
                else:
                    violation = self.create_violation_entry(
                        'unpaired_directional_stops',
                        'medium',
                        f'Even-stop station with unpaired configuration: {single_count} single + {multi_count} multi',
                        parent_station=parent_station,
                        stop_names=stop_names,
                        details={
                            'expected':'even single-directional, even multi-directional',
                            'actual':f'{single_count} single-directional, {multi_count} multi-directional',
                            'directional_analysis':stop_id_analysis
                        }
                    )
                    stop_violations = []
                    for sid in stop_ids:
                        stop_violations.append(self.create_violation_entry(
                            'stop_id_from_invalid_parent_topology',
                            'medium',
                            'Stop is part of parent_station with invalid topology',
                            stop_id=sid,
                            parent_station=parent_station
                        ))
                    return 'Undefined', [violation] + stop_violations
            
            else:
                if single_count % 2 == 0 and multi_count % 2 == 1:
                    return 'Hybrid', None
                else:
                    violation = self.create_violation_entry(
                        'unpaired_directional_stops',
                        'medium',
                        f'Odd-stop station with unpaired configuration: {single_count} single + {multi_count} multi',
                        parent_station=parent_station,
                        stop_names=stop_names,
                        details={
                            'expected':'even single-directional, odd multi-directional',
                            'actual':f'{single_count} single-directional, {multi_count} multi-directional',
                            'directional_analysis':stop_id_analysis
                        }
                    )
                    stop_violations = []
                    for sid in stop_ids:
                        stop_violations.append(self.create_violation_entry(
                            'stop_id_from_invalid_parent_topology',
                            'medium',
                            'Stop is part of parent_station with invalid topology',
                            stop_id=sid,
                            parent_station=parent_station
                        ))
                    return 'Undefined', [violation] + stop_violations
        
        return 'Undefined', None

    def _validate_stop_id_directions(self, df):
        """STEP 2: Validate individual stop ID direction behavior"""
        
        print("Validating individual stop ID directions...")
        
        violations = []
        stop_id_labels = {}
        
        # Get all unique stop_ids with their context
        stop_id_data = df[['stop_id', 'stop_name', 'parent_station']].drop_duplicates()
        
        for _, row in stop_id_data.iterrows():
            stop_id = row['stop_id']
            stop_name = row['stop_name']
            parent_station = row['parent_station']
            
            # Analyze this specific stop_id
            directions, direction_counts = self._get_stop_id_directions(df, stop_id, parent_station)
            
            if len(directions) > 1:
                stop_id_labels[stop_id] = 'multi_directional'
            elif len(directions) == 1:
                stop_id_labels[stop_id] = 'single_directional'
            else:
                # No direction data - this is a stop_id violation
                stop_id_labels[stop_id] = 'no_data'
                violation = self.create_violation_entry(
                    'stop_id_without_direction_data',
                    'high',
                    f'Stop id: {stop_id} has no direction data',
                    stop_id=stop_id,
                    stop_name=stop_name,
                    parent_station=parent_station,
                    details={'direction_details':{
                        'directions': [directions],
                        'direction_counts': {direction_counts},
                        'is_multi_directional': False}
                    }
                )
                violations.append(violation)
                print(f"🚩 Stop ID {stop_id}: No direction data")
        
        print(f"Stop ID validation complete: {len(violations)} violations found")
        return violations, stop_id_labels

    def _get_stop_id_directions(self, df, stop_id, parent_station):
        """Helper: Get direction information for a specific stop_id within a parent station"""
        stop_data = df[
            (df['stop_id'].astype(str) == str(stop_id)) & 
            (df['parent_station'].astype(str) == str(parent_station))
        ]
        
        if stop_data.empty:
            return [], {}
        
        directions = stop_data['direction_id'].unique().tolist()
        direction_counts = stop_data['direction_id'].value_counts().to_dict()
        
        return directions, direction_counts

    def _build_stop_hierarchies(self, df, violations):
        """Build hierarchies with 2-layer validation labels"""

        labels = self._stop_validation_labels

        # ✅ Stop name labels not used for now
        stop_name_hierarchy, stop_id_hierarchy = self.create_bidirectional_stop_hierarchies(
            df,
            {},  # stop_name_labels
            labels['parent_labels'],  # correct: keyed by just parent_station
            labels['stop_id_labels']
        )

        # Store
        self._stop_name_hierarchy = stop_name_hierarchy
        self._stop_id_hierarchy = stop_id_hierarchy

        self.add_log_references_to_hierarchies({
            'stop_name_hierarchy': stop_name_hierarchy,
            'stop_id_hierarchy': stop_id_hierarchy
        }, violations)

    def create_bidirectional_stop_hierarchies(self, df, stop_name_labels, parent_station_labels, stop_id_labels):
        stop_name_hierarchy = {}
        stop_id_hierarchy = {}

        for _, row in df.drop_duplicates(['stop_id', 'stop_name', 'parent_station']).iterrows():
            stop_id = str(row['stop_id'])
            stop_name = row['stop_name']
            parent_station = str(row['parent_station'])

            # stop_name -> parent_station -> list of stop_ids
            stop_name_hierarchy.setdefault(stop_name, {}).setdefault(parent_station, []).append(stop_id)

            # stop_id -> (parent_station, stop_name, label)
            stop_id_hierarchy[stop_id] = {
                'parent_station': parent_station,
                'stop_name': stop_name,
                'label': stop_id_labels.get(stop_id, 'Unknown'),
                'has_violation': False
            }

        return stop_name_hierarchy, stop_id_hierarchy
    
#   ================ DIRECTION TOPOLOGY VALIDATION (following stop pattern) =====================
  
    def create_and_validate_direction_topology(self, df):
        """Main direction topology validation method (mirrors stop validation logic)."""
        print("\n=== COMPLETE DIRECTION VALIDATION WORKFLOW ===")
        
        # STEP 1: Validate direction pattern diversity
        direction_violations, direction_labels = self._classify_all_directions_diversity(df)
        
        # STEP 2: Validate stop coverage within directions
        stop_id_violations, stop_id_labels = self._validate_stop_id_issues(df)
        
        print(f"Found {len(direction_violations)} direction + {len(stop_id_violations)} stop violations")
        
        # Store labels for hierarchy creation
        self._direction_validation_labels = {
            'direction_labels': direction_labels,
            'stop_id_labels': stop_id_labels
        }
        
        # Collect all violations
        all_violations = direction_violations + stop_id_violations
        
        self.store_violations(
        'direction_topology',
        labels={'direction': direction_labels, 'stop_id': stop_id_labels},
        violations={'direction': direction_violations, 'stop_id': stop_id_violations}
        )
        
        # Create navigation structures and hierarchies
        self._build_direction_hierarchies(df, all_violations)
        
        print(f"✅ Direction topology validation complete with {len(all_violations)} total violations")

    def _classify_all_directions_diversity(self, df):
        """Detect if directions have consistent stop patterns or contain partial routes."""
        violations = []
        direction_labels = {}

        grouped = df.groupby('direction_id')

        for direction_id, group in grouped:
            patterns = group.groupby(['trip_id', 'start_date'])['stop_id'].apply(tuple)
            pattern_counts = patterns.value_counts()

            if len(pattern_counts) == 1:
                direction_labels[str(direction_id)] = 'Full-Route-Only'
            else:
                direction_labels[str(direction_id)] = 'Multi-Pattern'
                violation = self.create_violation_entry(
                    violation_type='multiple_patterns',
                    severity='medium',
                    description=f"Direction {direction_id} has multiple stop patterns",
                    direction_id=str(direction_id),
                    pattern_summary={
                        'unique_pattern_count': int(len(pattern_counts)),
                        'most_common_pattern_freq': int(pattern_counts.iloc[0]),
                        'top_patterns': {
                            "->".join(map(str, k)): int(v)
                            for k, v in pattern_counts.head(3).items()
                        }
                    }
                )
                violations.append(violation)

        return violations, direction_labels

    def _validate_stop_id_issues(self, df):
        """Assess coverage of each stop across direction instances."""

        # Count unique direction occurrences
        direction_counts = df.groupby('direction_id').apply(
            lambda g: g[['trip_id', 'start_date']].drop_duplicates().shape[0]
        ).rename("total_instances")

        # Count how often each stop appears in a direction
        stop_counts = df.groupby(['stop_id', 'direction_id']).apply(
            lambda x: x.groupby(['trip_id', 'start_date']).ngroups
        ).rename("stop_instances").reset_index()

        merged = stop_counts.merge(
            direction_counts.rename("total_instances"),
            on="direction_id"
        )

        # Calculate missing stats
        merged["missing_percent"] = (
            (merged["total_instances"] - merged["stop_instances"]) /
            merged["total_instances"] * 100
        )
        merged["missing_count"] = (
            merged["total_instances"] - merged["stop_instances"]
        )

        violations = []
        stop_labels = {}

        for _, row in merged.iterrows():
            stop_id = str(row["stop_id"])
            direction_id = str(row["direction_id"])
            missing_percent = row["missing_percent"]
            missing_count = row["missing_count"]
            total_count = row["total_instances"]

            # Filter the original df to just the relevant missing rows
            expected_trips = df[df["direction_id"] == direction_id][['trip_id', 'start_date', 'time_type']].drop_duplicates()
            actual_trips = df[(df["stop_id"] == stop_id) & (df["direction_id"] == direction_id)][['trip_id', 'start_date']].drop_duplicates()

            # Merge to identify which trips are missing
            merged_trips = expected_trips.merge(actual_trips, on=['trip_id', 'start_date'], how='left', indicator=True)
            missing_trips = merged_trips[merged_trips['_merge'] == 'left_only']

            time_type_counts = missing_trips["time_type"].value_counts()

            # Label the stop
            stop_labels[(stop_id, direction_id)] = 'varying_coverage' if missing_percent > 5 else 'high_coverage'


            if missing_percent > 5:
                severity = (
                    "high" if missing_percent > 50
                    else "medium" if missing_percent > 25
                    else "low"
                )

                violation = self.create_violation_entry(
                    violation_type="missing_stop_in_direction",
                    severity=severity,
                    description=f"Stop {stop_id} is missing from {missing_count} out of {total_count} trips in direction {direction_id}",
                    direction_id=direction_id,
                    stop_id=stop_id,
                    details={
                        "missing_count": int(missing_count),
                        "total_trips": int(total_count),
                        "missing_percentage": round(100 * missing_count / total_count, 1),
                        "missing_by_time_type": time_type_counts.to_dict()
                    }
                )

                violations.append(violation)

        return violations, stop_labels

    def _build_direction_hierarchies(self, df, violations):
        """Build consistent navigation structure for direction-level validation."""
        labels = self._direction_validation_labels

        direction_hierarchies = self._create_bidirectional_direction_hierarchies(df, labels['direction_labels'], labels['stop_id_labels'])
        self._direction_hierarchies = direction_hierarchies

        # Store in navigation_structures
        self.navigation_structures['direction_hierarchies'] = direction_hierarchies

        # Attach violations to hierarchy entries if needed
        self.add_log_references_to_hierarchies(direction_hierarchies, violations)

    def _create_bidirectional_direction_hierarchies(self, df, direction_labels, stop_id_labels):
        """Standardized creation of direction-based hierarchies."""
        by_direction = {}
        by_stop_id = {}

        for direction_id, group in df.groupby('direction_id'):
            dir_id = str(direction_id)

            stop_id_sequence = (
                group.sort_values(['trip_id', 'start_date', 'stop_sequence'])
                .drop_duplicates('stop_id')['stop_id']
                .astype(str)
                .tolist()
            )

            by_direction[dir_id] = {
                'canonical_stop_ids': stop_id_sequence,
                'label': direction_labels.get(dir_id, 'Unknown'),
                'num_trips': group['trip_id'].nunique(),
                'has_violation': False  # Add this
            }
            



        for stop_id, group in df.groupby('stop_id'):
            sid = str(stop_id)
            directions = group['direction_id'].astype(str).unique().tolist()

            by_stop_id[sid] = {
                'directions': directions,
                'direction_labels': [direction_labels.get(d, 'Unknown') for d in directions],
                'label': stop_id_labels.get(sid, 'Unknown'),
                'has_violation': False  # Add this too
            }

        return {
            'by_direction': by_direction,
            'by_stop_id': by_stop_id
        }

#   ================ DIRECTION TOPOLOGY VALIDATION (following stop pattern) =====================
    
    def add_log_references_to_hierarchies(self, hierarchies, violations):
        for violation in violations:
            stop_id = str(violation.get('stop_id', ''))
            direction_id = str(violation.get('direction_id', ''))

            for key in hierarchies:
                if 'stop' in key and stop_id:
                    if stop_id in hierarchies[key]:
                        hierarchies[key][stop_id]['has_violation'] = True

                if 'direction' in key and direction_id:
                    if direction_id in hierarchies[key]:
                        hierarchies[key][direction_id]['has_violation'] = True

    def export_logs_and_navigation(self, export_dir="exported_logs"):
        route_folder = Path(export_dir) / self.route_long_name
        route_folder.mkdir(parents=True, exist_ok=True)

        with open(route_folder / "stop_topology.json", 'w', encoding='utf-8') as f:
            json.dump(sanitize_keys(self.stop_topology_logs), f, indent=2, ensure_ascii=False)

        with open(route_folder / "direction_topology.json", 'w', encoding='utf-8') as f:
            json.dump(sanitize_keys(self.direction_topology_logs), f, indent=2, ensure_ascii=False)

        with open(route_folder / "navigation_structures.json", 'w', encoding='utf-8') as f:
            json.dump(sanitize_keys(self.navigation_structures), f, indent=2, ensure_ascii=False)

        print(f"📤 Exported full logs and navigation for route {self.route_id} -> {self.route_long_name}")

    def export_global_flat_logs(self, export_dir="exported_logs"):
        stop_violation_file = Path(export_dir) / "global_stop_violations.json"
        dir_violation_file = Path(export_dir) / "global_direction_violations.json"

        stop_violations = list(self.stop_topology_logs.get("stop_id_direction_assignment_violations", {}).values()) + \
                          list(self.stop_topology_logs.get("parent_station_classifications", {}).values())

        direction_violations = list(self.direction_topology_logs.get("stop_id_pattern_assignment_violations", {}).values()) + \
                               list(self.direction_topology_logs.get("direction_classifications", {}).values())

        stop_violations = [v for v in stop_violations if isinstance(v, dict) and v.get("violation_type")]
        direction_violations = [v for v in direction_violations if isinstance(v, dict) and v.get("violation_type")]

        updated_stops = load_json_array(stop_violation_file)
        updated_dirs = load_json_array(dir_violation_file)

        updated_stops.append({self.route_long_name: stop_violations})
        updated_dirs.append({self.route_long_name: direction_violations})

        save_json_array(updated_stops, stop_violation_file)
        save_json_array(updated_dirs, dir_violation_file)

    def export_classifications(self, export_dir: str):
        export_dir = Path(export_dir) / self.route_long_name
        export_dir.mkdir(parents=True, exist_ok=True)

        stop_class_file = Path(export_dir).parent / "global_stop_classifications.json"
        dir_class_file = Path(export_dir).parent / "global_direction_classifications.json"
        dir_stop_class_file = Path(export_dir).parent / "global_direction_stop_classifications.json"

        stop_class_entries = []
        for sid, label in self._stop_validation_labels.get("stop_id_labels", {}).items():
            stop_class_entries.append({"stop_id": sid, "label": label, "route_id": self.route_id})
        for pid, label in self._stop_validation_labels.get("parent_labels", {}).items():
            stop_class_entries.append({"parent_station": pid, "label": label, "route_id": self.route_id})

        updated_stop_classes = append_unique(load_json_array(stop_class_file), [{self.route_long_name: stop_class_entries}])
        save_json_array(updated_stop_classes, stop_class_file)

        direction_class_entries = [
            {"direction_id": did, "label": label, "route_id": self.route_id}
            for did, label in self._direction_validation_labels.get("direction_labels", {}).items()
        ]
        updated_dir_classes = append_unique(load_json_array(dir_class_file), [{self.route_long_name: direction_class_entries}])
        save_json_array(updated_dir_classes, dir_class_file)

        direction_stop_entries = []
        for key, label in self._direction_validation_labels.get("stop_id_labels", {}).items():
            if isinstance(key, tuple) and len(key) == 2:
                stop_id, direction_id = key
            else:
                stop_id, direction_id = key, None
            entry = {"stop_id": stop_id, "label": label, "route_id": self.route_id}
            if direction_id is not None:
                entry["direction_id"] = direction_id
            direction_stop_entries.append(entry)

        updated_dir_stop_classes = append_unique(load_json_array(dir_stop_class_file), [{self.route_long_name: direction_stop_entries}])
        save_json_array(updated_dir_stop_classes, dir_stop_class_file)

        print(f"📦 Classification export complete: "
              f"{len(stop_class_entries)} stop topology labels, "
              f"{len(direction_class_entries)} direction labels, "
              f"{len(direction_stop_entries)} direction-stop labels.")






















































#     def calculate_travel_times_and_delays(self, df):
#         """Calculate travel times using stop-level pattern violations"""
#         print("\n=== CALCULATING TRAVEL TIMES AND DELAYS ===")
        
#         df = df.sort_values(['trip_id', 'direction_id', 'start_date', 'stop_sequence'])
#         trip_groups = ['trip_id', 'direction_id', 'start_date']
        
#         # Get previous stop info
#         df['previous_stop'] = df.groupby(trip_groups)['stop_name'].shift(1)
#         df['prev_delay'] = df.groupby(trip_groups)['departure_delay'].shift(1)
        
#         # Calculate incremental delay
#         df['incremental_delay'] = df['departure_delay'] - df['prev_delay']
        
#         print("  Using stop-level pattern violations for travel time validation")
        
#         def is_travel_time_valid(row):
#             """Simple lookup in stop-level pattern violations log"""
            
#             # First stop in trip - no travel time possible
#             if pd.isna(row['previous_stop']):
#                 return False
            
#             # Check if this specific stop has a gap violation
            
#             violation_key = self.get_combination_key_from_row(row)

            
#             if hasattr(self, '_pattern_violations_log'):
#                 return violation_key not in self._pattern_violations_log  # Valid if NOT in violations log
            
#             return True  # Default to valid if no violations log
        
#         # Apply validation
#         df['travel_time_valid'] = df.apply(is_travel_time_valid, axis=1)
        
#         # Set incremental delay to NaN for invalid segments
#         df.loc[~df['travel_time_valid'], 'incremental_delay'] = np.nan
        
#         # Calculate travel times
#         time_columns = ['scheduled_departure_time', 'observed_departure_time']
#         for time_col in time_columns:
#             if time_col in df.columns:
#                 prefix = time_col.split('_')[0]
#                 prev_col = f'prev_{prefix}_departure'
#                 df[prev_col] = df.groupby(trip_groups)[time_col].shift(1)
#                 travel_col = f'{prefix}_travel_time'
                
#                 df[travel_col] = df[time_col] - df[prev_col]
#                 df.loc[~df['travel_time_valid'], travel_col] = pd.NaT
#                 df = df.drop(columns=[prev_col])
        
#         # Clean up
#         df = df.drop(columns=['prev_delay'])
        
#         valid_segments = df['travel_time_valid'].sum()
        
#         print(f"Calculation complete: {valid_segments}/{len(df)} valid segments ({valid_segments/len(df)*100:.1f}%)")
#         print(f"  Using stop-level pattern violations for precise validation")
        
#         return df

#     def identify_and_classify_stops(self, df):
#         """Detect regulatory stops"""
#         print("\n=== DETECTING REGULATORY STOPS ===")
        
#         df = df.copy()
#         df['is_regulatory'] = False
        
#         # Extract seconds and group by route-direction-trip_type
#         df['departure_seconds'] = df['scheduled_departure_time'].dt.second
        
#         regulatory_analysis = df.groupby(['direction_id', 'stop_id', 'trip_type']).agg({
#             'stop_name': 'first',
#             'departure_seconds': [
#                 lambda x: (x == 0).sum(),
#                 'count'
#             ]
#         }).reset_index()
        
#         regulatory_analysis.columns = [
#             'direction_id', 'stop_id', 'trip_type', 'stop_name', 
#             'zero_seconds_count', 'total_records'
#         ]
        
#         regulatory_analysis['zero_seconds_ratio'] = (
#             regulatory_analysis['zero_seconds_count'] / regulatory_analysis['total_records']
#         )
        
#         regulatory_analysis['is_perfectly_regulatory'] = regulatory_analysis['zero_seconds_ratio'] == 1.0
#         regulatory_analysis['is_regulatory'] = regulatory_analysis['zero_seconds_ratio'] >= 0.95
#         regulatory_analysis['has_anomaly'] = (
#             (regulatory_analysis['zero_seconds_ratio'] >= 0.95) & 
#             (regulatory_analysis['zero_seconds_ratio'] < 1.0)
#         )
        
#         regulatory_combinations = regulatory_analysis[regulatory_analysis['is_regulatory']].copy()
        
#         print(f"Found {len(regulatory_combinations)} regulatory combinations")
#         if regulatory_combinations['has_anomaly'].sum() > 0:
#             print(f"  - {regulatory_combinations['has_anomaly'].sum()} with anomalies")
        
#         # Update main dataframe
#         if len(regulatory_combinations) > 0:
#             for _, row in regulatory_combinations.iterrows():
#                 mask = (df['direction_id'] == row['direction_id']) & \
#                     (df['stop_id'] == row['stop_id']) & \
#                     (df['trip_type'] == row['trip_type'])
#                 df.loc[mask, 'is_regulatory'] = True
        
#         # Create logs
#         self._regulatory_stops_log = {}
#         regulatory_violations_log = {}
        
#         for _, row in regulatory_combinations.iterrows():
#             key = self.get_combination_key(row['stop_id'], row['direction_id'], row['trip_type'])
            
#             self._regulatory_stops_log[key] = {
#                 'route_id': self.route_id,
#                 'route_long_name': self.route_long_name,
#                 'route_short_name': self.route_short_name,
#                 'stop_id': row['stop_id'],
#                 'stop_name': row['stop_name'],
#                 'direction_id': row['direction_id'],
#                 'trip_type': row['trip_type'],
#                 'is_regulatory': True,
#                 'is_perfect': bool(row['is_perfectly_regulatory']),
#                 'has_anomaly': bool(row['has_anomaly']),
#                 'zero_seconds_ratio': row['zero_seconds_ratio'],
#                 'total_records': int(row['total_records'])
#             }
            
#             # Create violation for incomplete regulation
#             if row['has_anomaly']:
#                 violation = self.create_violation_entry(
#                     violation_type='incomplete_regulation',
#                     severity='low',
#                     description=f"Only {row['zero_seconds_ratio']:.1%} regulated",
#                     stop_id=row['stop_id'],
#                     stop_name=row['stop_name'],
#                     direction_id=row['direction_id'],
#                     trip_type=row['trip_type'],
#                     zero_seconds_ratio=row['zero_seconds_ratio'],
#                     total_records=int(row['total_records'])
#                 )
                
#                 violation_key = f"{key}"
#                 self.add_violation_to_log(regulatory_violations_log, violation_key, violation)
        
#         self._regulatory_violations_log = regulatory_violations_log
        
#         df = df.drop('departure_seconds', axis=1)
#         print(f"Regulatory violations created: {len(regulatory_violations_log)}")
#         print(f"Regulatory analysis complete: {len(self._regulatory_stops_log)} combinations, {len(regulatory_violations_log)} violations")
#         return df
   
# #   ======================================= Handle Navigational Maps =========================================

#         """Create route name ↔ route ID mapping for easy lookups"""
#         return {
#             f'{self.route_short_name}': {
#                 'route_id': self.route_id,
#                 'route_long_name': self.route_long_name
#             }
#         }  

#     # 1. ADD HISTOGRAM AND PUNCTUALITY ANALYSIS GENERATION
#     def create_histograms_and_punctuality_analysis(self, df):
#         """Create histograms and punctuality analysis aggregated across all trip types"""
#         print("\n=== CREATING HISTOGRAMS AND PUNCTUALITY ANALYSIS ===")
        
#         df = df.copy()
        
#         # Initialize logs
#         histograms_log = {}
#         punctuality_log = {}
        
#         # Group by stop, direction, and time_type (aggregating across trip_types)
#         analysis_groups = df.groupby(['stop_id', 'stop_name', 'direction_id', 'time_type'])
        
#         print(f"Processing {len(analysis_groups)} unique stop-direction-time combinations")
        
#         for (stop_id, stop_name, direction_id, time_type), group in analysis_groups:
            
#             # Create combination key WITHOUT trip_type
#             combo_key = f"{self.route_id}_{stop_id}_{direction_id}_{time_type}"
            
#             print(f"Analyzing: {stop_name} (Dir {direction_id}, {time_type})")
            
#             # HISTOGRAMS ANALYSIS
#             histogram_data = self._create_histogram_analysis(group, stop_id, stop_name, direction_id, time_type)
#             if histogram_data:
#                 histograms_log[combo_key] = histogram_data
            
#             # PUNCTUALITY ANALYSIS  
#             punctuality_data = self._create_punctuality_analysis(group, stop_id, stop_name, direction_id, time_type)
#             if punctuality_data:
#                 punctuality_log[combo_key] = punctuality_data
        
#         # Store in class
#         self._histograms_log = histograms_log
#         self._punctuality_log = punctuality_log
        
#         print(f"Analysis complete:")
#         print(f"  - {len(histograms_log)} histogram combinations")
#         print(f"  - {len(punctuality_log)} punctuality combinations")
        
#         return histograms_log, punctuality_log

#     def _create_histogram_analysis(self, group, stop_id, stop_name, direction_id, time_type, bins=20):
#         """Create histogram analysis for a specific combination"""
        
#         if len(group) < 10:  # Minimum sample size
#             return None
        
#         # Get clean delay data
#         total_delays = group['departure_delay'].dropna()
#         incremental_delays = group['incremental_delay'].dropna()
        
#         if len(total_delays) < 5:  # Need sufficient data
#             return None
        
#         histograms = {}
        
#         # Total delay histogram
#         if len(total_delays) >= 5:
#             histograms['total_delay'] = self._create_normalized_histogram(
#                 total_delays, bins, 
#                 f'Total Delay - {stop_name} (Dir {direction_id}, {time_type.replace("_", " ").title()})'
#             )
        
#         # Incremental delay histogram (only if valid travel times)
#         valid_incremental = incremental_delays[group['travel_time_valid'] == True] if 'travel_time_valid' in group.columns else incremental_delays
#         if len(valid_incremental) >= 5:
#             histograms['incremental_delay'] = self._create_normalized_histogram(
#                 valid_incremental, bins,
#                 f'Incremental Delay - {stop_name} (Dir {direction_id}, {time_type.replace("_", " ").title()})'
#             )
        
#         if not histograms:
#             return None
        
#         # Create standardized log entry
#         return {
#             # SEARCHABLE FIELDS - consistent with other logs
#             'route_id': str(self.route_id),
#             'route_name': self.route_long_name,
#             'route_short_name': self.route_short_name,
#             'stop_id': str(stop_id),
#             'stop_name': stop_name,
#             'direction_id': str(direction_id),
#             'time_type': time_type,
#             # Note: NO trip_type field since we aggregate across all trip types
            
#             # HISTOGRAM DATA
#             'histograms': histograms,
#             'metadata': {
#                 'bins_used': bins,
#                 'total_sample_size': len(group),
#                 'total_delay_sample_size': len(total_delays),
#                 'incremental_delay_sample_size': len(valid_incremental),
#                 'trip_types_included': sorted(group['trip_type'].unique().tolist()) if 'trip_type' in group.columns else ['all'],
#                 'histogram_count': len(histograms)
#             }
#         }

#     def _create_punctuality_analysis(self, group, stop_id, stop_name, direction_id, time_type):
#         """Create punctuality analysis for a specific combination"""
        
#         if len(group) < 5:  # Minimum sample size
#             return None
        
#         # Get clean delay data
#         delays = group['departure_delay'].dropna()
        
#         if len(delays) < 5:
#             return None
        
#         # Calculate punctuality metrics
#         punctuality_metrics = self._calculate_punctuality_metrics(delays)
        
#         if not punctuality_metrics:
#             return None
        
#         # Create standardized log entry
#         return {
#             # SEARCHABLE FIELDS - consistent with other logs
#             'route_id': str(self.route_id),
#             'route_name': self.route_long_name,
#             'route_short_name': self.route_short_name,
#             'stop_id': str(stop_id),
#             'stop_name': stop_name,
#             'direction_id': str(direction_id),
#             'time_type': time_type,
#             # Note: NO trip_type field since we aggregate across all trip types
            
#             # PUNCTUALITY DATA
#             'punctuality_metrics': punctuality_metrics,
#             'metadata': {
#                 'sample_size': len(delays),
#                 'trip_types_included': sorted(group['trip_type'].unique().tolist()) if 'trip_type' in group.columns else ['all'],
#                 'analysis_type': 'departure_delay_based'
#             }
#         }

#     def _create_normalized_histogram(self, data, bins, title):
#         """Create a normalized histogram from delay data"""
        
#         # Calculate histogram
#         counts, bin_edges = np.histogram(data, bins=bins)
        
#         # Normalize to probability distribution (sum = 1.0)
#         probabilities = counts / counts.sum()
        
#         # Create bin labels for display
#         bin_labels = []
#         for i in range(len(bin_edges) - 1):
#             label = f"{bin_edges[i]:.0f}s to {bin_edges[i+1]:.0f}s"
#             bin_labels.append(label)
        
#         # Calculate statistics
#         stats = {
#             'mean': float(data.mean()),
#             'median': float(data.median()),
#             'std': float(data.std()),
#             'min': float(data.min()),
#             'max': float(data.max()),
#             'percentile_25': float(np.percentile(data, 25)),
#             'percentile_75': float(np.percentile(data, 75)),
#             'percentile_95': float(np.percentile(data, 95)),
#             'sample_size': len(data)
#         }
        
#         return {
#             'title': title,
#             'bin_edges': bin_edges.tolist(),
#             'bin_centers': ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
#             'bin_labels': bin_labels,
#             'counts': counts.tolist(),
#             'probabilities': probabilities.tolist(),
#             'statistics': stats
#         }

#     def _calculate_punctuality_metrics(self, delays):
#         """Calculate comprehensive punctuality metrics"""
        
#         try:
#             # Basic statistics
#             mean_delay = float(delays.mean())
#             median_delay = float(delays.median())
#             std_delay = float(delays.std()) if len(delays) > 1 else 0.0
            
#             # Punctuality thresholds (in seconds)
#             thresholds = {
#                 'early': delays < -60,          # More than 1 min early
#                 'on_time': (delays >= -60) & (delays <= 300),  # Within 1 min early to 5 min late
#                 'slightly_late': (delays > 300) & (delays <= 600),  # 5-10 min late
#                 'late': (delays > 600) & (delays <= 1200),          # 10-20 min late
#                 'very_late': delays > 1200      # More than 20 min late
#             }
            
#             # Calculate percentages
#             total_count = len(delays)
#             percentages = {}
#             counts = {}
            
#             for category, condition in thresholds.items():
#                 count = condition.sum()
#                 counts[category] = int(count)
#                 percentages[category] = float(count / total_count * 100)
            
#             # Additional metrics
#             percentiles = {
#                 'p5': float(delays.quantile(0.05)),
#                 'p25': float(delays.quantile(0.25)),
#                 'p75': float(delays.quantile(0.75)),
#                 'p95': float(delays.quantile(0.95))
#             }
            
#             # Performance indicators
#             on_time_performance = percentages['on_time']  # OTP
#             reliability_index = 100 - std_delay / 60  # Simple reliability metric
            
#             return {
#                 'basic_statistics': {
#                     'mean_delay_seconds': mean_delay,
#                     'median_delay_seconds': median_delay,
#                     'std_delay_seconds': std_delay,
#                     'min_delay_seconds': float(delays.min()),
#                     'max_delay_seconds': float(delays.max())
#                 },
#                 'percentiles': percentiles,
#                 'punctuality_categories': {
#                     'counts': counts,
#                     'percentages': percentages
#                 },
#                 'performance_indicators': {
#                     'on_time_performance_percent': on_time_performance,
#                     'reliability_index': max(0, reliability_index),  # Cap at 0 minimum
#                     'punctuality_score': on_time_performance  # Could be enhanced later
#                 },
#                 'sample_size': total_count
#             }
            
#         except Exception as e:
#             print(f"Error calculating punctuality metrics: {e}")
#             return None

# # Clean implementation of Navigation Maps and Master Indexers
#     def create_simplified_navigation_maps(self):
#         """Create cleaner navigation structure"""
        
#         # Basic route→stop→direction mapping
#         basic_route_navigation = self._build_basic_route_navigation()
        
#         # Basic stop→route→direction mapping  
#         basic_stop_navigation = self._build_basic_stop_navigation()
        
#         # Store in simple structure
#         self._navigation_maps = {
#             'stop_to_combinations': basic_stop_navigation,
#             'route_to_combinations': basic_route_navigation,
#             'stop_name_to_stop_ids': self._stop_name_to_stop_ids
#         }
        
#         return self._navigation_maps

#     def _build_basic_stop_navigation(self):
#         """Build basic stop navigation without complex embedded details"""
#         stop_nav = {}
        
#         # Group by stop→route→direction
#         stop_groups = self.df_final.groupby(['stop_id', 'stop_name', 'direction_id'])
        
#         for (stop_id, stop_name, direction_id), group in stop_groups:
#             route_id = str(self.route_id)
            
#             # Initialize nested structure
#             if str(stop_id) not in stop_nav:
#                 stop_nav[str(stop_id)] = {
#                     "stop_name": stop_name,
#                     "routes": {}
#                 }
            
#             if route_id not in stop_nav[str(stop_id)]["routes"]:
#                 stop_nav[str(stop_id)]["routes"][route_id] = {
#                     "route_name": self.route_long_name,
#                     "route_short_name": self.route_short_name,
#                     "directions": {}
#                 }
            
#             # Add direction with simple summary
#             stop_nav[str(stop_id)]["routes"][route_id]["directions"][str(direction_id)] = {
#                 "trip_types": sorted(group['trip_type'].unique().tolist()),
#                 "time_types": sorted(group['time_type'].unique().tolist()),
#                 "total_records": len(group),
#                 "has_violations": self.has_violations_for_combination(stop_id, direction_id, group['trip_type'].iloc[0])
#             }
        
#         return stop_nav

#     def _get_trip_type_details_for_stop(self, stop_id, direction_id):
#         """Get detailed trip type information for a specific stop-direction combination"""
        
#         # Get data for this specific stop-direction
#         stop_direction_data = self.df_final[
#             (self.df_final['stop_id'].astype(str) == str(stop_id)) &
#             (self.df_final['direction_id'].astype(str) == str(direction_id))
#         ]
        
#         if len(stop_direction_data) == 0:
#             return {}
        
#         trip_type_details = {}
        
#         # Analyze each trip type at this stop
#         for trip_type in stop_direction_data['trip_type'].unique():
#             trip_data = stop_direction_data[stop_direction_data['trip_type'] == trip_type]
            
#             # Get time type breakdown
#             time_type_breakdown = trip_data['time_type'].value_counts().to_dict()
#             available_time_types = sorted(time_type_breakdown.keys(),
#                 key=lambda x: ['am_rush', 'day', 'pm_rush', 'night', 'weekend'].index(x) 
#                 if x in ['am_rush', 'day', 'pm_rush', 'night', 'weekend'] else 999)
            
#             # Get violation info from master log indexer if available
#             combo_key = self.get_combination_key(stop_id, direction_id, trip_type)
#             has_violations = False
#             violation_types = []
            
#             if hasattr(self, '_master_log_indexer') and combo_key in self._master_log_indexer:
#                 log_entry = self._master_log_indexer[combo_key]
#                 has_violations = log_entry.get('has_any_violation', False)
                
#                 if log_entry.get('has_topology_violation'):
#                     violation_types.append('topology')
#                 if log_entry.get('has_pattern_violation'):
#                     violation_types.append('pattern')
#                 if log_entry.get('has_regulatory_violation'):
#                     violation_types.append('regulatory')
            
#             # Get pattern description if available
#             pattern_description = trip_type
#             if (hasattr(self, '_trip_types_log') and 
#                 combo_key in self._trip_types_log):
#                 pattern_description = self._trip_types_log[combo_key].get('pattern_description', trip_type)
            
#             # Determine service characteristics
#             service_characteristics = []
#             if len(available_time_types) >= 4:
#                 service_characteristics.append('all_day')
#             elif set(available_time_types).issubset({'am_rush', 'pm_rush'}):
#                 service_characteristics.append('peak_only')
#             elif set(available_time_types).issubset({'day', 'night', 'weekend'}):
#                 service_characteristics.append('off_peak')
#             elif 'weekend' in available_time_types and len(available_time_types) == 1:
#                 service_characteristics.append('weekend_only')
            
#             trip_type_details[trip_type] = {
#                 # Basic info
#                 'total_records': len(trip_data),
#                 'unique_trips': trip_data['trip_id'].nunique(),
#                 'pattern_description': pattern_description,
                
#                 # Time type availability - KEY NAVIGATION INFO
#                 'time_types_available': available_time_types,
#                 'time_type_record_counts': time_type_breakdown,
                
#                 # Violation info (for UI indicators)
#                 'has_violations': has_violations,
#                 'violation_types': violation_types,
#                 'is_regulatory': combo_key in getattr(self, '_regulatory_stops_log', {})
#             }
        
#         return trip_type_details

#     def _get_time_type_details_for_stop(self, stop_id, direction_id):
#         """Get detailed time type information for a specific stop-direction combination"""
        
#         # Get data for this specific stop-direction
#         stop_direction_data = self.df_final[
#             (self.df_final['stop_id'].astype(str) == str(stop_id)) &
#             (self.df_final['direction_id'].astype(str) == str(direction_id))
#         ]
        
#         if len(stop_direction_data) == 0:
#             return {}
        
#         time_type_details = {}
        
#         # Analyze each time type at this stop
#         for time_type in stop_direction_data['time_type'].unique():
#             time_data = stop_direction_data[stop_direction_data['time_type'] == time_type]
            
#             # Get trip type breakdown (which trip types contribute to this time type)
#             trip_type_breakdown = time_data['trip_type'].value_counts().to_dict()
#             contributing_trip_types = sorted(trip_type_breakdown.keys(),
#                 key=lambda x: ['full', 'partial_1', 'partial_2', 'partial_3', 'partial_4', 'partial_5'].index(x) 
#                 if x in ['full', 'partial_1', 'partial_2', 'partial_3', 'partial_4', 'partial_5'] else 999)
            
#             # Check analysis availability for this time type
#             analysis_key = f"{self.route_id}_{stop_id}_{direction_id}_{time_type}"
            
#             histogram_available = False
#             histogram_sample_size = 0
#             histogram_types = []
            
#             punctuality_available = False
#             punctuality_sample_size = 0
#             on_time_performance = None
#             performance_level = None
            
#             if hasattr(self, '_histograms_log') and analysis_key in self._histograms_log:
#                 histogram_available = True
#                 hist_data = self._histograms_log[analysis_key]
#                 histogram_sample_size = hist_data.get('metadata', {}).get('total_sample_size', 0)
#                 histogram_types = list(hist_data.get('histograms', {}).keys())
            
#             if hasattr(self, '_punctuality_log') and analysis_key in self._punctuality_log:
#                 punctuality_available = True
#                 punct_data = self._punctuality_log[analysis_key]
#                 punctuality_sample_size = punct_data.get('metadata', {}).get('sample_size', 0)
#                 on_time_performance = punct_data.get('punctuality_metrics', {}).get('performance_indicators', {}).get('on_time_performance_percent')
                
#                 # Performance level assessment
#                 if on_time_performance is not None:
#                     if on_time_performance >= 85:
#                         performance_level = 'excellent'
#                     elif on_time_performance >= 70:
#                         performance_level = 'good'
#                     elif on_time_performance >= 50:
#                         performance_level = 'fair'
#                     else:
#                         performance_level = 'poor'
            
#             # Check if any contributing trip types have violations
#             violations_summary = {
#                 'any_trip_type_has_violations': False,
#                 'trip_types_with_violations': [],
#                 'violation_types_present': []
#             }
            
#             for trip_type in contributing_trip_types:
#                 combo_key = self.get_combination_key(stop_id, direction_id, trip_type)
                
#                 if hasattr(self, '_master_log_indexer') and combo_key in self._master_log_indexer:
#                     log_entry = self._master_log_indexer[combo_key]
#                     if log_entry.get('has_any_violation', False):
#                         violations_summary['any_trip_type_has_violations'] = True
#                         violations_summary['trip_types_with_violations'].append(trip_type)
                        
#                         # Collect violation types
#                         if log_entry.get('has_topology_violation'):
#                             violations_summary['violation_types_present'].append('topology')
#                         if log_entry.get('has_pattern_violation'):
#                             violations_summary['violation_types_present'].append('pattern')
#                         if log_entry.get('has_regulatory_violation'):
#                             violations_summary['violation_types_present'].append('regulatory')
            
#             # Remove duplicates from violation types
#             violations_summary['violation_types_present'] = list(set(violations_summary['violation_types_present']))
            
#             # Determine time period characteristics
#             time_characteristics = []
#             if time_type == 'am_rush':
#                 time_characteristics.append('morning_peak')
#             elif time_type == 'pm_rush':
#                 time_characteristics.append('evening_peak')
#             elif time_type in ['am_rush', 'pm_rush']:
#                 time_characteristics.append('peak_hour')
#             elif time_type == 'day':
#                 time_characteristics.append('midday')
#             elif time_type == 'night':
#                 time_characteristics.append('late_night')
#             elif time_type == 'weekend':
#                 time_characteristics.append('weekend_service')
            
#             time_type_details[time_type] = {
#                 # Basic info
#                 'total_records': len(time_data),
#                 'unique_trips': time_data['trip_id'].nunique(),
                
#                 # Trip type contributions - KEY INFO for understanding data composition
#                 'contributing_trip_types': contributing_trip_types,
#                 'trip_type_record_counts': trip_type_breakdown
#             }
        
#         return time_type_details

#     def create_master_log_indexer(self):
#         """Create master log indexer - flags based on presence in logs (combination level)"""
#         print("\n=== CREATING MASTER LOG INDEXER (COMBINATION LEVEL) ===")
        
#         master_log_indexer = {}
        
#         # Get all valid combinations at route_stop_direction_trip_type level
#         combinations = self.df_final.groupby([
#             'stop_id', 'stop_name', 'direction_id', 'trip_type'
#         ]).agg({
#             'time_type': lambda x: sorted(list(x.unique())),
#         }).size().reset_index(name='total_records')
        
#         # Add time_types_list column
#         time_types_by_combo = self.df_final.groupby([
#             'stop_id', 'stop_name', 'direction_id', 'trip_type'
#         ])['time_type'].apply(lambda x: sorted(list(x.unique()))).reset_index()
        
#         combinations = combinations.merge(
#             time_types_by_combo, 
#             on=['stop_id', 'stop_name', 'direction_id', 'trip_type']
#         )
#         combinations.columns = [
#             'stop_id', 'stop_name', 'direction_id', 'trip_type', 
#             'total_records', 'time_types_list'
#         ]
        
#         print(f"Creating log indexer for {len(combinations)} combinations")
        
#         for _, combo in combinations.iterrows():
#             stop_id = str(combo['stop_id'])
#             direction_id = str(combo['direction_id'])
#             trip_type = combo['trip_type']
            
#             indexer_key = self.get_combination_key(stop_id, direction_id, trip_type)
            
#             # ===== FLAGS BASED PURELY ON LOG PRESENCE =====
            
#             # Check regulatory status (from regulatory stops log)
#             is_regulatory = indexer_key in getattr(self, '_regulatory_stops_log', {})
            
#             # Check violation flags (from violation logs)
#             has_topology_violation = indexer_key in getattr(self, '_topology_violations_log', {})
#             has_pattern_violation = indexer_key in getattr(self, '_pattern_violations_log', {})
#             has_regulatory_violation = indexer_key in getattr(self, '_regulatory_violations_log', {})
            
#             # Get severity levels (if violations exist)
#             topology_severity = None
#             pattern_severity = None
#             regulatory_severity = None
            
#             if has_topology_violation:
#                 topology_severity = self._topology_violations_log[indexer_key].get('severity')
            
#             if has_pattern_violation:
#                 pattern_severity = self._pattern_violations_log[indexer_key].get('severity')
                
#             if has_regulatory_violation:
#                 regulatory_severity = self._regulatory_violations_log[indexer_key].get('severity')
            
#             # Determine overall severity
#             all_severities = [s for s in [topology_severity, pattern_severity, regulatory_severity] if s]
#             overall_severity = None
#             if all_severities:
#                 if 'high' in all_severities:
#                     overall_severity = 'high'
#                 elif 'medium' in all_severities:
#                     overall_severity = 'medium'
#                 else:
#                     overall_severity = 'low'
            
#             # Create master log indexer entry
#             master_log_indexer[indexer_key] = {
#                 # Basic identifiers
#                 'route_id': str(self.route_id),
#                 'route_short_name': self.route_short_name,
#                 'stop_id': stop_id,
#                 'stop_name': combo['stop_name'],
#                 'direction_id': direction_id,
#                 'trip_type': trip_type,
#                 'total_records': combo['total_records'],
#                 'time_types': combo['time_types_list'],
                
#                 # ===== SINGLE SOURCE OF TRUTH FLAGS =====
#                 # These flags correspond 1:1 with log presence
                
#                 'is_regulatory': is_regulatory,
#                 'has_topology_violation': has_topology_violation,
#                 'has_pattern_violation': has_pattern_violation,
#                 'has_regulatory_violation': has_regulatory_violation,
                
#                 # Additional useful flags
#                 'has_any_violation': has_topology_violation or has_pattern_violation or has_regulatory_violation,
#                 'violation_count': sum([has_topology_violation, has_pattern_violation, has_regulatory_violation]),
#                 'overall_severity': overall_severity,
                
#                 # Severity breakdown
#                 'severity_details': {
#                     'topology_severity': topology_severity,
#                     'pattern_severity': pattern_severity,
#                     'regulatory_severity': regulatory_severity
#                 }
#             }
        
#         self._master_log_indexer = master_log_indexer
        
#         # Statistics
#         total_combinations = len(master_log_indexer)
#         regulatory_count = sum(1 for entry in master_log_indexer.values() if entry['is_regulatory'])
#         violation_count = sum(1 for entry in master_log_indexer.values() if entry['has_any_violation'])
        
#         print(f"Master log indexer created: {total_combinations} combinations")
#         print(f"  - Regulatory combinations: {regulatory_count}")
#         print(f"  - Combinations with violations: {violation_count}")
#         print(f"✅ Flags correspond 1:1 with log presence")
        
#         return master_log_indexer

#     def create_master_analysis_indexer(self):
#         """Create master analysis indexer - flags based on analysis availability (time_type level)"""
#         print("\n=== CREATING MASTER ANALYSIS INDEXER (TIME_TYPE LEVEL) ===")
        
#         master_analysis_indexer = {}
        
#         # Get all time_type level combinations (route_stop_direction_time_type)
#         analysis_combinations = self.df_final.groupby([
#             'stop_id', 'stop_name', 'direction_id', 'time_type'
#         ]).agg({
#             'trip_type': lambda x: sorted(list(x.unique())),
#         }).size().reset_index(name='total_records')
        
#         # Add trip_types_list column  
#         trip_types_by_combo = self.df_final.groupby([
#             'stop_id', 'stop_name', 'direction_id', 'time_type'
#         ])['trip_type'].apply(lambda x: sorted(list(x.unique()))).reset_index()
        
#         analysis_combinations = analysis_combinations.merge(
#             trip_types_by_combo, 
#             on=['stop_id', 'stop_name', 'direction_id', 'time_type']
#         )
#         analysis_combinations.columns = [
#             'stop_id', 'stop_name', 'direction_id', 'time_type', 
#             'total_records', 'trip_types_list'
#         ]
        
#         print(f"Creating analysis indexer for {len(analysis_combinations)} time_type combinations")
        
#         for _, combo in analysis_combinations.iterrows():
#             stop_id = str(combo['stop_id'])
#             direction_id = str(combo['direction_id'])
#             time_type = combo['time_type']
            
#             # Time_type level key format
#             indexer_key = f"{self.route_id}_{stop_id}_{direction_id}_{time_type}"
            
#             # ===== FLAGS BASED PURELY ON ANALYSIS AVAILABILITY =====
            
#             # Check histogram availability
#             has_histograms = indexer_key in getattr(self, '_histograms_log', {})
            
#             # Check punctuality analysis availability
#             has_punctuality = indexer_key in getattr(self, '_punctuality_log', {})
            
#             # Get analysis details if available
#             histogram_types = []
#             histogram_sample_size = 0
#             punctuality_sample_size = 0
#             on_time_performance = None
            
#             if has_histograms:
#                 histogram_data = self._histograms_log[indexer_key]
#                 histogram_types = list(histogram_data.get('histograms', {}).keys())
#                 histogram_sample_size = histogram_data.get('metadata', {}).get('total_sample_size', 0)
            
#             if has_punctuality:
#                 punctuality_data = self._punctuality_log[indexer_key]
#                 punctuality_sample_size = punctuality_data.get('metadata', {}).get('sample_size', 0)
#                 on_time_performance = punctuality_data.get('punctuality_metrics', {}).get('performance_indicators', {}).get('on_time_performance_percent')
            
#             # Performance assessment (if punctuality data available)
#             performance_level = None
#             if on_time_performance is not None:
#                 if on_time_performance >= 85:
#                     performance_level = 'excellent'
#                 elif on_time_performance >= 70:
#                     performance_level = 'good'
#                 elif on_time_performance >= 50:
#                     performance_level = 'fair'
#                 else:
#                     performance_level = 'poor'
            
#             # Create master analysis indexer entry
#             master_analysis_indexer[indexer_key] = {
#                 # Basic identifiers
#                 'route_id': str(self.route_id),
#                 'route_short_name': self.route_short_name,
#                 'stop_id': stop_id,
#                 'stop_name': combo['stop_name'],
#                 'direction_id': direction_id,
#                 'time_type': time_type,
#                 'total_records': combo['total_records'],
#                 'trip_types': combo['trip_types_list'],
                
#                 # ===== SINGLE SOURCE OF TRUTH ANALYSIS FLAGS =====
#                 # These flags correspond 1:1 with analysis log presence
                
#                 'has_histograms': has_histograms,
#                 'has_punctuality': has_punctuality,
#                 'has_any_analysis': has_histograms or has_punctuality,
                
#                 # Analysis details
#                 'analysis_details': {
#                     'histogram_types': histogram_types,
#                     'histogram_sample_size': histogram_sample_size,
#                     'punctuality_sample_size': punctuality_sample_size,
#                     'on_time_performance_percent': on_time_performance,
#                     'performance_level': performance_level
#                 }
#             }
        
#         self._master_analysis_indexer = master_analysis_indexer
        
#         # Statistics
#         total_analysis_combinations = len(master_analysis_indexer)
#         histogram_count = sum(1 for entry in master_analysis_indexer.values() if entry['has_histograms'])
#         punctuality_count = sum(1 for entry in master_analysis_indexer.values() if entry['has_punctuality'])
        
#         print(f"Master analysis indexer created: {total_analysis_combinations} time_type combinations")
#         print(f"  - With histogram analysis: {histogram_count}")
#         print(f"  - With punctuality analysis: {punctuality_count}")
#         print(f"✅ Flags correspond 1:1 with analysis availability")
        
#         return master_analysis_indexer

#     def _sort_navigation_arrays(self, stop_to_combinations, route_to_combinations):
#         """Sort all arrays in navigation structures for consistency"""
        
#         time_order = ['am_rush', 'day', 'pm_rush', 'night', 'weekend']
#         trip_type_order = ['full', 'partial_1', 'partial_2', 'partial_3', 'partial_4', 'partial_5']
        
#         def sort_time_types(time_types_list):
#             return sorted(time_types_list, key=lambda x: time_order.index(x) if x in time_order else 999)
        
#         def sort_trip_types(trip_types_list):
#             return sorted(trip_types_list, key=lambda x: trip_type_order.index(x) if x in trip_type_order else 999)
        
#         # Sort route_to_combinations
#         for route_id in route_to_combinations:
#             if "directions" in route_to_combinations[route_id]:
#                 route_to_combinations[route_id]["directions"].sort()
            
#             for direction_id in route_to_combinations[route_id]:
#                 if direction_id in ["directions", "route_name", "route_short_name"]:
#                     continue
                
#                 if "stop_ids" in route_to_combinations[route_id][direction_id]:
#                     route_to_combinations[route_id][direction_id]["stop_ids"].sort()
                
#                 for stop_id in route_to_combinations[route_id][direction_id].get("stop_ids", []):
#                     if stop_id in route_to_combinations[route_id][direction_id]:
#                         stop_data = route_to_combinations[route_id][direction_id][stop_id]
                        
#                         if "trip_types" in stop_data:
#                             stop_data["trip_types"] = sort_trip_types(stop_data["trip_types"])
#                         if "time_types" in stop_data:
#                             stop_data["time_types"] = sort_time_types(stop_data["time_types"])
        
#         # Sort stop_to_combinations (same logic)
#         for stop_id in stop_to_combinations:
#             if "routes" in stop_to_combinations[stop_id]:
#                 stop_to_combinations[stop_id]["routes"].sort()
            
#             for route_id in stop_to_combinations[stop_id]:
#                 if route_id in ["routes", "stop_name"]:
#                     continue
                
#                 if "directions" in stop_to_combinations[stop_id][route_id]:
#                     stop_to_combinations[stop_id][route_id]["directions"].sort()
                
#                 for direction_id in stop_to_combinations[stop_id][route_id]:
#                     if direction_id in ["directions", "route_name", "route_short_name"]:
#                         continue
                    
#                     direction_data = stop_to_combinations[stop_id][route_id][direction_id]
                    
#                     if "trip_types" in direction_data:
#                         direction_data["trip_types"] = sort_trip_types(direction_data["trip_types"])
#                     if "time_types" in direction_data:
#                         direction_data["time_types"] = sort_time_types(direction_data["time_types"])

#     def export_all_data(self):
#         """Export all data as JSON files with simple global merging"""
#         print("\n=== EXPORTING ALL DATA ===")
        
#         # Simple JSON serializer
#         def clean_for_json(obj):
#             """Convert objects to JSON-serializable format"""
#             if hasattr(obj, 'tolist'):
#                 return obj.tolist()
#             elif hasattr(obj, 'item'):
#                 return obj.item()
#             elif isinstance(obj, (dict, list)):
#                 return obj
#             elif isinstance(obj, set):
#                 return list(obj)
#             else:
#                 return str(obj)
        
#         # Simple file operations
#         def load_json(file_path):
#             """Load existing JSON file or return empty dict"""
#             try:
#                 if file_path.exists():
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         return json.load(f)
#             except Exception as e:
#                 print(f"Warning: Could not load {file_path.name}: {e}")
#             return {}
        
#         def save_json(data, file_path):
#             """Save data as JSON file"""
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 json.dump(data, f, indent=2, ensure_ascii=False, default=clean_for_json)
        
#         def merge_route_data(existing, new, route_id):
#             """Merge route data by removing old route entries and adding new ones"""
#             # Remove old entries for this route
#             cleaned = {
#                 key: value for key, value in existing.items()
#                 if not (key.startswith(f"{route_id}_") or 
#                     (isinstance(value, dict) and str(value.get('route_id')) == str(route_id)))
#             }
#             # Add new entries
#             cleaned.update(new)
#             return cleaned
        
#         def merge_navigation(existing, new):
#             """Simple navigation merge - just update with new data"""
#             if not existing:
#                 return new
            
#             # Deep merge for nested structures
#             result = existing.copy()
#             for key, value in new.items():
#                 if key in result and isinstance(result[key], dict) and isinstance(value, dict):
#                     result[key].update(value)
#                 else:
#                     result[key] = value
#             return result
        
#         # Setup output folder
#         output_folder = Path('transit_analysis_global')
#         output_folder.mkdir(exist_ok=True)
        
#         # Define what to export
#         log_files = {
#             'master_log_indexer.json': self._master_log_indexer,
#             'master_analysis_indexer.json': getattr(self, '_master_analysis_indexer', {}),
#             'log_topology_violations.json': self._topology_violations_log,
#             'log_pattern_violations.json': self._pattern_violations_log,
#             'log_regulatory_violations.json': self._regulatory_violations_log,
#             'log_trip_types.json': self._trip_types_log,
#             'log_regulatory_stops.json': self._regulatory_stops_log,
#             'log_histograms.json': self._histograms_log,
#             'log_punctuality.json': self._punctuality_log,
#         }
        
#         navigation_files = {
#             'global_stop_to_combinations.json': self._navigation_maps.get('stop_to_combinations', {}),
#             'global_route_to_combinations.json': self._navigation_maps.get('route_to_combinations', {}),
#             'global_stop_name_to_stop_ids.json': self._navigation_maps.get('stop_name_to_stop_ids', {}),
#             'global_route_short_name_to_info.json': self._create_route_mapping()
#         }
        
#         exported_files = []
        
#         # Export log files (with route-specific merging)
#         for filename, data in log_files.items():
#             if not data:  # Skip empty data
#                 continue
                
#             file_path = output_folder / filename
#             existing = load_json(file_path)
#             merged = merge_route_data(existing, data, self.route_id)
#             save_json(merged, file_path)
#             exported_files.append(filename)
            
#             print(f"✅ {filename}: {len(data)} entries")
        
#         # Export navigation files (accumulative merging)
#         for filename, data in navigation_files.items():
#             if not data:
#                 continue
                
#             file_path = output_folder / filename
#             existing = load_json(file_path)
#             merged = merge_navigation(existing, data)
#             save_json(merged, file_path)
#             exported_files.append(filename)
            
#             print(f"✅ {filename}: Navigation data")
        
#         # Create simple summary
#         summary = {
#             'data_summary': {
#                 'total_routes_analyzed': 1,  # Will be aggregated when multiple routes processed
#                 'total_stop_names': len(self._stop_name_to_stop_ids),
#                 'total_unique_stop_ids': len(set(entry['stop_id'] for entry in self._master_log_indexer.values())),
#                 'total_combinations': len(self._master_log_indexer),
#                 'has_topology_violations': len(self._topology_violations_log) > 0,
#                 'has_pattern_violations': len(self._pattern_violations_log) > 0,
#                 'has_regulatory_violations': len(self._regulatory_violations_log) > 0,
#                 'has_any_violations': any([
#                     len(self._topology_violations_log) > 0,
#                     len(self._pattern_violations_log) > 0,
#                     len(self._regulatory_violations_log) > 0
#                 ])
#             },
#             'files_exported': exported_files,
#             'route_info': {
#                 'route_id': self.route_id,
#                 'route_short_name': self.route_short_name,
#                 'route_long_name': self.route_long_name
#             }
#         }
        
#         # Simple summary merging (just aggregate counts)
#         summary_file = output_folder / 'global_summary.json'
#         existing_summary = load_json(summary_file)
        
#         if existing_summary:
#             # Aggregate the counts
#             existing_data = existing_summary.get('data_summary', {})
#             new_data = summary['data_summary']
            
#             summary['data_summary'] = {
#                 'total_routes_analyzed': existing_data.get('total_routes_analyzed', 0) + 1,
#                 'total_stop_names': existing_data.get('total_stop_names', 0) + new_data['total_stop_names'],
#                 'total_unique_stop_ids': existing_data.get('total_unique_stop_ids', 0) + new_data['total_unique_stop_ids'],
#                 'total_combinations': existing_data.get('total_combinations', 0) + new_data['total_combinations'],
#                 'has_topology_violations': existing_data.get('has_topology_violations', False) or new_data['has_topology_violations'],
#                 'has_pattern_violations': existing_data.get('has_pattern_violations', False) or new_data['has_pattern_violations'],
#                 'has_regulatory_violations': existing_data.get('has_regulatory_violations', False) or new_data['has_regulatory_violations'],
#                 'has_any_violations': existing_data.get('has_any_violations', False) or new_data['has_any_violations']
#             }
        
#         save_json(summary, summary_file)
        
#         # Clean output
#         total_combinations = summary['data_summary']['total_combinations']
#         has_violations = summary['data_summary']['has_any_violations']
        
#         print(f"\n🌍 EXPORT COMPLETE:")
#         print(f"  - Route: {self.route_short_name}")
#         print(f"  - Combinations: {total_combinations}")
#         print(f"  - Issues: {'Yes' if has_violations else 'No'}")
#         print(f"  - Files: {output_folder}")
        
#         return {
#             'output_folder': str(output_folder),
#             'summary_file': str(summary_file),
#             'combinations': total_combinations,
#             'files_exported': exported_files
#         }


#     def has_violations_for_combination(self, stop_id, direction_id, trip_type):
#         """Check if combination has any violations"""
#         combo_key = self.get_combination_key(stop_id, direction_id, trip_type)
#         return any([
#             combo_key in self._topology_violations_log,
#             combo_key in self._pattern_violations_log,
#             combo_key in self._regulatory_violations_log
#         ])

#     def has_analysis_for_combination(self, stop_id, direction_id, time_type):
#         """Check if combination has analysis data"""
#         analysis_key = f"{self.route_id}_{stop_id}_{direction_id}_{time_type}"
#         return any([
#             analysis_key in self._histograms_log,
#             analysis_key in self._punctuality_log
#         ])
