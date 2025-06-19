import pandas as pd
import json as json 
from pathlib import Path
from hashlib import md5
import numpy as np
from collections import defaultdict, Counter
from statistical.lv_logger import LVLogger
import matplotlib.pyplot as plt
import os

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
    
def sanitize_values(obj):
    """Recursively convert NumPy/Pandas types to native Python types."""
    if isinstance(obj, dict):
        return {sanitize_values(k): sanitize_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_values(item) for item in obj]
    elif isinstance(obj, (np.integer, pd.Int64Dtype)):
        return int(obj)
    elif isinstance(obj, (np.floating, pd.Float64Dtype)):
        return float(obj)
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

def save_json_safe(data, path):
    """Safely save sanitized data to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_values(sanitize_keys(data)), f, indent=2, ensure_ascii=False)

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
        self.route_info = {
            "route_id": str(self.route_id),
            "route_name": self.route_long_name,
            "route_short_name": self.route_short_name
        }

        self.log = LVLogger(self.route_info)

        # STEP 0: Preprocessing
        self.df_before = self.prepare_columns(raw_data)

        # STEP 1: Stop topology validation
        print("STEP 1: Stop topology validation...")
        self.create_and_validate_stop_topology(self.df_before)

        # STEP 2: Direction topology validation
        print("STEP 2: Direction topology validation...")
        self.create_and_validate_direction_topology(self.df_before)

        # STEP 3: Regulatory stop detection
        print("STEP 3: Regulatory stop detection...")
        self.df_with_regulatory = self.identify_and_classify_stops(self.df_before)

        # STEP 4: Performance metrics
        print("STEP 4: Performance analysis...")
        self.df_final = self.df_with_regulatory.copy()
        self.generate_performance_logs()
        self.generate_travel_time_log()

        # STEP 6: Export everything
        export_dir = "exported_logs"
        self.export_all_logs_frontend_friendly(export_dir=export_dir)
        self.export_summary_dashboard_files(export_dir=export_dir)

        print("✅ DataFormer initialization complete!")

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
        
        # Sort once and get the first scheduled_departure_time per trip
        trip_start_times = (
            df.sort_values('stop_sequence')
            .groupby(['trip_id', 'direction_id', 'start_date'])['scheduled_departure_time']
            .first()
            .reset_index(name='start_time')
        )

        # Merge back to original DataFrame
        df = df.merge(trip_start_times, on=['trip_id', 'direction_id', 'start_date'], how='left')

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
    
#   ================ CLEARLY SEPARATED VALIDATION METHODS =====================

    def create_and_validate_stop_topology(self, df):
        """Main topology validation method for 2-layer approach using LVLogger."""
        print("\n=== COMPLETE STOP VALIDATION WORKFLOW ===")

        # STEP 1: Validate and log parent station topology
        self._classify_all_parent_stations_diversity(df)

        # STEP 2: Validate and log stop ID direction behavior
        self._validate_stop_id_directions(df)

        # Build navigational hierarchies based on logged labels and violations
        self.build_topology_hierarchy("stop", df)

        print("✅ Topology validation complete and logs stored.")

    def _classify_all_parent_stations_diversity(self, df):
        print("Validating parent station topology...")

        # Replace missing or invalid parent_station with stop_name (assumes 1:1 mapping)
        df = df.copy()
        df['normalized_parent'] = df.apply(
            lambda row: str(row['parent_station']) if pd.notna(row['parent_station']) and str(row['parent_station']).lower() not in {'none', 'nan'} else str(row['stop_name']),
            axis=1
        )

        for parent_station in df['normalized_parent'].unique():
            self._analyze_single_parent_station_topology(df, str(parent_station), parent_col='normalized_parent')

    def _analyze_single_parent_station_topology(self, df, parent_station, parent_col='parent_station'):
        parent_data = df[df[parent_col].astype(str) == parent_station]

        stop_ids = parent_data['stop_id'].unique().tolist()
        stop_names = parent_data['stop_name'].unique().tolist()

        stop_id_analysis = {}
        missing_data_stop_ids = []

        for stop_id in stop_ids:
            directions, direction_counts = self._get_stop_id_directions(df, stop_id, parent_station)
            is_multi = len(directions) > 1
            stop_id_analysis[str(stop_id)] = {
                'directions': directions,
                'direction_counts': direction_counts,
                'is_multi_directional': is_multi
            }
            if len(directions) == 0:
                missing_data_stop_ids.append(stop_id)

        total_stops = len(stop_ids)
        multi_count = sum(1 for analysis in stop_id_analysis.values() if analysis['is_multi_directional'])
        single_count = total_stops - multi_count

        def log_and_return(label, parent_violation=None):
            self.log.add_label('stop_topology', 'parent_station', parent_station, label)

            if parent_violation:
                self.log.add_violation('stop_topology', 'parent_station', parent_violation)
                print(f"🚩 Parent station {parent_station} [{label}]: {parent_violation['description']}")

                for sid in stop_ids:
                    stop_violation = self.log.create_violation_entry(
                        'stop_id_from_invalid_parent_topology',
                        'medium',
                        'Stop is part of parent_station with invalid topology',
                        parent_station=parent_station,
                        stop_id=sid,
                        details=stop_id_analysis[sid]
                    )
                    self.log.add_violation('stop_topology', 'stop_id', stop_violation)
            else:
                print(f"✅ Parent station {parent_station} [{label}]: Valid topology")

            return label

        # ⚠️ Incomplete data check
        if missing_data_stop_ids:
            warning = self.log.create_violation_entry(
                'stop_id_missing_direction_data',
                'medium',
                f'Some stop_ids in parent_station {parent_station} have no direction data',
                parent_station=parent_station,
                stop_ids=missing_data_stop_ids,
                details=stop_id_analysis
            )
            return log_and_return('Undefined', warning)

        # Classification logic
        if total_stops == 1:
            return log_and_return('Shared' if multi_count == 1 else 'Unidirectional')

        if total_stops == 2:
            if single_count == 2 and multi_count == 0:
                return log_and_return('Bidirectional')
            else:
                violation = self.log.create_violation_entry(
                    'two_stop_misassigned_directions',
                    'high',
                    f'2-stop station with wrong configuration: {single_count} single + {multi_count} multi (expected: 2 single + 0 multi)',
                    parent_station=parent_station,
                    details={
                        'expected': '2 single-directional, 0 multi-directional',
                        'actual': f'{single_count} single-directional, {multi_count} multi-directional',
                        'directional_analysis': stop_id_analysis
                    }
                )
                return log_and_return('Undefined', violation)

        if total_stops % 2 == 0:
            if single_count == total_stops and multi_count == 0:
                return log_and_return('Bidirectional')
            elif single_count % 2 == 0 and multi_count % 2 == 0:
                return log_and_return('Hybrid')
            else:
                violation = self.log.create_violation_entry(
                    'unpaired_directional_stops',
                    'medium',
                    f'Even-stop station with unpaired configuration: {single_count} single + {multi_count} multi',
                    parent_station=parent_station,
                    stop_names=stop_names,
                    details={
                        'expected': 'even single-directional, even multi-directional',
                        'actual': f'{single_count} single-directional, {multi_count} multi-directional',
                        'directional_analysis': stop_id_analysis
                    }
                )
                return log_and_return('Undefined', violation)

        if single_count % 2 == 0 and multi_count % 2 == 1:
            return log_and_return('Hybrid')
        else:
            violation = self.log.create_violation_entry(
                'unpaired_directional_stops',
                'medium',
                f'Odd-stop station with unpaired configuration: {single_count} single + {multi_count} multi',
                parent_station=parent_station,
                stop_names=stop_names,
                details={
                    'expected': 'even single-directional, odd multi-directional',
                    'actual': f'{single_count} single-directional, {multi_count} multi-directional',
                    'directional_analysis': stop_id_analysis
                }
            )
            return log_and_return('Undefined', violation)

    def _validate_stop_id_directions(self, df):
        """STEP 2: Validate and log individual stop ID direction behavior."""
        print("Validating individual stop ID directions...")

        stop_id_data = df[['stop_id', 'stop_name', 'parent_station']].drop_duplicates()

        for _, row in stop_id_data.iterrows():
            stop_id = row['stop_id']
            stop_name = row['stop_name']
            parent_station = row['parent_station']

            # Analyze this specific stop_id
            directions, direction_counts = self._get_stop_id_directions(df, stop_id, parent_station)

            # Assign label based on direction count
            if len(directions) > 1:
                label = 'multi_directional'
            elif len(directions) == 1:
                label = 'single_directional'
            else:
                label = 'no_data'

            self.log.add_label('stop_topology', 'stop_id', stop_id, label)

            # Always check for violation if no directions
            if len(directions) == 0:
                violation = self.log.create_violation_entry(
                    'stop_id_without_direction_data',
                    'high',
                    f'Stop id: {stop_id} has no direction data',
                    stop_id=stop_id,
                    stop_name=stop_name,
                    parent_station=parent_station,
                    details={
                        'direction_details': {
                            'directions': [],
                            'direction_counts': {},
                            'is_multi_directional': False
                        }
                    }
                )
                self.log.add_violation('stop_topology', 'stop_id', violation)
                print(f"🚩 Stop ID {stop_id}: No direction data")

        print("✅ Stop ID validation complete.")

    def _get_stop_id_directions(self, df, stop_id, parent_station) -> tuple[list, dict]:
        """Helper: Get direction information for a specific stop_id within a parent station"""
        stop_data = df[
            (df['stop_id'].astype(str) == str(stop_id)) & 
            (df['parent_station'].astype(str) == str(parent_station))
        ]

        if stop_data.empty:
            violation = self.log.create_violation_entry(
                'stop_id_missing_from_schedule',
                'medium',
                f"No data found for stop_id {stop_id} in parent_station {parent_station}",
                stop_id=stop_id,
                parent_station=parent_station
            )
            self.log.add_violation('stop_topology', 'stop_id', violation)
            print(f"⚠️ {violation['description']}")
            return [], {}

        directions = stop_data['direction_id'].unique().tolist()
        direction_counts = stop_data['direction_id'].value_counts().to_dict()

        return directions, direction_counts

#   ================ DIRECTION TOPOLOGY VALIDATION (following stop pattern) =====================

    def create_and_validate_direction_topology(self, df):
        print("\n=== COMPLETE DIRECTION VALIDATION WORKFLOW ===")

        # STEP 1: Analyze all directions and extract pattern/coverage issues, now logged directly
        self._analyze_all_directions(df)

        # STEP 2: Build final hierarchies using logs + cached canonical patterns
        self.build_topology_hierarchy("direction", df)

        print(f"✅ Direction validation complete")

    def _analyze_all_directions(self, df):
        grouped = df.groupby("direction_id")
        for direction_id, group in grouped:
            direction_id = str(direction_id)
            self._analyze_single_direction(group, direction_id)

    def _analyze_single_direction(self, df, direction_id):
        trip_groups = df.groupby(['trip_id', 'start_date'])
        pattern_instance_counter = Counter()
        pattern_map = {}  # Maps pattern tuple to raw DataFrame

        # Step 1: Collect all patterns
        for _, trip_df in trip_groups:
            sorted_trip = trip_df.sort_values("stop_sequence")
            pattern = tuple(sorted_trip['stop_id'])
            pattern_instance_counter[pattern] += 1
            pattern_map[pattern] = sorted_trip

        if not pattern_instance_counter:
            self.log.add_label("direction_topology", "direction", direction_id, "no_data")
            return

        # Step 2: Find canonical from longest strictly valid pattern (start from 1, diff == 1)
        valid_candidates = []
        for pattern in pattern_instance_counter:
            trip_df = pattern_map[pattern]
            stop_seqs = trip_df['stop_sequence'].tolist()
            if stop_seqs and stop_seqs[0] == 1 and np.all(np.diff(stop_seqs) == 1):
                valid_candidates.append((pattern, len(pattern)))

        if not valid_candidates:
            self.log.add_label("direction_topology", "direction", direction_id, "no_valid_canonical")
            return

        canonical = max(valid_candidates, key=lambda x: x[1])[0]
        canonical_str = self.convert_pattern_to_position_string(list(canonical), list(canonical))
        canonical_count = pattern_instance_counter[canonical]

        # Cache canonical
        self.log.direction_topology_logs['metadata']['canonical_patterns'][direction_id] = list(canonical)

        # Step 3: Compare other patterns to canonical
        alt_counter = Counter()
        for pattern, count in pattern_instance_counter.items():
            if pattern == canonical:
                continue
            alt_str = self.convert_pattern_to_position_string(list(pattern), list(canonical))
            alt_counter[alt_str] += count

        # Step 4: Label and log direction
        label = "Multiple Patterns Detected" if alt_counter else "Full Route Only"
        self.log.add_label("direction_topology", "direction", direction_id, label)

        if alt_counter:
            v = self.log.create_violation_entry(
                "diverse_patterns_in_direction", "low",
                f"Direction {direction_id} has {len(alt_counter)} alternative patterns.",
                direction_id=direction_id,
                details={
                    "canonical_pattern": {canonical_str: canonical_count},
                    "alternative_patterns": dict(alt_counter)
                }
            )
            self.log.add_violation("direction_topology", "direction", v)

        # Step 5: Compute stop coverage for canonical
        total_trips = sum(pattern_instance_counter.values())
        stop_counts = defaultdict(int)
        for pattern, count in pattern_instance_counter.items():
            for sid in pattern:
                stop_counts[sid] += count

        stop_name_map = self.log.navigation_structures['stop_hierarchies']['by_stop_id']
        stop_index_map = {sid: idx + 1 for idx, sid in enumerate(canonical)}

        for sid in canonical:
            sid_str = str(sid)
            count = stop_counts.get(sid, 0)
            percent_missing = round(100 - (count / total_trips * 100), 1) if total_trips > 0 else 0.0

            label = "varying_coverage" if percent_missing > 5 else "high_coverage"
            self.log.add_label("direction_topology", "stop_id", sid_str, {direction_id: label})

            if percent_missing > 5:
                stop_name = stop_name_map.get(sid_str, {}).get("stop_name", "Unknown")
                position = stop_index_map.get(sid, "?")
                v = self.log.create_violation_entry(
                    "missing_stop_coverage", "medium",
                    f"Stop {stop_name}, with stop_id {sid_str} in canonical position {position} "
                    f"is missing from {percent_missing:.1f}% of trips in direction {direction_id}",
                    stop_id=sid_str,
                    stop_name=stop_name,
                    direction_id=direction_id,
                    details={
                        'canonical_position': position,
                        'missing_count': total_trips - count,
                        'total_trips': total_trips,
                        'missing_percentage': percent_missing
                    }
                )
                self.log.add_violation("direction_topology", "stop_id", v)

    def convert_pattern_to_position_string(self, pattern, canonical):
        canonical_index = {stop_id: idx + 1 for idx, stop_id in enumerate(canonical)}
        canonical_pos = {stop_id: idx for idx, stop_id in enumerate(canonical)}

        segments = []
        i = 0
        while i < len(pattern):
            stop_id = pattern[i]
            if stop_id not in canonical_index:
                segments.append("*")
                i += 1
                continue

            start = canonical_index[stop_id]
            end = start
            current_pos = canonical_pos[stop_id]

            while (
                i + 1 < len(pattern)
                and pattern[i + 1] in canonical_pos
                and canonical_pos[pattern[i + 1]] == current_pos + 1
            ):
                i += 1
                current_pos += 1
                end = canonical_index[pattern[i]]

            segments.append(f"{start}" if start == end else f"{start}-{end}")
            i += 1
            if i < len(pattern) and pattern[i] in canonical_index:
                segments.append("_")

        return "".join(segments)

#   ============================== NAVIGATIONAL HIERARCHIES ===================================

    def build_topology_hierarchy(self, entity_type: str, df: pd.DataFrame = None):
        """
        Generalized builder for stop or direction topology hierarchies.
        
        Parameters:
        - entity_type: either 'stop' or 'direction'
        - df: Required for stop topology (to extract stop_name and parent_station)
        """
        assert entity_type in {"stop", "direction"}

        if entity_type == "stop":
            stop_hier = self.log.stop_topology_logs
            stop_id_labels = stop_hier.get("stop_id_labels", {})
            parent_labels = stop_hier.get("parent_station_labels", {})
            stop_violations = stop_hier.get("stop_id_violations", {}).values()
            parent_violations = stop_hier.get("parent_station_violations", {}).values()

            # Extract violating stop_ids
            violating_stop_ids = {
                str(v["stop_id"]) for v in list(stop_violations) + list(parent_violations) if "stop_id" in v
            }

            stop_name_hierarchy = {}
            stop_id_hierarchy = {}
            stop_to_parent = {}
            parent_to_stops = defaultdict(list)
            parent_has_violation = defaultdict(bool)

            def normalize_parent(stop_name, parent_station):
                if pd.isna(parent_station) or str(parent_station).lower() in {"none", "nan"}:
                    return str(stop_name)
                return str(parent_station)

            for _, row in df.drop_duplicates(["stop_id", "stop_name", "parent_station"]).iterrows():
                stop_id = str(row["stop_id"])
                stop_name = str(row["stop_name"]) if pd.notna(row["stop_name"]) else "UNKNOWN_STOP"
                parent_station = str(row["parent_station"]) if pd.notna(row["parent_station"]) else ""

                normalized_parent = normalize_parent(stop_name, parent_station)
                stop_to_parent[stop_id] = (stop_name, normalized_parent)
                parent_to_stops[(stop_name, normalized_parent)].append(stop_id)

                label = stop_id_labels.get(stop_id, "Unknown")
                stop_has_violation = stop_id in violating_stop_ids

                stop_id_hierarchy[stop_id] = {
                    "parent_station": normalized_parent,
                    "stop_name": stop_name,
                    "label": label,
                    "has_violation": stop_has_violation,
                }

                if stop_has_violation:
                    parent_has_violation[(stop_name, normalized_parent)] = True

            for (stop_name, normalized_parent), stop_ids in parent_to_stops.items():
                inherited_violation = parent_has_violation.get((stop_name, normalized_parent), False)
                label = parent_labels.get(normalized_parent, "Unknown")

                for sid in stop_ids:
                    stop_id_hierarchy[sid]["has_violation"] |= inherited_violation

                stop_name_hierarchy.setdefault(stop_name, {})[normalized_parent] = {
                    "stop_ids": stop_ids,
                    "label": label,
                    "has_violation": inherited_violation
                }

            self.log.navigation_structures["stop_hierarchies"] = {
                "by_stop_name": stop_name_hierarchy,
                "by_stop_id": stop_id_hierarchy
            }

        elif entity_type == "direction":  # direction
            dir_logs = self.log.direction_topology_logs
            dir_labels = dir_logs.get("direction_labels", {})
            stop_labels = dir_logs.get("stop_id_labels", {})
            canonical_patterns = dir_logs.get("metadata", {}).get("canonical_patterns", {})
            dir_violations = dir_logs.get("direction_violations", {}).values()
            stop_violations = dir_logs.get("stop_id_violations", {}).values()

            violating_directions = {str(v["direction_id"]) for v in dir_violations if "direction_id" in v}
            violating_stops = {str(v["stop_id"]) for v in stop_violations if "stop_id" in v}

            by_direction = {}
            by_stop_id = {}

            for direction_id, canonical in canonical_patterns.items():
                direction_id = str(direction_id)
                canonical_pattern_str = self.convert_pattern_to_position_string(canonical, canonical)

                stop_ids = []
                has_violation = direction_id in violating_directions

                for idx, sid in enumerate(canonical):
                    sid = str(sid)
                    label = stop_labels.get(sid, {}).get(direction_id, "Unknown")
                    stop_ids.append({idx + 1: {"stop_id": sid, "label": label}})

                    by_stop_id.setdefault(sid, {
                        "label": {},
                        "direction_ids": [],
                        "has_violation": False
                    })

                    by_stop_id[sid]["label"][direction_id] = label
                    by_stop_id[sid]["direction_ids"].append(direction_id)
                    by_stop_id[sid]["has_violation"] |= sid in violating_stops

                by_direction[direction_id] = {
                    "label": dir_labels.get(direction_id, "Unknown"),
                    "canonical_pattern_string_description": canonical_pattern_str,
                    "stop_ids": stop_ids,
                    "has_violation": has_violation
                }

            self.log.navigation_structures["direction_hierarchies"] = {
                "by_direction": by_direction,
                "by_stop_id": by_stop_id
            }
        else:
            return None

#   ======================== REGULATORY LABELS AND VIOLATIONS, HISTOGRAMS AND PUNCTUALITY DIAGRAMS ================================
    def generate_performance_logs(self):
        """Generate histograms (total + incremental), punctuality metrics, and evaluate regulatory stop performance."""
        print("\n=== GENERATING PERFORMANCE HISTOGRAMS AND PUNCTUALITY METRICS ===")

        df = self.calculate_travel_times_and_delays(self.df_final.copy())

        histograms_log = {}
        punctuality_log = {}

        group_cols = ['direction_id', 'stop_id', 'stop_name', 'time_type']

        for (direction_id, stop_id, stop_name, time_type), group in df[df['valid_segment']].groupby(group_cols):
            delay_data = group['departure_delay'].dropna()
            incr_data = group['incremental_delay'].dropna()

            if len(delay_data) >= 5:
                combined_data = pd.concat([delay_data, incr_data])
                bin_edges = np.histogram_bin_edges(combined_data, bins='fd')

                total_histogram = self._create_normalized_histogram(delay_data, bin_edges)
                incr_histogram = self._create_normalized_histogram(incr_data, bin_edges)

                if total_histogram or incr_histogram:
                    key = f"stop_id_{stop_id}_direction_{direction_id}_time_{time_type}"
                    histograms_log[key] = {
                        'route_id': self.route_id,
                        'direction_id': direction_id,
                        'stop_id': stop_id,
                        'stop_name': stop_name,
                        'time_type': time_type,
                        'total_delay_histogram': total_histogram,
                        'incremental_delay_histogram': incr_histogram
                    }

            if len(delay_data) >= 5:
                punctuality = self._calculate_punctuality_metrics(delay_data)
                if punctuality:
                    key = f"stop_id_{stop_id}_direction_{direction_id}_time_{time_type}"
                    punctuality_log[key] = {
                        'route_id': self.route_id,
                        'stop_id': stop_id,
                        'stop_name': stop_name,
                        'direction_id': direction_id,
                        'time_type': time_type,
                        'punctuality': punctuality
                    }

        self.log.performance_logs['histograms_stops'] = histograms_log
        self.log.performance_logs['punctuality_barcharts'] = punctuality_log

        print(f"  ✅ {len(histograms_log)} histograms and {len(punctuality_log)} punctuality entries created.")

        # === Evaluate performance of stops labeled as regulatory ===
        print("  🧪 Evaluating regulatory stop performance...")
        self.analyze_and_classify_regulatory_behavior()

        reg_labels = self.log.performance_logs.get("labels", {}).get("stops_regulatory", {})
        all_punctuality = self.log.performance_logs.get('punctuality_barcharts', {})
        underperforming = 0
        for stop_id, directions in reg_labels.items():
            for direction_id in directions:
                key = f"stop_id_{stop_id}_direction_{direction_id}_time_all_day"
                punctuality_entry = all_punctuality.get(key)

                if punctuality_entry:
                    ratio = punctuality_entry['punctuality'].get('on_time_ratio', 1.0)
                    if ratio < 0.80:
                        underperforming +=1
                        violation = self.log.create_violation_entry(
                            violation_type='regulatory_stop_underperforming',
                            severity='medium',
                            description=f"Regulatory stop {stop_id} in direction {direction_id} has only {ratio:.1%} on-time departures.",
                            stop_id=stop_id,
                            stop_name=punctuality_entry.get('stop_name'),
                            direction_id=direction_id,
                            on_time_ratio=round(ratio, 3)
                        )
                        self.log.add_violation("performance", "regulatory_performance", violation)

        print(f"  ⚠️  Underperforming regulatory stops: {underperforming}")

    def calculate_travel_times_and_delays(self, df):
        """Calculate incremental travel times and delays between consecutive stops."""
        print("\n=== CALCULATING TRAVEL TIMES AND DELAYS ===")

        df = df.sort_values(['trip_id', 'direction_id', 'start_date', 'stop_sequence']).copy()
        group_cols = ['trip_id', 'direction_id', 'start_date']

        # Identify rows with consecutive stop_sequence = 1 gap
        df['prev_stop_sequence'] = df.groupby(group_cols)['stop_sequence'].shift(1)
        df['valid_segment'] = (df['stop_sequence'] - df['prev_stop_sequence']) == 1

        # Delay computation
        df['prev_delay'] = df.groupby(group_cols)['departure_delay'].shift(1)
        df['incremental_delay'] = df['departure_delay'] - df['prev_delay']

        # Time computation
        for col in ['scheduled_departure_time', 'observed_departure_time']:
            if col in df.columns:
                prefix = col.split('_')[0]
                df[f'prev_{prefix}_departure_time'] = df.groupby(group_cols)[col].shift(1)
                df[f'{prefix}_travel_time'] = df[col] - df[f'prev_{prefix}_departure_time']
                df.loc[~df['valid_segment'], f'{prefix}_travel_time'] = pd.NaT

        df.loc[~df['valid_segment'], 'incremental_delay'] = np.nan

        print(f"  ✅ {df['valid_segment'].sum()} valid travel segments out of {len(df)}")
        return df

    def _create_normalized_histogram(self, data, bin_edges=None, bins='fd'):
        if len(data) < 5:
            return None

        if bin_edges is None:
            bin_edges = np.histogram_bin_edges(data, bins=bins)

        counts, _ = np.histogram(data, bins=bin_edges)
        probabilities = counts / counts.sum()
        bin_labels = [f"{int(bin_edges[i])}s to {int(bin_edges[i+1])}s" for i in range(len(bin_edges) - 1)]

        return {
            'bin_edges': bin_edges.tolist(),
            'bin_centers': ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
            'bin_labels': bin_labels,
            'counts': counts.tolist(),
            'probabilities': probabilities.tolist(),
            'statistics': {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'percentile_25': float(np.percentile(data, 25)),
                'percentile_75': float(np.percentile(data, 75)),
                'percentile_95': float(np.percentile(data, 95)),
                'sample_size': len(data)
            }
        }

    def _calculate_punctuality_metrics(self, delays):
        try:
            total = len(delays)
            thresholds = {
                'too_early': delays < -30,
                'on_time': (delays >= -30) & (delays <= 179),
                'too__late': delays > 179
            }
            counts = {k: int(v.sum()) for k, v in thresholds.items()}
            percentages = {k: round(c / total * 100, 2) for k, c in counts.items()}

            return {
                'basic_statistics': {
                    'mean_delay': float(delays.mean()),
                    'median_delay': float(delays.median()),
                    'std_delay': float(delays.std()),
                    'min_delay': float(delays.min()),
                    'max_delay': float(delays.max())
                },
                'punctuality_distribution': {
                    'counts': counts,
                    'percentages': percentages
                },
                'sample_size': total
            }
        except Exception as e:
            print(f"Error in punctuality metric calculation: {e}")
            return None

    def get_punctuality_flow(self, direction_id: str, time_type: str = "all_day"):
        """
        Return a list of punctuality metrics for each stop in canonical order for given direction.
        """
        flow = []

        # Get canonical stop sequence
        pattern_dict = self.log.direction_topology_logs['metadata']['canonical_patterns']
        stop_sequence = pattern_dict.get(direction_id, [])

        for stop_id in stop_sequence:
            key = f"stop_id_{stop_id}_direction_{direction_id}_time_{time_type}"
            entry = self.log.performance_logs['punctuality_barcharts'].get(key)
            if entry:
                flow.append({
                    'stop_id': stop_id,
                    'stop_name': entry['stop_name'],
                    'on_time_pct': entry['punctuality']['punctuality_distribution']['percentages']['on_time'],
                    'too_early_pct': entry['punctuality']['punctuality_distribution']['percentages']['too_early'],
                    'too_late_pct': entry['punctuality']['punctuality_distribution']['percentages']['too__late'],
                    'sample_size': entry['punctuality']['sample_size']
                })
        return flow

    def analyze_and_classify_regulatory_behavior(self, threshold: float = 0.95):
        """
        Identify regulatory stops: if ≥ `threshold` proportion of scheduled departures have `.second == 0`.
        Works with datetime.datetime or pandas.Timestamp.
        """
        df = self.df_final.copy()
        print(f"🔎 Analyzing regulatory stops using threshold {threshold*100:.1f}% zero-second scheduled times...")

        regulatory_count = 0

        for (direction_id, stop_id), group in df.groupby(["direction_id", "stop_id"]):
            times = group["scheduled_departure_time"].dropna()

            if times.empty:
                continue

            # Robustly extract seconds component
            zero_second_count = 0
            for t in times:
                try:
                    second = t.second  # works for datetime, pd.Timestamp
                except AttributeError:
                    try:
                        second = pd.to_datetime(t).second
                    except Exception:
                        continue
                if second == 0:
                    zero_second_count += 1

            proportion = zero_second_count / len(times)

            if proportion >= threshold:
                self.log.add_label("performance", "stops_regulatory", stop_id, direction_id)
                regulatory_count += 1

        print(f"✅ {regulatory_count} stops labeled as regulatory (≥ {threshold*100:.1f}% zero-second departures).")


    def get_punctuality_flow(self, direction_id: str, time_type: str = "all_day"):
        """
        Return a list of punctuality metrics for each stop in canonical order for given direction.
        """
        flow = []

        # Get canonical stop sequence
        pattern_dict = self.log.direction_topology_logs['metadata']['canonical_patterns']
        stop_sequence = pattern_dict.get(direction_id, [])

        for stop_id in stop_sequence:
            key = f"stop_id_{stop_id}_direction_{direction_id}_time_{time_type}"
            entry = self.log.performance_logs['punctuality_barcharts'].get(key)
            if entry:
                flow.append({
                    'stop_id': stop_id,
                    'stop_name': entry['stop_name'],
                    'on_time_pct': entry['punctuality']['punctuality_distribution']['percentages']['on_time'],
                    'too_early_pct': entry['punctuality']['punctuality_distribution']['percentages']['too_early'],
                    'too_late_pct': entry['punctuality']['punctuality_distribution']['percentages']['too__late'],
                    'sample_size': entry['punctuality']['sample_size']
                })
        return flow

    def plot_performance_entry(self, key: str):
        """
        Plots a histogram or punctuality bar chart stored under the given key.
        """

        # Try histogram first
        hist_entry = self.log.performance_logs['histograms_stops'].get(key)
        if hist_entry:
            h = hist_entry['histogram']
            plt.figure(figsize=(10, 4))
            plt.bar(h['bin_labels'], h['probabilities'], color='skyblue')
            plt.xticks(rotation=90)
            plt.ylabel("Probability")
            plt.title(f"Delay Histogram at {hist_entry['stop_name']} ({key})")
            plt.tight_layout()
            plt.show()
            return

        # Then try punctuality bar chart
        punct_entry = self.log.performance_logs['punctuality_barcharts'].get(key)
        if punct_entry:
            p = punct_entry['punctuality']['punctuality_distribution']['percentages']
            plt.figure(figsize=(6, 4))
            plt.bar(p.keys(), p.values(), color='green')
            plt.ylabel("Percentage")
            plt.title(f"Punctuality Bar Chart at {punct_entry['stop_name']} ({key})")
            plt.ylim(0, 100)
            plt.tight_layout()
            plt.show()
            return

        print(f"❌ No histogram or punctuality data found for key: {key}")

    def generate_travel_time_log(self):
        """
        Summarize travel times between consecutive stops for each direction and time_type.
        The result is stored in: self.log.performance_logs['travel_times']
        """
        print("\n=== GENERATING TRAVEL TIME STATISTICS ===")

        df = self.calculate_travel_times_and_delays(self.df_final.copy())
        travel_time_log = {}

        # Keep only valid travel segments
        df_valid = df[df['valid_segment'] & df['observed_travel_time'].notna()].copy()

        # Add from/to stop info
        df_valid.loc[:, 'from_stop_id'] = df_valid.groupby(['trip_id', 'direction_id', 'start_date'])['stop_id'].shift(1)
        df_valid.loc[:, 'from_stop_name'] = df_valid.groupby(['trip_id', 'direction_id', 'start_date'])['stop_name'].shift(1)
        df_valid.loc[:, 'to_stop_id'] = df_valid['stop_id']
        df_valid.loc[:, 'to_stop_name'] = df_valid['stop_name']

        # Drop rows where shift() produced NaN
        df_valid = df_valid.dropna(subset=['from_stop_id', 'observed_travel_time'])

        group_cols = ['route_id', 'direction_id', 'time_type', 'from_stop_id', 'to_stop_id']

        if 'route_id' not in df_valid.columns:
            df_valid['route_id'] = self.route_id

        for (route_id, direction_id, time_type, from_stop_id, to_stop_id), group in df_valid.groupby(group_cols):
            travel_times = group['observed_travel_time'].dt.total_seconds().dropna()

            if len(travel_times) >= 5:
                entry = {
                    'route_id': route_id,
                    'direction_id': direction_id,
                    'time_type': time_type,
                    'from_stop_id': from_stop_id,
                    'from_stop_name': group['from_stop_name'].iloc[0],
                    'to_stop_id': to_stop_id,
                    'to_stop_name': group['to_stop_name'].iloc[0],
                    'sample_size': len(travel_times),
                    'statistics': {
                        'mean': round(travel_times.mean(), 2),
                        'median': round(travel_times.median(), 2),
                        'std': round(travel_times.std(), 2),
                        'min': int(travel_times.min()),
                        'max': int(travel_times.max()),
                        'percentile_25': int(np.percentile(travel_times, 25)),
                        'percentile_75': int(np.percentile(travel_times, 75)),
                        'percentile_95': int(np.percentile(travel_times, 95))
                    }
                }

                key = f"{route_id}_dir{direction_id}_{from_stop_id}_to_{to_stop_id}_time_{time_type}"
                travel_time_log[key] = entry

        self.log.performance_logs['travel_times'] = travel_time_log
        print(f"  ✅ {len(travel_time_log)} travel time segments logged.")

#======================================= EXPORT ==========================================
    
    def add_log_references_to_hierarchies(self):
        """Propagate violation flags and types to stop and direction hierarchies for frontend rendering."""
        stop_hier = self.log.navigation_structures.get("stop_hierarchies", {})
        dir_hier = self.log.navigation_structures.get("direction_hierarchies", {})

        # === STOP-LEVEL VIOLATIONS (from multiple sources) ===
        stop_violation_sources = [
            self.log.stop_topology_logs.get("stop_id_violations", {}).values(),
            self.log.direction_topology_logs.get("stop_id_violations", {}).values(),
            self.log.performance_logs.get("regulatory_performance_violations", {}).values()
        ]

        for source in stop_violation_sources:
            for v in source:
                stop_id = str(v.get("stop_id", ""))
                if not stop_id:
                    continue
                label = v.get("label") or v.get("violation_type") or "unknown"

                # Stop hierarchies
                if stop_id in stop_hier.get("by_stop_id", {}):
                    entry = stop_hier["by_stop_id"][stop_id]
                    entry.setdefault("violation_types", []).append(label)
                    entry["has_violation"] = True

                # Direction hierarchies (by_stop_id layer)
                if stop_id in dir_hier.get("by_stop_id", {}):
                    entry = dir_hier["by_stop_id"][stop_id]
                    entry.setdefault("violation_types", []).append(label)
                    entry["has_violation"] = True

        # === DIRECTION-LEVEL VIOLATIONS ===
        for v in self.log.direction_topology_logs.get("direction_violations", {}).values():
            direction_id = str(v.get("direction_id", ""))
            if not direction_id:
                continue
            label = v.get("label") or v.get("violation_type") or "unknown"

            if direction_id in dir_hier.get("by_direction", {}):
                entry = dir_hier["by_direction"][direction_id]
                entry.setdefault("violation_types", []).append(label)
                entry["has_violation"] = True

        # === STOP-LEVEL LABELS (from multiple sources) ===
        label_sources = [
            self.log.stop_topology_logs.get("stop_id_labels", {}),
            self.log.direction_topology_logs.get("stop_id_labels", {}),
            self.log.performance_logs.get("labels", {}).get("stops_regulatory", {})
        ]

        for label_source in label_sources:
            for stop_id, label in label_source.items():
                stop_id = str(stop_id)
                if stop_id in stop_hier.get("by_stop_id", {}):
                    stop_hier["by_stop_id"][stop_id].setdefault("labels", []).append(label)
                if stop_id in dir_hier.get("by_stop_id", {}):
                    dir_hier["by_stop_id"][stop_id].setdefault("labels", []).append(label)
        
        


        print("✅ Violation flags and types added to navigation hierarchies.") 

    def export_all_logs_frontend_friendly(self, export_dir="exported_logs"):
        route_info = {
            "route_id": self.route_id,
            "route_short_name": self.route_short_name,
            "route_long_name": self.route_long_name
        }

        route_folder = Path(export_dir) / self.route_long_name
        route_folder.mkdir(parents=True, exist_ok=True)

        # === 1. Save all per-route logs ===
        save_json_safe(self.log.stop_topology_logs, route_folder / "stop_topology.json")
        save_json_safe(self.log.direction_topology_logs, route_folder / "direction_topology.json")
        save_json_safe(self.log.navigation_structures, route_folder / "navigation_structures.json")
        save_json_safe(self.log.performance_logs, route_folder / "performance_logs.json")

        print(f"📤 Exported full per-route logs for {self.route_long_name}")

        # === 2. Export global flat violations with route context ===
        def flatten_logs(log_dict, kind):
            return [
                {**v, **route_info, "source": kind}
                for v in log_dict.values()
                if isinstance(v, dict)
            ]

        global_folder = Path(export_dir)
        global_folder.mkdir(parents=True, exist_ok=True)

        flat_stop = flatten_logs(self.log.stop_topology_logs.get("stop_id_violations", {}), "stop_topology") + \
                    flatten_logs(self.log.stop_topology_logs.get("parent_station_violations", {}), "stop_topology") + \
                    flatten_logs(self.log.performance_logs.get("regulatory_violations", {}), "performance") + \
                    flatten_logs(self.log.performance_logs.get("regulatory_performance_violations", {}), "performance")

        flat_direction = flatten_logs(self.log.direction_topology_logs.get("stop_id_violations", {}), "direction_topology") + \
                        flatten_logs(self.log.direction_topology_logs.get("direction_violations", {}), "direction_topology")

        save_json_array(
            append_unique(load_json_array(global_folder / "global_stop_violations.json"), flat_stop),
            global_folder / "global_stop_violations.json"
        )
        save_json_array(
            append_unique(load_json_array(global_folder / "global_direction_violations.json"), flat_direction),
            global_folder / "global_direction_violations.json"
        )

        print("✅ Global flat violations updated.")


    def export_summary_dashboard_files(self, export_dir="exported_logs"):
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        def load_json_as_flat_df(path):
            if not path.exists():
                return pd.DataFrame()
            with open(path, encoding="utf-8") as f:
                return pd.DataFrame(json.load(f))

        stop_json = export_dir / "global_stop_violations.json"
        direction_json = export_dir / "global_direction_violations.json"
        performance_json = export_dir / "global_performance_violations.json"

        df_stop = load_json_as_flat_df(stop_json)
        df_dir = load_json_as_flat_df(direction_json)
        df_perf = load_json_as_flat_df(performance_json)

        if not df_stop.empty:
            df_stop.to_csv(export_dir / "global_stop_violations.csv", index=False)
        if not df_dir.empty:
            df_dir.to_csv(export_dir / "global_direction_violations.csv", index=False)
        if not df_perf.empty:
            df_perf.to_csv(export_dir / "global_performance_violations.csv", index=False)

        print("✅ CSVs exported for flat global violations.")

        def count_by_violation_type(df):
            return dict(df['violation_type'].value_counts()) if 'violation_type' in df else {}

        violation_counts = {
            "stop": count_by_violation_type(df_stop),
            "direction": count_by_violation_type(df_dir),
            "performance": count_by_violation_type(df_perf)
        }

        save_json_safe(violation_counts, export_dir / "global_violation_counts.json")

        summary_path = export_dir / "routes_summary.json"
        existing = []

        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                existing = json.load(f)

        this_summary = {
            "route_id": self.route_id,
            "route_short_name": self.route_short_name,
            "route_long_name": self.route_long_name,
            "violation_counts": {
                "stop_topology": len(self.log.stop_topology_logs.get("stop_id_violations", {})),
                "direction_topology": len(self.log.direction_topology_logs.get("direction_violations", {})),
                "regulatory": len(self.log.performance_logs.get("regulatory_violations", {})) +
                            len(self.log.performance_logs.get("regulatory_performance_violations", {}))
            }
        }

        existing = [r for r in existing if r["route_long_name"] != self.route_long_name]
        existing.append(this_summary)

        save_json_safe(existing, summary_path)

        print("✅ Summary dashboard files updated.")

    def _load_global_json(self, filename):
        path = self.global_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def export_global_navigation_maps(self):
        # 1. stop_name → stop_id
        stopname_to_stopid = self._load_global_json("global_stopname_to_stopid.json")
        for stop_id, meta in self.log.navigation_structures.get("stop_hierarchies", {}).items():
            stop_name = meta.get("stop_name")
            if stop_name:
                stopname_to_stopid.setdefault(stop_name, [])
                if stop_id not in stopname_to_stopid[stop_name]:
                    stopname_to_stopid[stop_name].append(stop_id)
        self._save_global_json("global_stopname_to_stopid.json", stopname_to_stopid)

        # 2. stop-level navigation structure (aggregated across routes)
        stop_nav = self._load_global_json("global_stop_navigation.json")
        for stop_id, meta in self.log.navigation_structures.get("stop_hierarchies", {}).items():
            if stop_id not in stop_nav:
                stop_nav[stop_id] = {
                    "stop_name": meta.get("stop_name"),
                    "parent_station": meta.get("parent_station"),
                    "routes": [],
                    "labels": meta.get("labels", []),
                    "flags": meta.get("flags", [])
                }
            if self.route_id not in stop_nav[stop_id]["routes"]:
                stop_nav[stop_id]["routes"].append(self.route_id)

        self._save_global_json("global_stop_navigation.json", stop_nav)

        # 3. route_name → route_id
        route_name_to_id = self._load_global_json("global_route_name_to_id.json")
        for dir_key, meta in self.log.navigation_structures.get("direction_hierarchies", {}).items():
            route_id = meta.get("route_id")
            long_name = meta.get("route_long_name")
            short_name = meta.get("route_short_name")
            for key in [route_id, long_name, short_name]:
                if key:
                    route_name_to_id.setdefault(key, [])
                    if route_id not in route_name_to_id[key]:
                        route_name_to_id[key].append(route_id)

        self._save_global_json("global_route_name_to_id.json", route_name_to_id)

    def export_global_stop_index(self, output_path="data/global/global_stop_index.json"):
        """
        Exports a global stop index mapping stop_id → metadata and stop_name → parent_station → metadata.
        Includes labels, violation types, routes, and directions.
        """

        stop_index_by_id = {}
        stop_index_by_name = defaultdict(lambda: defaultdict(dict))

        # Gather from stop hierarchy
        stop_hier = self.log.navigation_structures.get("stop_hierarchies", {}).get("by_stop_id", {})
        df = self.df_final

        for stop_id, stop_entry in stop_hier.items():
            stop_id = str(stop_id)
            stop_name = stop_entry.get("stop_name", "UNKNOWN_STOP")
            parent_station = stop_entry.get("parent_station", stop_name)

            stop_data = stop_index_by_id.setdefault(stop_id, {
                "stop_name": stop_name,
                "parent_station": parent_station,
                "labels": set(),
                "violation_types": set(),
                "routes": set(),
                "direction_ids": set()
            })

            # Labels
            label = stop_entry.get("label")
            if isinstance(label, dict):
                stop_data["labels"].update(label.values())
            elif label:
                stop_data["labels"].add(label)

            # Violations
            if stop_entry.get("has_violation"):
                for source in ["stop_topology_logs", "direction_topology_logs", "performance_logs"]:
                    logs = getattr(self.log, source, {})
                    for key in logs:
                        if "violation" in key:
                            for v in logs.get(key, {}).values():
                                if str(v.get("stop_id")) == stop_id:
                                    v_type = v.get("violation_type")
                                    if v_type:
                                        stop_data["violation_types"].add(v_type)

            # Route & direction info
            matching_rows = df[df["stop_id"] == int(stop_id) if stop_id.isdigit() else stop_id]
            stop_data["routes"].update(matching_rows["route_id"].astype(str).unique())
            stop_data["direction_ids"].update(matching_rows["direction_id"].astype(str).unique())

        # Format and store in by_stop_name
        for stop_id, meta in stop_index_by_id.items():
            stop_name = meta["stop_name"]
            parent_station = meta["parent_station"]
            stop_index_by_name[stop_name][parent_station] = {
                "stop_ids": stop_index_by_name[stop_name][parent_station].get("stop_ids", []) + [stop_id],
                "labels": sorted(meta["labels"]),
                "violation_types": sorted(meta["violation_types"]),
                "routes": sorted(meta["routes"]),
                "direction_ids": sorted(meta["direction_ids"])
            }

        # Convert sets to sorted lists
        stop_index_by_id = {
            sid: {
                **v,
                "labels": sorted(v["labels"]),
                "violation_types": sorted(v["violation_types"]),
                "routes": sorted(v["routes"]),
                "direction_ids": sorted(v["direction_ids"])
            } for sid, v in stop_index_by_id.items()
        }

        output_data = {
            "by_stop_id": stop_index_by_id,
            "by_stop_name": stop_index_by_name
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Global stop index exported to {output_path}")