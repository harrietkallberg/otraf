import pandas as pd
import numpy as np
import json as json 
import os
from pathlib import Path

class DataFormer:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.route_id = self.raw_data['route_id'].iloc[0]
        self.route_short_name = self.raw_data['route_short_name'].iloc[0]
        self.route_long_name = self.get_route_long_name()

        # Store results - existing violation logs
        self._stop_name_to_stop_ids = {}
        self._stop_id_to_stop_name = {}
        
        self.basic_nav_map = {}
        self._stop_name_hierarchy = {}
        self._stop_id_hierarchy = {}
        self._parent_station_violations = {}
        self._topology_violations = {}
        self._station_labels = {}
        self._route_dir_alt_trip_types_log = {}  
        self._route_dir_alt_trip_types_regulatory_stops_log = {}  

        self._route_dir_full_trip_log = {}
        self._topology_violations_log = {}       
        self._pattern_violations_log = {}
        self._regulatory_violations_log = {}

        self._histograms_log = {}
        self._punctuality_log = {}
        
        # Navigation structures
        self._stop_to_combinations = {}      
        self._route_to_combinations = {}

        # Process data through the pipeline
        self.df_before = self.prepare_columns(raw_data)
        
        # STEP 1: Early topology detection (basic flags only)
        self.create_and_validate_stop_topology(self.df_before)
        
        # STEP 2: Trip classification (creates trip_type column)
        self.df_classified = self.identify_and_classify_trips(self.df_before)
        
        # STEP 3: Finalize topology violations with trip_type granularity
        self.finalize_topology_violations_log(self.df_classified)
        
        # STEP 4: Continue with rest of pipeline
        self.df_regulatory = self.identify_and_classify_stops(self.df_classified)
        self.df_ready = self.calculate_travel_times_and_delays(self.df_regulatory)
        self.df_final = self.df_ready
        # Add after line with self.df_final = self.df_ready
        self.create_histograms_and_punctuality_analysis(self.df_final)

        self.create_simplified_navigation_maps()
        # Create master indexer after all processing
        self.create_master_log_indexer()
        self.create_master_analysis_indexer()
        
        self.export_all_data()

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
            'scheduled_departure_time', 'observed_departure_time', 'departure_delay', 'route_short_name','city']

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
                
            hour = row['start_time'].hour  # Use start_time instead of scheduled_departure_time
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

# for tracking alternative trip_types
    def get_route_dir_key(self, direction_id):
        return f"{self.route_id}_{direction_id}"
    
    def get_route_dir_key_from_row(self, row):
        """Generate standardized combination key from dataframe row"""
        return f"{self.route_id}_{row['direction_id']}"
    
# for tracking pattern inconsistencies for given trip_type
    def get_route_dir_trip_type_key(self, direction_id, trip_type):
        """Generate standardized combination key for all violation logs and master indexer"""
        return f"{self.route_id}_{direction_id}_{trip_type}"

    def get_route_dir_trip_type_key_from_row(self, row):
        """Generate standardized combination key from dataframe row"""
        return f"{self.route_id}_{row['direction_id']}_{row['trip_type']}"

# for tracking topology inconsistencies for given stop
    def get_route_dir_stop_id_key(self, stop_id, direction_id):
        """Generate standardized combination key for all violation logs and master indexer"""
        return f"{self.route_id}_{direction_id}_{stop_id}"

    def get_route_dir_stop_id_key_from_row(self, row):
        """Generate standardized combination key from dataframe row"""
        return f"{self.route_id}_{row['direction_id']}_{row['stop_id']}"

# for assigning regulatory behaviour of given stop in given trip type
    def get_route_dir_trip_type_stop_id_key(self, stop_id, direction_id, trip_type):
        """Generate standardized combination key for all violation logs and master indexer"""
        return f"{self.route_id}_{direction_id}_{trip_type}_{stop_id}"

    def get_route_dir_trip_type_stop_id_key_from_row(self, row):
        """Generate standardized combination key from dataframe row"""
        return f"{self.route_id}_{row['direction_id']}_{row['trip_type']}_{row['stop_id']}"

#   ====================================== Handle Violation Logging ==========================================

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

#   ================ Validating / Logging Violating RouteID-DirectionID-StopID Behaviours =====================
    def validate_parent_station_consistency(self, df):
        """Validate that each parent_station corresponds to exactly one stop_name"""
        
        # Group by parent_station and check unique stop_names
        parent_station_names = df.groupby('parent_station')['stop_name'].unique()
        
        violations = []
        
        for parent_station, stop_names in parent_station_names.items():
            if len(stop_names) > 1:
                violation = self.create_violation_entry(
                    'multiple_stop_names_per_parent',
                    'high',
                    f'Parent station {parent_station} has {len(stop_names)} different stop names',
                    {
                        'parent_station': parent_station,
                        'stop_names': stop_names.tolist(),
                        'affected_stop_ids': df[df['parent_station'] == parent_station]['stop_id'].unique().tolist(),
                        'stop_name_count': len(stop_names)
                    }
                )
                violations.append(violation)
        
        return violations
    
    def create_hierarchical_stop_mappings(self, df):
        """
        Create two enhanced hierarchical dictionaries:
        1. stop_name -> parent_station -> {stop_ids, label, metadata}
        2. stop_id -> {parent_station, stop_name, label, metadata}
        """
        
        # Basic mapping first
        stop_name_hierarchy = {}
        stop_id_hierarchy = {}
        
        for _, row in df[['stop_name', 'parent_station', 'stop_id']].drop_duplicates().iterrows():
            stop_name = row['stop_name']
            parent_station = str(row['parent_station'])
            stop_id = str(row['stop_id'])
            
            # Build basic stop_name hierarchy
            if stop_name not in stop_name_hierarchy:
                stop_name_hierarchy[stop_name] = {}
            
            if parent_station not in stop_name_hierarchy[stop_name]:
                stop_name_hierarchy[stop_name][parent_station] = []
            
            if stop_id not in stop_name_hierarchy[stop_name][parent_station]:
                stop_name_hierarchy[stop_name][parent_station].append(stop_id)
            
            # Build basic stop_id hierarchy  
            stop_id_hierarchy[stop_id] = {
                'parent_station': parent_station,
                'stop_name': stop_name
            }
        
        # Sort for consistency
        for stop_name in stop_name_hierarchy:
            for parent_station in stop_name_hierarchy[stop_name]:
                stop_name_hierarchy[stop_name][parent_station].sort()
        
        return stop_name_hierarchy, stop_id_hierarchy

    def create_and_validate_stop_topology(self, df):
        """Create hierarchical mappings and validate topology"""
        print("\n=== STOP TOPOLOGY VALIDATION ===")
        
        # First validate parent station consistency
        parent_violations = self.validate_parent_station_consistency(df)
        if parent_violations:
            print(f"🚩 Found {len(parent_violations)} parent station consistency violations:")
            for violation in parent_violations:
                print(f"  - {violation['issue']}")
                print(f"    Stop names: {violation['stop_names']}")
        else:
            print("✅ All parent stations have consistent stop names")
        
        # Create basic hierarchical mappings
        stop_name_hierarchy, stop_id_hierarchy = self.create_hierarchical_stop_mappings(df)
        
        # Validate topology and create enhanced hierarchies
        topology_violations = []
        enhanced_stop_hierarchy = {}
        enhanced_stop_id_hierarchy = {}
        
        for stop_name, parent_stations in stop_name_hierarchy.items():
            enhanced_stop_hierarchy[stop_name] = {}
            
            for parent_station, stop_ids in parent_stations.items():
                label, violation = self._validate_parent_station_topology(df, stop_name, parent_station)
                
                # Enhanced stop_name hierarchy
                enhanced_stop_hierarchy[stop_name][parent_station] = {
                    'stop_ids': stop_ids,
                    'label': label,
                    'has_violation': violation is not None,
                    'violation_type': violation.get('violation_type') if violation else None
                }
                
                # Enhanced stop_id hierarchy - add metadata to each stop_id
                for stop_id in stop_ids:
                    enhanced_stop_id_hierarchy[stop_id] = {
                        'parent_station': parent_station,
                        'stop_name': stop_name,
                        'label': label,  # Same label as the parent station
                        'has_violation': violation is not None,
                        'violation_type': violation.get('violation_type') if violation else None,
                        'sibling_stop_ids': [sid for sid in stop_ids if sid != stop_id]  # Other stop_ids at same station
                    }
                
                if violation:
                    topology_violations.append(violation)
                    print(f"🚩 {stop_name} (parent: {parent_station}) [{label}]: {violation['description']}")
                else:
                    print(f"✅ {stop_name} (parent: {parent_station}) [{label}]: Valid")
        
        # Store enhanced results
        self._stop_name_hierarchy = enhanced_stop_hierarchy
        self._stop_id_hierarchy = enhanced_stop_id_hierarchy  # Enhanced version
        self._parent_station_violations = parent_violations
        self._topology_violations = topology_violations
        
        total_violations = len(parent_violations) + len(topology_violations)
        print(f"Validation complete: {total_violations} total violations ({len(parent_violations)} parent consistency + {len(topology_violations)} topology)")
        
        return {
            'parent_station_violations': parent_violations,
            'topology_violations': topology_violations,
            'enhanced_stop_hierarchy': enhanced_stop_hierarchy,
            'enhanced_stop_id_hierarchy': enhanced_stop_id_hierarchy
        }

    def _validate_parent_station_topology(self, df, stop_name, parent_station):
        """Validate topology for a single parent_station's stop_ids"""
        
        stop_ids = self._stop_name_hierarchy[stop_name][parent_station]
        
        # Initialize classification lists
        bidirectional_stop_ids = []
        directional_stop_ids = []
        no_data_stop_ids = []
        stop_id_direction_details = {}
        
        # First pass: Classify each stop_id based on direction behavior
        for stop_id in stop_ids:
            stop_data = df[df['stop_id'].astype(str) == str(stop_id)]
            directions = stop_data['direction_id'].unique()
            # Get the direction counts
            direction_counts = stop_data['direction_id'].value_counts().to_dict()
            
            # Store detailed direction information
            stop_id_direction_details[stop_id] = {
                'directions': directions.tolist(),
                'direction_counts': direction_counts,  # This gives you {0: 150, 1: 143}
            }

            if len(directions) > 1:
                # Stop serves multiple directions → bidirectional stop id
                bidirectional_stop_ids.append(stop_id)
            elif len(directions) == 1:
                # Stop serves single direction → directional stop id
                directional_stop_ids.append(stop_id)
            else:
                # Stop has no direction data → no data, violation
                no_data_stop_ids.append(stop_id)
        
        # Second pass: Analyze and validate the classification
        num_total = len(stop_ids)
        num_bidirectional = len(bidirectional_stop_ids)
        num_directional = len(directional_stop_ids)
        
        # Check for no data violations first
        if no_data_stop_ids:
            label = "Undefined"
            return label, self.create_violation_entry(
                'stop_ids_without_direction_data', 
                'high',
                f'Stop id(s): {no_data_stop_ids} have no direction data',
                {
                    'parent_station': parent_station,
                    'stop_name': stop_name,
                    'classification': {
                        stop_id: {
                            'type': (
                                'no_data' if stop_id in no_data_stop_ids 
                                else 'bidirectional' if stop_id in bidirectional_stop_ids 
                                else 'directional'
                            ),
                            'direction_counts': details['direction_counts']
                        } for stop_id, details in stop_id_direction_details.items()
                    },
                    'station_label': "Missing data",
                    'all_stop_ids': stop_ids,
                    'no_data_stop_ids': no_data_stop_ids
                }
            )
        
        # Determine station type and validate
        if num_total == 1 and num_bidirectional == 1:
            # Single bidirectional stop → Shared station
            label = 'Shared'
            return label, None  # Valid
            
        elif num_total == 1 and num_directional == 1:
            # Single directional stop → Unidirectional station  
            label = 'Unidirectional'
            return label, None  # Valid
            
        elif num_total % 2 == 0:
            # Even number of stop_ids means even number in both categories, else violation.
            if num_directional % 2 == 0 and num_bidirectional % 2 == 0:
                label = 'Bidirectional'
                return label, None
            else: 
                label = 'Undefined'
                return label, self.create_violation_entry(
                    'odd_directional_count_mixed',
                    'high',
                    f'Odd station behaviour with {num_directional} directional stop ids and {num_bidirectional} bidirectional stop ids.',
                    {
                        'stop_name': stop_name,
                        'parent_station': parent_station,
                        'classification': {stop_id: {
                                'type': 'bidirectional' if stop_id in bidirectional_stop_ids else 'directional',
                                'direction_counts': details['direction_counts']
                            } for stop_id, details in stop_id_direction_details.items()
                        },
                        'station_label': label,
                        'all_stop_ids': stop_ids
                    }
                )
                
        elif num_total % 2 != 0:
            # Odd total - directional must be even, bidirectional can be odd
            if num_directional % 2 == 0 and num_bidirectional % 2 != 0:
                label = 'Mixed'
                return label, None
            else:
                label = 'Undefined'
                return label, self.create_violation_entry(
                    'odd_directional_count_mixed',
                    'high',
                    f'Odd station behaviour with {num_directional} directional stop ids and {num_bidirectional} bidirectional stop ids.',
                    {
                        'stop_name': stop_name,
                        'parent_station': parent_station,
                        'classification': {stop_id: {
                                'type': 'bidirectional' if stop_id in bidirectional_stop_ids else 'directional',
                                'direction_counts': details['direction_counts']
                            } for stop_id, details in stop_id_direction_details.items()
                        },
                        'station_label': label,
                        'all_stop_ids': stop_ids
                    }
                )
        
        return label, None  # No violations

#   ============ Validating / Logging Violating RouteID-DirectionID-TripType-Pattern Behaviours ==============
    
    def identify_and_classify_trips(self, df):
        """Unified trip classification with gap detection using stop-level logs"""
        print("\n=== UNIFIED TRIP CLASSIFICATION WITH GAP DETECTION ===")
        
        df = df.copy()
        
        # Get trip patterns
        trip_info = df.groupby(['trip_id', 'direction_id', 'start_date']).apply(
            lambda g: pd.Series({
                'trip_length': len(g),
                'pattern': tuple(g.sort_values('stop_sequence')['stop_id'].tolist()),
                'first_stop': g.sort_values('stop_sequence')['stop_name'].iloc[0],
                'last_stop': g.sort_values('stop_sequence')['stop_name'].iloc[-1]
            })
        ).reset_index()
        
        # Classify full vs partial
        max_stops = trip_info.groupby('direction_id')['trip_length'].max()
        trip_info = trip_info.merge(max_stops.reset_index().rename(columns={'trip_length': 'max_stops'}), on='direction_id')
        trip_info['is_full'] = trip_info['trip_length'] == trip_info['max_stops']
        
        trip_types_log = {}
        pattern_violations_log = {}
        direction_mapping = {}
        
        for direction in trip_info['direction_id'].unique():
            dir_trips = trip_info[trip_info['direction_id'] == direction]
            
            # Establish canonical pattern
            full_trips = dir_trips[dir_trips['is_full']]
            if len(full_trips) == 0:
                continue
                
            pattern_counts = full_trips['pattern'].value_counts()
            canonical = pattern_counts.index[0]
            
            print(f"Direction {direction}: canonical pattern {len(canonical)} stops (from {pattern_counts.iloc[0]} trips)")
            
            # Analyze all patterns against canonical
            all_patterns = dir_trips['pattern'].unique()
            pattern_mapping = {}
            counter = 0
            for i, pattern in enumerate(all_patterns):
                pattern_trips = dir_trips[dir_trips['pattern'] == pattern]
                trip_count = len(pattern_trips)
                is_full_length = len(pattern) == max_stops[direction]
                
                # Analyze pattern
                analysis = self._analyze_pattern(pattern, canonical)
                
                # Create trip type
                if pattern == canonical and is_full_length:
                    trip_type = 'full'
                else:
                    trip_type = f'alt_{counter+1}'
                
                pattern_mapping[pattern] = trip_type
                
                # CREATE STOP-LEVEL LOGS for each stop in this pattern
                canonical_description = self._create_pattern_description(canonical, canonical)
                pattern_description = self._create_pattern_description(pattern, canonical)
                
                for stop_id in pattern:
                    # Get stop name
                    stop_name = None
                    stop_records = df[df['stop_id'] == stop_id]
                    if not stop_records.empty:
                        stop_name = stop_records['stop_name'].iloc[0]
                    
                    # Create trip type log key
                    trip_type_key = self.get_route_dir_trip_type_key(direction_id, trip_type)
                    invalid_travel_time_key = self.get_route_dir_trip_type_stop_id_key(stop_id, direction_id, trip_type)
                    
                    trip_types_log[trip_type_key] = {
                        'direction_id': direction,
                        'trip_type': trip_type,

                        'pattern_length': len(pattern),
                        'is_canonical': pattern == canonical and is_full_length,
                        'has_gaps': analysis['type'] != 'consecutive',
                        'violation_type': analysis['type'] if analysis['type'] != 'consecutive' else None,
                        'travel_reliable': analysis['valid'],
                        'trip_count': trip_count,
                        'canonical_description': canonical_description,
                        'pattern_description': pattern_description
                    }
                    
                    # Create stop-level pattern violation using standard formatter
                    if analysis['type'] != 'consecutive':
                        
                        # Determine severity based on issue type
                        if analysis['type'] == 'has_swaps':
                            severity = 'high'  # Swaps are more severe
                            violation_type = 'pattern_has_swaps'
                            description = f'Stop sequence swaps invalidate travel times in {trip_type} trips'
                        elif analysis['type'] == 'has_gaps':
                            severity = 'medium'  # Gaps are medium severity
                            violation_type = 'pattern_has_gaps'
                            description = f'Stop sequence gaps invalidate travel times in {trip_type} trips'
                        elif analysis['type'] == 'has_swaps_and_gaps':
                            severity = 'high'  # Combined issues are most severe
                            violation_type = 'pattern_has_swaps_and_gaps'
                            description = f'Stop sequence swaps and gaps invalidate travel times in {trip_type} trips'
                        else:
                            severity = 'medium'  # Default for other issues
                            violation_type = 'pattern_issue'
                            description = f'Pattern issue ({analysis["type"]}) affects {trip_type} trips'
            

                        violation = self.create_violation_entry(
                            violation_type=violation_type,
                            severity=severity,
                            description=description,
                            stop_id=stop_id,
                            stop_name=stop_name,
                            direction_id=direction,
                            trip_type=trip_type,
                            trip_count=trip_count,
                            canonical_description=canonical_description,
                            problematic_description=pattern_description
                        )
                            
                        violation_key = self.get_combination_key(stop_id, direction, trip_type)
                        self.add_violation_to_log(pattern_violations_log, violation_key, violation)
            
            # Store direction mapping
            direction_mapping[direction] = pattern_mapping
        
        # Apply trip types to dataframe
        def get_trip_type(row):
            direction = row['direction_id']
            pattern = row['pattern']
            return direction_mapping.get(direction, {}).get(pattern, 'full')
        
        trip_info['trip_type'] = trip_info.apply(get_trip_type, axis=1)
        
        trip_mapping = dict(zip(
            trip_info[['trip_id', 'direction_id', 'start_date']].apply(tuple, axis=1),
            trip_info['trip_type']
        ))
        df['trip_type'] = df[['trip_id', 'direction_id', 'start_date']].apply(tuple, axis=1).map(trip_mapping).fillna('full')
        
        # Store violations
        self._trip_types_log = trip_types_log
        self._pattern_violations_log = pattern_violations_log
        
        # Count affected records (same logic as before)
        violated_trip_types = set()
        for violation in pattern_violations_log.values():
            direction_id = violation.get('direction_id')
            trip_type = violation.get('trip_type')
            violated_trip_types.add((direction_id, trip_type))

        affected_records = 0
        if violated_trip_types:
            for direction_id, trip_type in violated_trip_types:
                affected_records += len(df[(df['direction_id'] == direction_id) & (df['trip_type'] == trip_type)])

        non_full_records = len(df[df['trip_type'] != 'full'])

        print(f"Complete: {len(trip_info)} trips, {len(pattern_violations_log)} stop-level violations affecting {affected_records} records")
        print(f"Additional info: {non_full_records} total non-full trip records")
        
        return df

    def calculate_travel_times_and_delays(self, df):
        """Calculate travel times using stop-level pattern violations"""
        print("\n=== CALCULATING TRAVEL TIMES AND DELAYS ===")
        
        df = df.sort_values(['trip_id', 'direction_id', 'start_date', 'stop_sequence'])
        trip_groups = ['trip_id', 'direction_id', 'start_date']
        
        # Get previous stop info
        df['previous_stop'] = df.groupby(trip_groups)['stop_name'].shift(1)
        df['prev_delay'] = df.groupby(trip_groups)['departure_delay'].shift(1)
        
        # Calculate incremental delay
        df['incremental_delay'] = df['departure_delay'] - df['prev_delay']
        
        print("  Using stop-level pattern violations for travel time validation")
        
        def is_travel_time_valid(row):
            """Simple lookup in stop-level pattern violations log"""
            
            # First stop in trip - no travel time possible
            if pd.isna(row['previous_stop']):
                return False
            
            # Check if this specific stop has a gap violation
            
            violation_key = self.get_combination_key_from_row(row)

            
            if hasattr(self, '_pattern_violations_log'):
                return violation_key not in self._pattern_violations_log  # Valid if NOT in violations log
            
            return True  # Default to valid if no violations log
        
        # Apply validation
        df['travel_time_valid'] = df.apply(is_travel_time_valid, axis=1)
        
        # Set incremental delay to NaN for invalid segments
        df.loc[~df['travel_time_valid'], 'incremental_delay'] = np.nan
        
        # Calculate travel times
        time_columns = ['scheduled_departure_time', 'observed_departure_time']
        for time_col in time_columns:
            if time_col in df.columns:
                prefix = time_col.split('_')[0]
                prev_col = f'prev_{prefix}_departure'
                df[prev_col] = df.groupby(trip_groups)[time_col].shift(1)
                travel_col = f'{prefix}_travel_time'
                
                df[travel_col] = df[time_col] - df[prev_col]
                df.loc[~df['travel_time_valid'], travel_col] = pd.NaT
                df = df.drop(columns=[prev_col])
        
        # Clean up
        df = df.drop(columns=['prev_delay'])
        
        valid_segments = df['travel_time_valid'].sum()
        
        print(f"Calculation complete: {valid_segments}/{len(df)} valid segments ({valid_segments/len(df)*100:.1f}%)")
        print(f"  Using stop-level pattern violations for precise validation")
        
        return df

    def _analyze_pattern(self, pattern, canonical):
        """Analyze pattern using elegant sorting approach"""
        if not canonical or not pattern:
            return {'type': 'unknown', 'valid': False}
        
        p_list, c_list = list(pattern), list(canonical)
        
        # Check if already consecutive
        if self._is_consecutive(p_list, c_list):
            return {'type': 'consecutive', 'valid': True}
        
        # Check for invalid stops
        if not set(p_list).issubset(set(c_list)):
            return {'type': 'invalid_stops', 'valid': False}
        
        # Sort and analyze
        sorted_p = sorted(p_list, key=lambda x: c_list.index(x))
        order_changed = p_list != sorted_p
        consecutive_after_sort = self._is_consecutive(sorted_p, c_list)
        
        if order_changed and consecutive_after_sort:
            return {'type': 'has_swaps', 'valid': False}
        elif not order_changed and not consecutive_after_sort:
            return {'type': 'has_gaps', 'valid': False}
        elif order_changed and not consecutive_after_sort:
            return {'type': 'has_swaps_and_gaps', 'valid': False}
        else:
            return {'type': 'unknown', 'valid': False}

    def _is_consecutive(self, pattern, canonical):
        """Check if pattern appears consecutively in canonical"""
        n = len(pattern)
        return any(canonical[i:i+n] == pattern for i in range(len(canonical) - n + 1))

    def _create_pattern_description(self, pattern, canonical=None):
        """Create human-readable pattern description showing positions in canonical pattern like '1-2-3_6-7' for gaps"""
        if not pattern:
            return "Empty"
        
        if len(pattern) <= 1:
            if canonical and pattern[0] in canonical:
                return str(canonical.index(pattern[0]) + 1)  # 1-based indexing
            return str(pattern[0]) if pattern else "Empty"
        
        # If no canonical provided, use stop_ids directly (fallback)
        if not canonical:
            try:
                int_pattern = [int(x) for x in pattern]
            except (ValueError, TypeError):
                return "-".join(map(str, pattern))
            
            # Group consecutive numbers
            groups = []
            current_group = [int_pattern[0]]
            
            for i in range(1, len(int_pattern)):
                if int_pattern[i] == int_pattern[i-1] + 1:
                    current_group.append(int_pattern[i])
                else:
                    groups.append(current_group)
                    current_group = [int_pattern[i]]
            groups.append(current_group)
            
            # Format groups
            formatted_groups = []
            for group in groups:
                if len(group) == 1:
                    formatted_groups.append(str(group[0]))
                elif len(group) == 2:
                    formatted_groups.append(f"{group[0]}-{group[1]}")
                else:
                    formatted_groups.append(f"{group[0]}-{group[-1]}")
            
            return "_".join(formatted_groups)
        
        # Convert pattern stop_ids to positions in canonical pattern (1-based)
        try:
            positions = []
            for stop_id in pattern:
                if stop_id in canonical:
                    positions.append(canonical.index(stop_id) + 1)  # 1-based indexing
                else:
                    # Stop not in canonical - this shouldn't happen for valid patterns
                    positions.append(f"?{stop_id}")  # Mark as unknown
            
            # Group consecutive positions
            groups = []
            current_group = []
            
            for pos in positions:
                if isinstance(pos, str) and pos.startswith('?'):
                    # Handle unknown positions
                    if current_group:
                        groups.append(current_group)
                        current_group = []
                    groups.append([pos])
                else:
                    if not current_group:
                        current_group = [pos]
                    elif pos == current_group[-1] + 1:
                        current_group.append(pos)
                    else:
                        groups.append(current_group)
                        current_group = [pos]
            
            if current_group:
                groups.append(current_group)
            
            # Format groups
            formatted_groups = []
            for group in groups:
                if len(group) == 1:
                    formatted_groups.append(str(group[0]))
                elif len(group) == 2:
                    formatted_groups.append(f"{group[0]}-{group[1]}")
                else:
                    formatted_groups.append(f"{group[0]}-{group[-1]}")
            
            return "_".join(formatted_groups)
            
        except Exception:
            # Fallback to stop_id representation
            return "-".join(map(str, pattern))

#   ===== Validating / Logging Violating Regulatory RouteID-DirectionID-TripType-StopID Behaviours ===========

    def identify_and_classify_stops(self, df):
        """Detect regulatory stops"""
        print("\n=== DETECTING REGULATORY STOPS ===")
        
        df = df.copy()
        df['is_regulatory'] = False
        
        # Extract seconds and group by route-direction-trip_type
        df['departure_seconds'] = df['scheduled_departure_time'].dt.second
        
        regulatory_analysis = df.groupby(['direction_id', 'stop_id', 'trip_type']).agg({
            'stop_name': 'first',
            'departure_seconds': [
                lambda x: (x == 0).sum(),
                'count'
            ]
        }).reset_index()
        
        regulatory_analysis.columns = [
            'direction_id', 'stop_id', 'trip_type', 'stop_name', 
            'zero_seconds_count', 'total_records'
        ]
        
        regulatory_analysis['zero_seconds_ratio'] = (
            regulatory_analysis['zero_seconds_count'] / regulatory_analysis['total_records']
        )
        
        regulatory_analysis['is_perfectly_regulatory'] = regulatory_analysis['zero_seconds_ratio'] == 1.0
        regulatory_analysis['is_regulatory'] = regulatory_analysis['zero_seconds_ratio'] >= 0.95
        regulatory_analysis['has_anomaly'] = (
            (regulatory_analysis['zero_seconds_ratio'] >= 0.95) & 
            (regulatory_analysis['zero_seconds_ratio'] < 1.0)
        )
        
        regulatory_combinations = regulatory_analysis[regulatory_analysis['is_regulatory']].copy()
        
        print(f"Found {len(regulatory_combinations)} regulatory combinations")
        if regulatory_combinations['has_anomaly'].sum() > 0:
            print(f"  - {regulatory_combinations['has_anomaly'].sum()} with anomalies")
        
        # Update main dataframe
        if len(regulatory_combinations) > 0:
            for _, row in regulatory_combinations.iterrows():
                mask = (df['direction_id'] == row['direction_id']) & \
                    (df['stop_id'] == row['stop_id']) & \
                    (df['trip_type'] == row['trip_type'])
                df.loc[mask, 'is_regulatory'] = True
        
        # Create logs
        self._regulatory_stops_log = {}
        regulatory_violations_log = {}
        
        for _, row in regulatory_combinations.iterrows():
            key = self.get_combination_key(row['stop_id'], row['direction_id'], row['trip_type'])
            
            self._regulatory_stops_log[key] = {
                'route_id': self.route_id,
                'route_long_name': self.route_long_name,
                'route_short_name': self.route_short_name,
                'stop_id': row['stop_id'],
                'stop_name': row['stop_name'],
                'direction_id': row['direction_id'],
                'trip_type': row['trip_type'],
                'is_regulatory': True,
                'is_perfect': bool(row['is_perfectly_regulatory']),
                'has_anomaly': bool(row['has_anomaly']),
                'zero_seconds_ratio': row['zero_seconds_ratio'],
                'total_records': int(row['total_records'])
            }
            
            # Create violation for incomplete regulation
            if row['has_anomaly']:
                violation = self.create_violation_entry(
                    violation_type='incomplete_regulation',
                    severity='low',
                    description=f"Only {row['zero_seconds_ratio']:.1%} regulated",
                    stop_id=row['stop_id'],
                    stop_name=row['stop_name'],
                    direction_id=row['direction_id'],
                    trip_type=row['trip_type'],
                    zero_seconds_ratio=row['zero_seconds_ratio'],
                    total_records=int(row['total_records'])
                )
                
                violation_key = f"{key}"
                self.add_violation_to_log(regulatory_violations_log, violation_key, violation)
        
        self._regulatory_violations_log = regulatory_violations_log
        
        df = df.drop('departure_seconds', axis=1)
        print(f"Regulatory violations created: {len(regulatory_violations_log)}")
        print(f"Regulatory analysis complete: {len(self._regulatory_stops_log)} combinations, {len(regulatory_violations_log)} violations")
        return df
   
#   ======================================= Handle Navigational Maps =========================================

    def _get_topology_info_for_combination(self, stop_id, direction_id, trip_type):
        """Get topology information for specific combination"""
        combination_key = self.get_combination_key(stop_id, direction_id, trip_type)
        
        if combination_key in self._topology_violations_log:
            violation = self._topology_violations_log[combination_key]
            return {
                'topology_violation': True,
                'topology_violation_type': violation.get('violation_type'),
                'severity': violation.get('severity')
            }
        
        return {
            'topology_violation': False,
            'topology_violation_type': None,
            'severity': None
        }

    def _get_pattern_info_for_combination(self, stop_id, direction_id, trip_type):
        """Get pattern information for specific stop/direction/trip_type combination"""
        # Create the key for stop-level trip types log
        trip_type_key = self.get_combination_key(stop_id, direction_id, trip_type)
        
        if hasattr(self, '_trip_types_log') and trip_type_key in self._trip_types_log:
            trip_info = self._trip_types_log[trip_type_key]
            return {
                'pattern_violation': trip_info.get('has_issues', False),
                'pattern_violation_type': trip_info.get('violation_type'),
                'pattern_description': trip_info.get('pattern_description'),
                'severity': trip_info.get('severity')
            }
        
        return {
            'has_issues': False,
            'issue_type': None,
            'severity': None
        }

    def _get_regulatory_info_for_combination(self, stop_id, direction_id, trip_type):
        """Get regulatory information for specific stop/direction/trip_type combination"""
        key = self.get_combination_key(stop_id, direction_id, trip_type)
        
        if key in self._regulatory_stops_log:
            reg_info = self._regulatory_stops_log[key]
            return {
                'has_anomaly': reg_info.get('has_anomaly', False),
                'regulation_ratio': reg_info.get('zero_seconds_ratio', 1.0),
                'severity': 'low' if reg_info.get('has_anomaly') else None
            }
        
        return {
            'has_anomaly': False,
            'regulation_ratio': None,
            'severity': None
        }

    def _get_histogram_info_for_combination(self, stop_id, direction_id, time_type):
        """Get histogram information for specific stop/direction/time_type combination"""
        combo_key = f"{self.route_id}_{stop_id}_{direction_id}_{time_type}"
        
        if hasattr(self, '_histograms_log') and combo_key in self._histograms_log:
            histogram_data = self._histograms_log[combo_key]
            return {
                'has_histograms': True,
                'histogram_types': list(histogram_data['histograms'].keys()),
                'sample_size': histogram_data['metadata']['total_sample_size']
            }
        
        return {
            'has_histograms': False,
            'histogram_types': [],
            'sample_size': 0
        }

    def _get_punctuality_info_for_combination(self, stop_id, direction_id, time_type):
        """Get punctuality information for specific stop/direction/time_type combination"""
        combo_key = f"{self.route_id}_{stop_id}_{direction_id}_{time_type}"
        
        if hasattr(self, '_punctuality_log') and combo_key in self._punctuality_log:
            punctuality_data = self._punctuality_log[combo_key]
            otp = punctuality_data['punctuality_metrics']['performance_indicators']['on_time_performance_percent']
            return {
                'has_punctuality': True,
                'on_time_performance': otp,
                'sample_size': punctuality_data['metadata']['sample_size'],
                'performance_level': 'excellent' if otp >= 85 else 'good' if otp >= 70 else 'poor'
            }
        
        return {
            'has_punctuality': False,
            'on_time_performance': None,
            'sample_size': 0,
            'performance_level': None
        }

    def _determine_overall_severity(self, topology_severity, pattern_severity, regulatory_severity):
        """Determine overall severity from individual severities (updated to 3 parameters)"""
        severities = [s for s in [topology_severity, pattern_severity, regulatory_severity] if s is not None]
        
        if not severities:
            return 'none'
        
        severity_order = ['low', 'medium', 'high']
        for severity in reversed(severity_order):
            if severity in severities:
                return severity
        
        return 'none'

    # ======================================= Export All Data =========================================
    def _create_route_mapping(self):
        """Create route name ↔ route ID mapping for easy lookups"""
        return {
            f'{self.route_short_name}': {
                'route_id': self.route_id,
                'route_long_name': self.route_long_name
            }
        }  

    # 1. ADD HISTOGRAM AND PUNCTUALITY ANALYSIS GENERATION
    def create_histograms_and_punctuality_analysis(self, df):
        """Create histograms and punctuality analysis aggregated across all trip types"""
        print("\n=== CREATING HISTOGRAMS AND PUNCTUALITY ANALYSIS ===")
        
        df = df.copy()
        
        # Initialize logs
        histograms_log = {}
        punctuality_log = {}
        
        # Group by stop, direction, and time_type (aggregating across trip_types)
        analysis_groups = df.groupby(['stop_id', 'stop_name', 'direction_id', 'time_type'])
        
        print(f"Processing {len(analysis_groups)} unique stop-direction-time combinations")
        
        for (stop_id, stop_name, direction_id, time_type), group in analysis_groups:
            
            # Create combination key WITHOUT trip_type
            combo_key = f"{self.route_id}_{stop_id}_{direction_id}_{time_type}"
            
            print(f"Analyzing: {stop_name} (Dir {direction_id}, {time_type})")
            
            # HISTOGRAMS ANALYSIS
            histogram_data = self._create_histogram_analysis(group, stop_id, stop_name, direction_id, time_type)
            if histogram_data:
                histograms_log[combo_key] = histogram_data
            
            # PUNCTUALITY ANALYSIS  
            punctuality_data = self._create_punctuality_analysis(group, stop_id, stop_name, direction_id, time_type)
            if punctuality_data:
                punctuality_log[combo_key] = punctuality_data
        
        # Store in class
        self._histograms_log = histograms_log
        self._punctuality_log = punctuality_log
        
        print(f"Analysis complete:")
        print(f"  - {len(histograms_log)} histogram combinations")
        print(f"  - {len(punctuality_log)} punctuality combinations")
        
        return histograms_log, punctuality_log

    def _create_histogram_analysis(self, group, stop_id, stop_name, direction_id, time_type, bins=20):
        """Create histogram analysis for a specific combination"""
        
        if len(group) < 10:  # Minimum sample size
            return None
        
        # Get clean delay data
        total_delays = group['departure_delay'].dropna()
        incremental_delays = group['incremental_delay'].dropna()
        
        if len(total_delays) < 5:  # Need sufficient data
            return None
        
        histograms = {}
        
        # Total delay histogram
        if len(total_delays) >= 5:
            histograms['total_delay'] = self._create_normalized_histogram(
                total_delays, bins, 
                f'Total Delay - {stop_name} (Dir {direction_id}, {time_type.replace("_", " ").title()})'
            )
        
        # Incremental delay histogram (only if valid travel times)
        valid_incremental = incremental_delays[group['travel_time_valid'] == True] if 'travel_time_valid' in group.columns else incremental_delays
        if len(valid_incremental) >= 5:
            histograms['incremental_delay'] = self._create_normalized_histogram(
                valid_incremental, bins,
                f'Incremental Delay - {stop_name} (Dir {direction_id}, {time_type.replace("_", " ").title()})'
            )
        
        if not histograms:
            return None
        
        # Create standardized log entry
        return {
            # SEARCHABLE FIELDS - consistent with other logs
            'route_id': str(self.route_id),
            'route_name': self.route_long_name,
            'route_short_name': self.route_short_name,
            'stop_id': str(stop_id),
            'stop_name': stop_name,
            'direction_id': str(direction_id),
            'time_type': time_type,
            # Note: NO trip_type field since we aggregate across all trip types
            
            # HISTOGRAM DATA
            'histograms': histograms,
            'metadata': {
                'bins_used': bins,
                'total_sample_size': len(group),
                'total_delay_sample_size': len(total_delays),
                'incremental_delay_sample_size': len(valid_incremental),
                'trip_types_included': sorted(group['trip_type'].unique().tolist()) if 'trip_type' in group.columns else ['all'],
                'histogram_count': len(histograms)
            }
        }

    def _create_punctuality_analysis(self, group, stop_id, stop_name, direction_id, time_type):
        """Create punctuality analysis for a specific combination"""
        
        if len(group) < 5:  # Minimum sample size
            return None
        
        # Get clean delay data
        delays = group['departure_delay'].dropna()
        
        if len(delays) < 5:
            return None
        
        # Calculate punctuality metrics
        punctuality_metrics = self._calculate_punctuality_metrics(delays)
        
        if not punctuality_metrics:
            return None
        
        # Create standardized log entry
        return {
            # SEARCHABLE FIELDS - consistent with other logs
            'route_id': str(self.route_id),
            'route_name': self.route_long_name,
            'route_short_name': self.route_short_name,
            'stop_id': str(stop_id),
            'stop_name': stop_name,
            'direction_id': str(direction_id),
            'time_type': time_type,
            # Note: NO trip_type field since we aggregate across all trip types
            
            # PUNCTUALITY DATA
            'punctuality_metrics': punctuality_metrics,
            'metadata': {
                'sample_size': len(delays),
                'trip_types_included': sorted(group['trip_type'].unique().tolist()) if 'trip_type' in group.columns else ['all'],
                'analysis_type': 'departure_delay_based'
            }
        }

    def _create_normalized_histogram(self, data, bins, title):
        """Create a normalized histogram from delay data"""
        
        # Calculate histogram
        counts, bin_edges = np.histogram(data, bins=bins)
        
        # Normalize to probability distribution (sum = 1.0)
        probabilities = counts / counts.sum()
        
        # Create bin labels for display
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            label = f"{bin_edges[i]:.0f}s to {bin_edges[i+1]:.0f}s"
            bin_labels.append(label)
        
        # Calculate statistics
        stats = {
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
        
        return {
            'title': title,
            'bin_edges': bin_edges.tolist(),
            'bin_centers': ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
            'bin_labels': bin_labels,
            'counts': counts.tolist(),
            'probabilities': probabilities.tolist(),
            'statistics': stats
        }

    def _calculate_punctuality_metrics(self, delays):
        """Calculate comprehensive punctuality metrics"""
        
        try:
            # Basic statistics
            mean_delay = float(delays.mean())
            median_delay = float(delays.median())
            std_delay = float(delays.std()) if len(delays) > 1 else 0.0
            
            # Punctuality thresholds (in seconds)
            thresholds = {
                'early': delays < -60,          # More than 1 min early
                'on_time': (delays >= -60) & (delays <= 300),  # Within 1 min early to 5 min late
                'slightly_late': (delays > 300) & (delays <= 600),  # 5-10 min late
                'late': (delays > 600) & (delays <= 1200),          # 10-20 min late
                'very_late': delays > 1200      # More than 20 min late
            }
            
            # Calculate percentages
            total_count = len(delays)
            percentages = {}
            counts = {}
            
            for category, condition in thresholds.items():
                count = condition.sum()
                counts[category] = int(count)
                percentages[category] = float(count / total_count * 100)
            
            # Additional metrics
            percentiles = {
                'p5': float(delays.quantile(0.05)),
                'p25': float(delays.quantile(0.25)),
                'p75': float(delays.quantile(0.75)),
                'p95': float(delays.quantile(0.95))
            }
            
            # Performance indicators
            on_time_performance = percentages['on_time']  # OTP
            reliability_index = 100 - std_delay / 60  # Simple reliability metric
            
            return {
                'basic_statistics': {
                    'mean_delay_seconds': mean_delay,
                    'median_delay_seconds': median_delay,
                    'std_delay_seconds': std_delay,
                    'min_delay_seconds': float(delays.min()),
                    'max_delay_seconds': float(delays.max())
                },
                'percentiles': percentiles,
                'punctuality_categories': {
                    'counts': counts,
                    'percentages': percentages
                },
                'performance_indicators': {
                    'on_time_performance_percent': on_time_performance,
                    'reliability_index': max(0, reliability_index),  # Cap at 0 minimum
                    'punctuality_score': on_time_performance  # Could be enhanced later
                },
                'sample_size': total_count
            }
            
        except Exception as e:
            print(f"Error calculating punctuality metrics: {e}")
            return None

# Clean implementation of Navigation Maps and Master Indexers
    def create_simplified_navigation_maps(self):
        """Create cleaner navigation structure"""
        
        # Basic route→stop→direction mapping
        basic_route_navigation = self._build_basic_route_navigation()
        
        # Basic stop→route→direction mapping  
        basic_stop_navigation = self._build_basic_stop_navigation()
        
        # Store in simple structure
        self._navigation_maps = {
            'stop_to_combinations': basic_stop_navigation,
            'route_to_combinations': basic_route_navigation,
            'stop_name_to_stop_ids': self._stop_name_to_stop_ids
        }
        
        return self._navigation_maps

    def _build_basic_route_navigation(self):
        """Build basic route navigation without complex embedded details"""
        route_nav = {}
        
        # Group by route→direction→stop
        route_groups = self.df_final.groupby(['stop_id', 'stop_name', 'direction_id'])
        
        for (stop_id, stop_name, direction_id), group in route_groups:
            route_id = str(self.route_id)
            
            # Initialize nested structure
            if route_id not in route_nav:
                route_nav[route_id] = {
                    "route_name": self.route_long_name,
                    "route_short_name": self.route_short_name,
                    "directions": {}
                }
            
            if str(direction_id) not in route_nav[route_id]["directions"]:
                route_nav[route_id]["directions"][str(direction_id)] = {
                    "stops": {}
                }
            
            # Add stop with simple summary
            route_nav[route_id]["directions"][str(direction_id)]["stops"][str(stop_id)] = {
                "stop_name": stop_name,
                "trip_types": sorted(group['trip_type'].unique().tolist()),
                "time_types": sorted(group['time_type'].unique().tolist()),
                "total_records": len(group),
                "has_violations": self.has_violations_for_combination(stop_id, direction_id, group['trip_type'].iloc[0])
            }
        
        return route_nav

    def _build_basic_stop_navigation(self):
        """Build basic stop navigation without complex embedded details"""
        stop_nav = {}
        
        # Group by stop→route→direction
        stop_groups = self.df_final.groupby(['stop_id', 'stop_name', 'direction_id'])
        
        for (stop_id, stop_name, direction_id), group in stop_groups:
            route_id = str(self.route_id)
            
            # Initialize nested structure
            if str(stop_id) not in stop_nav:
                stop_nav[str(stop_id)] = {
                    "stop_name": stop_name,
                    "routes": {}
                }
            
            if route_id not in stop_nav[str(stop_id)]["routes"]:
                stop_nav[str(stop_id)]["routes"][route_id] = {
                    "route_name": self.route_long_name,
                    "route_short_name": self.route_short_name,
                    "directions": {}
                }
            
            # Add direction with simple summary
            stop_nav[str(stop_id)]["routes"][route_id]["directions"][str(direction_id)] = {
                "trip_types": sorted(group['trip_type'].unique().tolist()),
                "time_types": sorted(group['time_type'].unique().tolist()),
                "total_records": len(group),
                "has_violations": self.has_violations_for_combination(stop_id, direction_id, group['trip_type'].iloc[0])
            }
        
        return stop_nav

    def _get_trip_type_details_for_stop(self, stop_id, direction_id):
        """Get detailed trip type information for a specific stop-direction combination"""
        
        # Get data for this specific stop-direction
        stop_direction_data = self.df_final[
            (self.df_final['stop_id'].astype(str) == str(stop_id)) &
            (self.df_final['direction_id'].astype(str) == str(direction_id))
        ]
        
        if len(stop_direction_data) == 0:
            return {}
        
        trip_type_details = {}
        
        # Analyze each trip type at this stop
        for trip_type in stop_direction_data['trip_type'].unique():
            trip_data = stop_direction_data[stop_direction_data['trip_type'] == trip_type]
            
            # Get time type breakdown
            time_type_breakdown = trip_data['time_type'].value_counts().to_dict()
            available_time_types = sorted(time_type_breakdown.keys(),
                key=lambda x: ['am_rush', 'day', 'pm_rush', 'night', 'weekend'].index(x) 
                if x in ['am_rush', 'day', 'pm_rush', 'night', 'weekend'] else 999)
            
            # Get violation info from master log indexer if available
            combo_key = self.get_combination_key(stop_id, direction_id, trip_type)
            has_violations = False
            violation_types = []
            
            if hasattr(self, '_master_log_indexer') and combo_key in self._master_log_indexer:
                log_entry = self._master_log_indexer[combo_key]
                has_violations = log_entry.get('has_any_violation', False)
                
                if log_entry.get('has_topology_violation'):
                    violation_types.append('topology')
                if log_entry.get('has_pattern_violation'):
                    violation_types.append('pattern')
                if log_entry.get('has_regulatory_violation'):
                    violation_types.append('regulatory')
            
            # Get pattern description if available
            pattern_description = trip_type
            if (hasattr(self, '_trip_types_log') and 
                combo_key in self._trip_types_log):
                pattern_description = self._trip_types_log[combo_key].get('pattern_description', trip_type)
            
            # Determine service characteristics
            service_characteristics = []
            if len(available_time_types) >= 4:
                service_characteristics.append('all_day')
            elif set(available_time_types).issubset({'am_rush', 'pm_rush'}):
                service_characteristics.append('peak_only')
            elif set(available_time_types).issubset({'day', 'night', 'weekend'}):
                service_characteristics.append('off_peak')
            elif 'weekend' in available_time_types and len(available_time_types) == 1:
                service_characteristics.append('weekend_only')
            
            trip_type_details[trip_type] = {
                # Basic info
                'total_records': len(trip_data),
                'unique_trips': trip_data['trip_id'].nunique(),
                'pattern_description': pattern_description,
                
                # Time type availability - KEY NAVIGATION INFO
                'time_types_available': available_time_types,
                'time_type_record_counts': time_type_breakdown,
                
                # Violation info (for UI indicators)
                'has_violations': has_violations,
                'violation_types': violation_types,
                'is_regulatory': combo_key in getattr(self, '_regulatory_stops_log', {})
            }
        
        return trip_type_details

    def _get_time_type_details_for_stop(self, stop_id, direction_id):
        """Get detailed time type information for a specific stop-direction combination"""
        
        # Get data for this specific stop-direction
        stop_direction_data = self.df_final[
            (self.df_final['stop_id'].astype(str) == str(stop_id)) &
            (self.df_final['direction_id'].astype(str) == str(direction_id))
        ]
        
        if len(stop_direction_data) == 0:
            return {}
        
        time_type_details = {}
        
        # Analyze each time type at this stop
        for time_type in stop_direction_data['time_type'].unique():
            time_data = stop_direction_data[stop_direction_data['time_type'] == time_type]
            
            # Get trip type breakdown (which trip types contribute to this time type)
            trip_type_breakdown = time_data['trip_type'].value_counts().to_dict()
            contributing_trip_types = sorted(trip_type_breakdown.keys(),
                key=lambda x: ['full', 'partial_1', 'partial_2', 'partial_3', 'partial_4', 'partial_5'].index(x) 
                if x in ['full', 'partial_1', 'partial_2', 'partial_3', 'partial_4', 'partial_5'] else 999)
            
            # Check analysis availability for this time type
            analysis_key = f"{self.route_id}_{stop_id}_{direction_id}_{time_type}"
            
            histogram_available = False
            histogram_sample_size = 0
            histogram_types = []
            
            punctuality_available = False
            punctuality_sample_size = 0
            on_time_performance = None
            performance_level = None
            
            if hasattr(self, '_histograms_log') and analysis_key in self._histograms_log:
                histogram_available = True
                hist_data = self._histograms_log[analysis_key]
                histogram_sample_size = hist_data.get('metadata', {}).get('total_sample_size', 0)
                histogram_types = list(hist_data.get('histograms', {}).keys())
            
            if hasattr(self, '_punctuality_log') and analysis_key in self._punctuality_log:
                punctuality_available = True
                punct_data = self._punctuality_log[analysis_key]
                punctuality_sample_size = punct_data.get('metadata', {}).get('sample_size', 0)
                on_time_performance = punct_data.get('punctuality_metrics', {}).get('performance_indicators', {}).get('on_time_performance_percent')
                
                # Performance level assessment
                if on_time_performance is not None:
                    if on_time_performance >= 85:
                        performance_level = 'excellent'
                    elif on_time_performance >= 70:
                        performance_level = 'good'
                    elif on_time_performance >= 50:
                        performance_level = 'fair'
                    else:
                        performance_level = 'poor'
            
            # Check if any contributing trip types have violations
            violations_summary = {
                'any_trip_type_has_violations': False,
                'trip_types_with_violations': [],
                'violation_types_present': []
            }
            
            for trip_type in contributing_trip_types:
                combo_key = self.get_combination_key(stop_id, direction_id, trip_type)
                
                if hasattr(self, '_master_log_indexer') and combo_key in self._master_log_indexer:
                    log_entry = self._master_log_indexer[combo_key]
                    if log_entry.get('has_any_violation', False):
                        violations_summary['any_trip_type_has_violations'] = True
                        violations_summary['trip_types_with_violations'].append(trip_type)
                        
                        # Collect violation types
                        if log_entry.get('has_topology_violation'):
                            violations_summary['violation_types_present'].append('topology')
                        if log_entry.get('has_pattern_violation'):
                            violations_summary['violation_types_present'].append('pattern')
                        if log_entry.get('has_regulatory_violation'):
                            violations_summary['violation_types_present'].append('regulatory')
            
            # Remove duplicates from violation types
            violations_summary['violation_types_present'] = list(set(violations_summary['violation_types_present']))
            
            # Determine time period characteristics
            time_characteristics = []
            if time_type == 'am_rush':
                time_characteristics.append('morning_peak')
            elif time_type == 'pm_rush':
                time_characteristics.append('evening_peak')
            elif time_type in ['am_rush', 'pm_rush']:
                time_characteristics.append('peak_hour')
            elif time_type == 'day':
                time_characteristics.append('midday')
            elif time_type == 'night':
                time_characteristics.append('late_night')
            elif time_type == 'weekend':
                time_characteristics.append('weekend_service')
            
            time_type_details[time_type] = {
                # Basic info
                'total_records': len(time_data),
                'unique_trips': time_data['trip_id'].nunique(),
                
                # Trip type contributions - KEY INFO for understanding data composition
                'contributing_trip_types': contributing_trip_types,
                'trip_type_record_counts': trip_type_breakdown
            }
        
        return time_type_details

    def create_master_log_indexer(self):
        """Create master log indexer - flags based on presence in logs (combination level)"""
        print("\n=== CREATING MASTER LOG INDEXER (COMBINATION LEVEL) ===")
        
        master_log_indexer = {}
        
        # Get all valid combinations at route_stop_direction_trip_type level
        combinations = self.df_final.groupby([
            'stop_id', 'stop_name', 'direction_id', 'trip_type'
        ]).agg({
            'time_type': lambda x: sorted(list(x.unique())),
        }).size().reset_index(name='total_records')
        
        # Add time_types_list column
        time_types_by_combo = self.df_final.groupby([
            'stop_id', 'stop_name', 'direction_id', 'trip_type'
        ])['time_type'].apply(lambda x: sorted(list(x.unique()))).reset_index()
        
        combinations = combinations.merge(
            time_types_by_combo, 
            on=['stop_id', 'stop_name', 'direction_id', 'trip_type']
        )
        combinations.columns = [
            'stop_id', 'stop_name', 'direction_id', 'trip_type', 
            'total_records', 'time_types_list'
        ]
        
        print(f"Creating log indexer for {len(combinations)} combinations")
        
        for _, combo in combinations.iterrows():
            stop_id = str(combo['stop_id'])
            direction_id = str(combo['direction_id'])
            trip_type = combo['trip_type']
            
            indexer_key = self.get_combination_key(stop_id, direction_id, trip_type)
            
            # ===== FLAGS BASED PURELY ON LOG PRESENCE =====
            
            # Check regulatory status (from regulatory stops log)
            is_regulatory = indexer_key in getattr(self, '_regulatory_stops_log', {})
            
            # Check violation flags (from violation logs)
            has_topology_violation = indexer_key in getattr(self, '_topology_violations_log', {})
            has_pattern_violation = indexer_key in getattr(self, '_pattern_violations_log', {})
            has_regulatory_violation = indexer_key in getattr(self, '_regulatory_violations_log', {})
            
            # Get severity levels (if violations exist)
            topology_severity = None
            pattern_severity = None
            regulatory_severity = None
            
            if has_topology_violation:
                topology_severity = self._topology_violations_log[indexer_key].get('severity')
            
            if has_pattern_violation:
                pattern_severity = self._pattern_violations_log[indexer_key].get('severity')
                
            if has_regulatory_violation:
                regulatory_severity = self._regulatory_violations_log[indexer_key].get('severity')
            
            # Determine overall severity
            all_severities = [s for s in [topology_severity, pattern_severity, regulatory_severity] if s]
            overall_severity = None
            if all_severities:
                if 'high' in all_severities:
                    overall_severity = 'high'
                elif 'medium' in all_severities:
                    overall_severity = 'medium'
                else:
                    overall_severity = 'low'
            
            # Create master log indexer entry
            master_log_indexer[indexer_key] = {
                # Basic identifiers
                'route_id': str(self.route_id),
                'route_short_name': self.route_short_name,
                'stop_id': stop_id,
                'stop_name': combo['stop_name'],
                'direction_id': direction_id,
                'trip_type': trip_type,
                'total_records': combo['total_records'],
                'time_types': combo['time_types_list'],
                
                # ===== SINGLE SOURCE OF TRUTH FLAGS =====
                # These flags correspond 1:1 with log presence
                
                'is_regulatory': is_regulatory,
                'has_topology_violation': has_topology_violation,
                'has_pattern_violation': has_pattern_violation,
                'has_regulatory_violation': has_regulatory_violation,
                
                # Additional useful flags
                'has_any_violation': has_topology_violation or has_pattern_violation or has_regulatory_violation,
                'violation_count': sum([has_topology_violation, has_pattern_violation, has_regulatory_violation]),
                'overall_severity': overall_severity,
                
                # Severity breakdown
                'severity_details': {
                    'topology_severity': topology_severity,
                    'pattern_severity': pattern_severity,
                    'regulatory_severity': regulatory_severity
                }
            }
        
        self._master_log_indexer = master_log_indexer
        
        # Statistics
        total_combinations = len(master_log_indexer)
        regulatory_count = sum(1 for entry in master_log_indexer.values() if entry['is_regulatory'])
        violation_count = sum(1 for entry in master_log_indexer.values() if entry['has_any_violation'])
        
        print(f"Master log indexer created: {total_combinations} combinations")
        print(f"  - Regulatory combinations: {regulatory_count}")
        print(f"  - Combinations with violations: {violation_count}")
        print(f"✅ Flags correspond 1:1 with log presence")
        
        return master_log_indexer

    def create_master_analysis_indexer(self):
        """Create master analysis indexer - flags based on analysis availability (time_type level)"""
        print("\n=== CREATING MASTER ANALYSIS INDEXER (TIME_TYPE LEVEL) ===")
        
        master_analysis_indexer = {}
        
        # Get all time_type level combinations (route_stop_direction_time_type)
        analysis_combinations = self.df_final.groupby([
            'stop_id', 'stop_name', 'direction_id', 'time_type'
        ]).agg({
            'trip_type': lambda x: sorted(list(x.unique())),
        }).size().reset_index(name='total_records')
        
        # Add trip_types_list column  
        trip_types_by_combo = self.df_final.groupby([
            'stop_id', 'stop_name', 'direction_id', 'time_type'
        ])['trip_type'].apply(lambda x: sorted(list(x.unique()))).reset_index()
        
        analysis_combinations = analysis_combinations.merge(
            trip_types_by_combo, 
            on=['stop_id', 'stop_name', 'direction_id', 'time_type']
        )
        analysis_combinations.columns = [
            'stop_id', 'stop_name', 'direction_id', 'time_type', 
            'total_records', 'trip_types_list'
        ]
        
        print(f"Creating analysis indexer for {len(analysis_combinations)} time_type combinations")
        
        for _, combo in analysis_combinations.iterrows():
            stop_id = str(combo['stop_id'])
            direction_id = str(combo['direction_id'])
            time_type = combo['time_type']
            
            # Time_type level key format
            indexer_key = f"{self.route_id}_{stop_id}_{direction_id}_{time_type}"
            
            # ===== FLAGS BASED PURELY ON ANALYSIS AVAILABILITY =====
            
            # Check histogram availability
            has_histograms = indexer_key in getattr(self, '_histograms_log', {})
            
            # Check punctuality analysis availability
            has_punctuality = indexer_key in getattr(self, '_punctuality_log', {})
            
            # Get analysis details if available
            histogram_types = []
            histogram_sample_size = 0
            punctuality_sample_size = 0
            on_time_performance = None
            
            if has_histograms:
                histogram_data = self._histograms_log[indexer_key]
                histogram_types = list(histogram_data.get('histograms', {}).keys())
                histogram_sample_size = histogram_data.get('metadata', {}).get('total_sample_size', 0)
            
            if has_punctuality:
                punctuality_data = self._punctuality_log[indexer_key]
                punctuality_sample_size = punctuality_data.get('metadata', {}).get('sample_size', 0)
                on_time_performance = punctuality_data.get('punctuality_metrics', {}).get('performance_indicators', {}).get('on_time_performance_percent')
            
            # Performance assessment (if punctuality data available)
            performance_level = None
            if on_time_performance is not None:
                if on_time_performance >= 85:
                    performance_level = 'excellent'
                elif on_time_performance >= 70:
                    performance_level = 'good'
                elif on_time_performance >= 50:
                    performance_level = 'fair'
                else:
                    performance_level = 'poor'
            
            # Create master analysis indexer entry
            master_analysis_indexer[indexer_key] = {
                # Basic identifiers
                'route_id': str(self.route_id),
                'route_short_name': self.route_short_name,
                'stop_id': stop_id,
                'stop_name': combo['stop_name'],
                'direction_id': direction_id,
                'time_type': time_type,
                'total_records': combo['total_records'],
                'trip_types': combo['trip_types_list'],
                
                # ===== SINGLE SOURCE OF TRUTH ANALYSIS FLAGS =====
                # These flags correspond 1:1 with analysis log presence
                
                'has_histograms': has_histograms,
                'has_punctuality': has_punctuality,
                'has_any_analysis': has_histograms or has_punctuality,
                
                # Analysis details
                'analysis_details': {
                    'histogram_types': histogram_types,
                    'histogram_sample_size': histogram_sample_size,
                    'punctuality_sample_size': punctuality_sample_size,
                    'on_time_performance_percent': on_time_performance,
                    'performance_level': performance_level
                }
            }
        
        self._master_analysis_indexer = master_analysis_indexer
        
        # Statistics
        total_analysis_combinations = len(master_analysis_indexer)
        histogram_count = sum(1 for entry in master_analysis_indexer.values() if entry['has_histograms'])
        punctuality_count = sum(1 for entry in master_analysis_indexer.values() if entry['has_punctuality'])
        
        print(f"Master analysis indexer created: {total_analysis_combinations} time_type combinations")
        print(f"  - With histogram analysis: {histogram_count}")
        print(f"  - With punctuality analysis: {punctuality_count}")
        print(f"✅ Flags correspond 1:1 with analysis availability")
        
        return master_analysis_indexer

    def _sort_navigation_arrays(self, stop_to_combinations, route_to_combinations):
        """Sort all arrays in navigation structures for consistency"""
        
        time_order = ['am_rush', 'day', 'pm_rush', 'night', 'weekend']
        trip_type_order = ['full', 'partial_1', 'partial_2', 'partial_3', 'partial_4', 'partial_5']
        
        def sort_time_types(time_types_list):
            return sorted(time_types_list, key=lambda x: time_order.index(x) if x in time_order else 999)
        
        def sort_trip_types(trip_types_list):
            return sorted(trip_types_list, key=lambda x: trip_type_order.index(x) if x in trip_type_order else 999)
        
        # Sort route_to_combinations
        for route_id in route_to_combinations:
            if "directions" in route_to_combinations[route_id]:
                route_to_combinations[route_id]["directions"].sort()
            
            for direction_id in route_to_combinations[route_id]:
                if direction_id in ["directions", "route_name", "route_short_name"]:
                    continue
                
                if "stop_ids" in route_to_combinations[route_id][direction_id]:
                    route_to_combinations[route_id][direction_id]["stop_ids"].sort()
                
                for stop_id in route_to_combinations[route_id][direction_id].get("stop_ids", []):
                    if stop_id in route_to_combinations[route_id][direction_id]:
                        stop_data = route_to_combinations[route_id][direction_id][stop_id]
                        
                        if "trip_types" in stop_data:
                            stop_data["trip_types"] = sort_trip_types(stop_data["trip_types"])
                        if "time_types" in stop_data:
                            stop_data["time_types"] = sort_time_types(stop_data["time_types"])
        
        # Sort stop_to_combinations (same logic)
        for stop_id in stop_to_combinations:
            if "routes" in stop_to_combinations[stop_id]:
                stop_to_combinations[stop_id]["routes"].sort()
            
            for route_id in stop_to_combinations[stop_id]:
                if route_id in ["routes", "stop_name"]:
                    continue
                
                if "directions" in stop_to_combinations[stop_id][route_id]:
                    stop_to_combinations[stop_id][route_id]["directions"].sort()
                
                for direction_id in stop_to_combinations[stop_id][route_id]:
                    if direction_id in ["directions", "route_name", "route_short_name"]:
                        continue
                    
                    direction_data = stop_to_combinations[stop_id][route_id][direction_id]
                    
                    if "trip_types" in direction_data:
                        direction_data["trip_types"] = sort_trip_types(direction_data["trip_types"])
                    if "time_types" in direction_data:
                        direction_data["time_types"] = sort_time_types(direction_data["time_types"])

    def export_all_data(self):
        """Export all data as JSON files with simple global merging"""
        print("\n=== EXPORTING ALL DATA ===")
        
        # Simple JSON serializer
        def clean_for_json(obj):
            """Convert objects to JSON-serializable format"""
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, (dict, list)):
                return obj
            elif isinstance(obj, set):
                return list(obj)
            else:
                return str(obj)
        
        # Simple file operations
        def load_json(file_path):
            """Load existing JSON file or return empty dict"""
            try:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {file_path.name}: {e}")
            return {}
        
        def save_json(data, file_path):
            """Save data as JSON file"""
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=clean_for_json)
        
        def merge_route_data(existing, new, route_id):
            """Merge route data by removing old route entries and adding new ones"""
            # Remove old entries for this route
            cleaned = {
                key: value for key, value in existing.items()
                if not (key.startswith(f"{route_id}_") or 
                    (isinstance(value, dict) and str(value.get('route_id')) == str(route_id)))
            }
            # Add new entries
            cleaned.update(new)
            return cleaned
        
        def merge_navigation(existing, new):
            """Simple navigation merge - just update with new data"""
            if not existing:
                return new
            
            # Deep merge for nested structures
            result = existing.copy()
            for key, value in new.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key].update(value)
                else:
                    result[key] = value
            return result
        
        # Setup output folder
        output_folder = Path('transit_analysis_global')
        output_folder.mkdir(exist_ok=True)
        
        # Define what to export
        log_files = {
            'master_log_indexer.json': self._master_log_indexer,
            'master_analysis_indexer.json': getattr(self, '_master_analysis_indexer', {}),
            'log_topology_violations.json': self._topology_violations_log,
            'log_pattern_violations.json': self._pattern_violations_log,
            'log_regulatory_violations.json': self._regulatory_violations_log,
            'log_trip_types.json': self._trip_types_log,
            'log_regulatory_stops.json': self._regulatory_stops_log,
            'log_histograms.json': self._histograms_log,
            'log_punctuality.json': self._punctuality_log,
        }
        
        navigation_files = {
            'global_stop_to_combinations.json': self._navigation_maps.get('stop_to_combinations', {}),
            'global_route_to_combinations.json': self._navigation_maps.get('route_to_combinations', {}),
            'global_stop_name_to_stop_ids.json': self._navigation_maps.get('stop_name_to_stop_ids', {}),
            'global_route_short_name_to_info.json': self._create_route_mapping()
        }
        
        exported_files = []
        
        # Export log files (with route-specific merging)
        for filename, data in log_files.items():
            if not data:  # Skip empty data
                continue
                
            file_path = output_folder / filename
            existing = load_json(file_path)
            merged = merge_route_data(existing, data, self.route_id)
            save_json(merged, file_path)
            exported_files.append(filename)
            
            print(f"✅ {filename}: {len(data)} entries")
        
        # Export navigation files (accumulative merging)
        for filename, data in navigation_files.items():
            if not data:
                continue
                
            file_path = output_folder / filename
            existing = load_json(file_path)
            merged = merge_navigation(existing, data)
            save_json(merged, file_path)
            exported_files.append(filename)
            
            print(f"✅ {filename}: Navigation data")
        
        # Create simple summary
        summary = {
            'data_summary': {
                'total_routes_analyzed': 1,  # Will be aggregated when multiple routes processed
                'total_stop_names': len(self._stop_name_to_stop_ids),
                'total_unique_stop_ids': len(set(entry['stop_id'] for entry in self._master_log_indexer.values())),
                'total_combinations': len(self._master_log_indexer),
                'has_topology_violations': len(self._topology_violations_log) > 0,
                'has_pattern_violations': len(self._pattern_violations_log) > 0,
                'has_regulatory_violations': len(self._regulatory_violations_log) > 0,
                'has_any_violations': any([
                    len(self._topology_violations_log) > 0,
                    len(self._pattern_violations_log) > 0,
                    len(self._regulatory_violations_log) > 0
                ])
            },
            'files_exported': exported_files,
            'route_info': {
                'route_id': self.route_id,
                'route_short_name': self.route_short_name,
                'route_long_name': self.route_long_name
            }
        }
        
        # Simple summary merging (just aggregate counts)
        summary_file = output_folder / 'global_summary.json'
        existing_summary = load_json(summary_file)
        
        if existing_summary:
            # Aggregate the counts
            existing_data = existing_summary.get('data_summary', {})
            new_data = summary['data_summary']
            
            summary['data_summary'] = {
                'total_routes_analyzed': existing_data.get('total_routes_analyzed', 0) + 1,
                'total_stop_names': existing_data.get('total_stop_names', 0) + new_data['total_stop_names'],
                'total_unique_stop_ids': existing_data.get('total_unique_stop_ids', 0) + new_data['total_unique_stop_ids'],
                'total_combinations': existing_data.get('total_combinations', 0) + new_data['total_combinations'],
                'has_topology_violations': existing_data.get('has_topology_violations', False) or new_data['has_topology_violations'],
                'has_pattern_violations': existing_data.get('has_pattern_violations', False) or new_data['has_pattern_violations'],
                'has_regulatory_violations': existing_data.get('has_regulatory_violations', False) or new_data['has_regulatory_violations'],
                'has_any_violations': existing_data.get('has_any_violations', False) or new_data['has_any_violations']
            }
        
        save_json(summary, summary_file)
        
        # Clean output
        total_combinations = summary['data_summary']['total_combinations']
        has_violations = summary['data_summary']['has_any_violations']
        
        print(f"\n🌍 EXPORT COMPLETE:")
        print(f"  - Route: {self.route_short_name}")
        print(f"  - Combinations: {total_combinations}")
        print(f"  - Issues: {'Yes' if has_violations else 'No'}")
        print(f"  - Files: {output_folder}")
        
        return {
            'output_folder': str(output_folder),
            'summary_file': str(summary_file),
            'combinations': total_combinations,
            'files_exported': exported_files
        }


    def has_violations_for_combination(self, stop_id, direction_id, trip_type):
        """Check if combination has any violations"""
        combo_key = self.get_combination_key(stop_id, direction_id, trip_type)
        return any([
            combo_key in self._topology_violations_log,
            combo_key in self._pattern_violations_log,
            combo_key in self._regulatory_violations_log
        ])

    def has_analysis_for_combination(self, stop_id, direction_id, time_type):
        """Check if combination has analysis data"""
        analysis_key = f"{self.route_id}_{stop_id}_{direction_id}_{time_type}"
        return any([
            analysis_key in self._histograms_log,
            analysis_key in self._punctuality_log
        ])
