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
        self._trip_types_log = {}  
        self._regulatory_stops_log = {}  

        self._topology_violations_log = {}       
        self._pattern_violations_log = {}
        self._regulatory_violations_log = {}
        self._histograms_log = {}
        self._punctuality_log = {}
        
        # Master indexer
        self._master_indexer = {}
        
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

        # Create master indexer after all processing
        self.create_master_indexer()
        
        # STEP 7: Create analysis master indexer (time_type level)
        self.create_analysis_master_indexer()
        
        # Create navigation and export
        self.create_unified_navigation_maps()
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

    def get_combination_key(self, stop_id, direction_id, trip_type):
        """Generate standardized combination key for all violation logs and master indexer"""
        return f"{self.route_id}_{stop_id}_{direction_id}_{trip_type}"

    def get_combination_key_from_row(self, row):
        """Generate standardized combination key from dataframe row"""
        return f"{self.route_id}_{row['stop_id']}_{row['direction_id']}_{row['trip_type']}"


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
            'stop_name': details.get('stop_name'),
            'stop_id': str(details.get('stop_id', '')),
            'direction_id': str(details.get('direction_id', '')),
            'trip_type': details.get('trip_type', ''),
            
            # Original details
            **details
        }

    def add_violation_to_log(self, log_dict, key, violation_entry):
        """Add violation to specified log with consistent key format"""
        log_dict[key] = violation_entry
        return violation_entry

#   ================ Validating / Logging Violating RouteID-DirectionID-StopID Behaviours =====================

    def create_and_validate_stop_topology(self, df):
        """Create stop-to-direction mapping and detect violations, store temp results for later detailed logging"""
        print("\n=== STOP TOPOLOGY VALIDATION ===")
        
        # Build stop_name to stop_ids mapping for NAVIGATION
        stop_name_to_stop_ids = {}
        for _, row in df[['stop_name', 'stop_id']].drop_duplicates().iterrows():
            stop_name = row['stop_name']
            stop_id = str(row['stop_id'])
            
            if stop_name not in stop_name_to_stop_ids:
                stop_name_to_stop_ids[stop_name] = []
            if stop_id not in stop_name_to_stop_ids[stop_name]:
                stop_name_to_stop_ids[stop_name].append(stop_id)
        
        # Sort for consistency
        for stop_name in stop_name_to_stop_ids:
            stop_name_to_stop_ids[stop_name].sort()
        
        print(f"Built mapping for {len(stop_name_to_stop_ids)} stop names")
        
        # ✅ DETECT violations but store them temporarily (without trip_type granularity yet)
        temp_violations = {}  # Temporary storage
        
        for stop_name, stop_ids in stop_name_to_stop_ids.items():
            violations = self._detect_topology_violation(df, stop_ids)
            
            if violations:
                print(f"🚩 {stop_name}: {len(violations)} violation(s) detected")
                
                # Store violations temporarily by stop_id + direction_id
                for violation_info in violations:
                    problematic_stop_ids = violation_info['problematic_stop_ids']
                    violation = violation_info['violation']
                    
                    # Get all direction_ids that exist for these problematic stop_ids
                    affected_directions = df[
                        df['stop_id'].astype(str).isin([str(sid) for sid in problematic_stop_ids])
                    ]['direction_id'].unique()
                    
                    for stop_id in problematic_stop_ids:
                        for direction_id in affected_directions:
                            temp_key = f"{stop_id}_{direction_id}"
                            temp_violations[temp_key] = {
                                'violation': violation,
                                'stop_id': str(stop_id),
                                'direction_id': str(direction_id),
                                'stop_name': stop_name
                            }
                    
                    print(f"  - {violation['violation_type']}: {len(problematic_stop_ids)} stop_ids × {len(affected_directions)} directions")
            else:
                print(f"✅ {stop_name}: Valid mapping")
        
        # Store results
        self._stop_name_to_stop_ids = stop_name_to_stop_ids
        self._temp_topology_violations = temp_violations  # Store temporarily
        
        # Add basic flags to dataframe (will be updated later with trip_type granularity)
        self._add_basic_topology_flags(df, temp_violations)
        
        print(f"Validation complete: {len(temp_violations)} stop_id-direction violations detected (detailed logging deferred)")
        return temp_violations

    def _add_basic_topology_flags(self, df, temp_violations):
        """Add basic topology violation flags to dataframe (before trip_type exists)"""
        
        # Extract flagged combinations from temp violation keys
        flagged_combinations = set()
        critical_combinations = set()
        
        for temp_key, violation_data in temp_violations.items():
            stop_id, direction_id = temp_key.split('_')
            combo = (stop_id, direction_id)
            flagged_combinations.add(combo)
            
            if violation_data['violation'].get('severity') == 'high':
                critical_combinations.add(combo)
        
        # Apply flags
        df['topology_flagged'] = df.apply(
            lambda row: (str(row['stop_id']), str(row['direction_id'])) in flagged_combinations, axis=1
        )
        df['topology_critical'] = df.apply(
            lambda row: (str(row['stop_id']), str(row['direction_id'])) in critical_combinations, axis=1
        )
        
        if flagged_combinations:
            print(f"🚩 Basic flags added: {df['topology_flagged'].sum()} flagged, {df['topology_critical'].sum()} critical records")

    def finalize_topology_violations_log(self, df):
        """Create final topology violations log with trip_type granularity after trip classification"""
        print("\n=== FINALIZING TOPOLOGY VIOLATIONS LOG ===")
        
        if not hasattr(self, '_temp_topology_violations'):
            self._topology_violations_log = {}
            return
        
        topology_violations_log = {}
        
        # Get all unique combinations that exist in the data
        combinations = df.groupby(['stop_id', 'stop_name', 'direction_id', 'trip_type']).size().reset_index()
        
        violation_count = 0
        for temp_key, violation_data in self._temp_topology_violations.items():
            stop_id, direction_id = temp_key.split('_')
            
            # Find all trip_types for this stop_id + direction_id combination
            matching_combos = combinations[
                (combinations['stop_id'].astype(str) == stop_id) & 
                (combinations['direction_id'].astype(str) == direction_id)
            ]
            
            # Create detailed violation entry for each trip_type
            for _, combo in matching_combos.iterrows():
                detailed_key = self.get_combination_key(combo['stop_id'], combo['direction_id'], combo['trip_type'])
                
                # Get total records for this stop_id + direction_id combination
                stop_direction_records = len(df[
                    (df['stop_id'].astype(str) == str(combo['stop_id'])) &
                    (df['direction_id'].astype(str) == str(combo['direction_id']))
                ])

                # Create detailed violation with trip_type information
                detailed_violation = violation_data['violation'].copy()
                detailed_violation.update({
                    'stop_id': str(combo['stop_id']),
                    'direction_id': str(combo['direction_id']),
                    'trip_type': combo['trip_type'],
                    'stop_name': combo['stop_name'],
                    'stop_id_direction_records': stop_direction_records
                })
                
                topology_violations_log[detailed_key] = detailed_violation
                violation_count += 1
        
        # Store final log and clean up temp
        self._topology_violations_log = topology_violations_log
        delattr(self, '_temp_topology_violations')  # Clean up temporary storage
        
        print(f"Finalized {violation_count} detailed topology violations with trip_type granularity")
        return topology_violations_log

    def _detect_topology_violation(self, df, stop_name, stop_ids):
        """Detect topology violations and return violations per problematic stop_id"""
        
        # Analyze direction behavior for each stop_id
        stop_direction_map = {}
        for stop_id in stop_ids:
            direction_counts = df[df['stop_id'] == stop_id]['direction_id'].value_counts().to_dict()
            stop_direction_map[stop_id] = {
                'is_bidirectional': len(direction_counts) > 1,
                'dominant_direction': max(direction_counts.items(), key=lambda x: x[1])[0] if direction_counts else None,
                'counts': direction_counts
            }
        
        num_stop_ids = len(stop_ids)
        bidirectional = [sid for sid, data in stop_direction_map.items() if data['is_bidirectional']]
        directional = [sid for sid, data in stop_direction_map.items() if not data['is_bidirectional']]
        
        violations = []
        
        def create_violation(violation_type, severity, description, problematic_stops, **extra_details):
            """Helper to create violation with standard topology analysis"""
            base_details = {
                'stop_name': stop_name,
                'all_stop_ids_for_stop': stop_ids,
                **extra_details
            }
            
            # Add topology analysis based on number of problematic stops
            if len(problematic_stops) == 1:
                # Single stop - add direction_analysis
                stop_id = problematic_stops[0]
                base_details['problematic_stop_id'] = stop_id
                base_details['direction_analysis'] = {
                    'is_bidirectional': stop_direction_map[stop_id]['is_bidirectional'],
                    'dominant_direction': stop_direction_map[stop_id]['dominant_direction'],
                    'direction_counts': stop_direction_map[stop_id]['counts'],
                    'total_records': sum(stop_direction_map[stop_id]['counts'].values())
                }
            else:
                # Multiple stops - add topology_analysis for all
                base_details['problematic_stop_ids'] = problematic_stops
                base_details['topology_analysis'] = {
                    str(sid): {
                        'is_bidirectional': stop_direction_map[sid]['is_bidirectional'],
                        'dominant_direction': stop_direction_map[sid]['dominant_direction'],
                        'direction_counts': stop_direction_map[sid]['counts'],
                        'total_records': sum(stop_direction_map[sid]['counts'].values())
                    } for sid in problematic_stops
                }
            
            return {
                'problematic_stop_ids': problematic_stops,
                'violation': self.create_violation_entry(violation_type, severity, description, **base_details)
            }
        
        # Case 1: Single stop_id - always valid
        if num_stop_ids == 1:
            return violations
        
        # Case 2: Two stop_ids - should be directional pair
        elif num_stop_ids == 2:
            for stop_id in bidirectional:
                contamination_rate = (
                    sum(stop_direction_map[stop_id]['counts'].values()) - 
                    max(stop_direction_map[stop_id]['counts'].values())
                ) / sum(stop_direction_map[stop_id]['counts'].values())
                
                violations.append(create_violation(
                    'directional_contamination',
                    'high' if contamination_rate > 0.3 else 'medium',
                    f'Stop_id {stop_id} serves both directions (contamination: {contamination_rate:.1%})',
                    [stop_id],
                    contamination_rate=contamination_rate
                ))
        
        # Case 3: Three stop_ids - should be 1 shared + 2 directional
        elif num_stop_ids == 3:
            if len(bidirectional) > 1:
                # Flag extra shared stops
                for stop_id in bidirectional[1:]:
                    violations.append(create_violation(
                        'unexpected_shared_stop',
                        'high',
                        f'Stop_id {stop_id} is unexpectedly shared (expected only 1 shared stop)',
                        [stop_id],
                        expected_shared_count=1,
                        actual_shared_count=len(bidirectional)
                    ))
            elif len(bidirectional) == 0:
                # Flag missing shared stop
                violations.append(create_violation(
                    'missing_shared_stop',
                    'high',
                    f'Expected 1 shared stop, found 0 (all {len(directional)} are directional)',
                    stop_ids,
                    expected_shared_count=1,
                    actual_shared_count=0
                ))
        
        # Case 4: Four stop_ids - should be 2 pairs of directional
        elif num_stop_ids == 4:
            # Flag unexpected shared stops
            for stop_id in bidirectional:
                violations.append(create_violation(
                    'unexpected_shared_stop',
                    'high',
                    f'Stop_id {stop_id} is unexpectedly shared (expected 0 shared stops)',
                    [stop_id],
                    expected_shared_count=0,
                    actual_shared_count=len(bidirectional)
                ))
            
            # Check directional pairing
            if not bidirectional:
                direction_counts = {}
                for stop_id in directional:
                    direction = stop_direction_map[stop_id]['dominant_direction']
                    direction_counts[direction] = direction_counts.get(direction, 0) + 1
                
                for direction, count in direction_counts.items():
                    if count != 2:
                        problematic_stops = [
                            sid for sid in directional 
                            if stop_direction_map[sid]['dominant_direction'] == direction
                        ]
                        violations.append(create_violation(
                            'improper_directional_pairing',
                            'medium',
                            f'Direction {direction} has {count} stops (expected 2)',
                            problematic_stops,
                            direction=direction,
                            actual_count=count,
                            expected_count=2
                        ))
        
        # Case 5+: More than 4 stop_ids - all are unexpected
        else:
            violations.append(create_violation(
                'unexpected_stop_count',
                'high',
                f'Stop has {num_stop_ids} stop_ids (expected 1-4, all are flagged)',
                stop_ids,
                actual_stop_count=num_stop_ids
            ))
        
        return violations

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
            
            for i, pattern in enumerate(all_patterns):
                pattern_trips = dir_trips[dir_trips['pattern'] == pattern]
                trip_count = len(pattern_trips)
                is_full_length = len(pattern) == max_stops[direction]
                
                # Analyze pattern
                analysis = self._analyze_pattern(pattern, canonical)
                
                # Create trip type
                if pattern == canonical and is_full_length:
                    trip_type = 'full'
                elif not is_full_length:
                    trip_type = f'partial_{len([p for p in all_patterns if len(p) < len(pattern)]) + 1}'
                else:
                    trip_type = f'partial_{i+1}'
                
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
                    
                    # Create stop-level trip type log key
                    trip_type_key = self.get_combination_key(stop_id, direction, trip_type)
                    
                    trip_types_log[trip_type_key] = {
                        'route_id': self.route_id,
                        'route_name': self.route_long_name,
                        'route_short_name': self.route_short_name,
                        'stop_id': stop_id,
                        'stop_name': stop_name,
                        'direction_id': direction,
                        'trip_type': trip_type,
                        'pattern_length': len(pattern),
                        'is_canonical': pattern == canonical and is_full_length,
                        'pattern_violation': analysis['type'] != 'consecutive',
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
                            pattern_issue_type=analysis['type'],
                            canonical_description=canonical_description,
                            problematic_description=pattern_description,
                            invalidates_travel_time=True  # All pattern issues invalidate travel time
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

    # SIMPLE TRAVEL TIME CALCULATOR:
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
    def create_canonical_combinations(self):
        """Create the single source of truth for all valid combinations"""
        print("\n=== CREATING CANONICAL COMBINATIONS (SINGLE SOURCE OF TRUTH) ===")
        
        # Use consistent grouping logic
        combinations = self.df_final.groupby([
            'stop_id', 'stop_name', 'direction_id', 'trip_type'
        ]).agg({
            'time_type': lambda x: sorted(list(x.unique())),  # All time_types for this combination
            'trip_id': 'nunique',  # Total unique trips
            # Add other aggregations you need
        }).reset_index()
        
        combinations.columns = ['stop_id', 'stop_name', 'direction_id', 'trip_type', 'time_types_list', 'total_records']
        
        # Apply any filtering logic consistently
        # combinations = combinations[combinations['total_records'] >= MIN_RECORDS]
        
        self._canonical_combinations = combinations
        print(f"Canonical combinations created: {len(combinations)} total")
        return combinations

    def create_master_indexer(self):
        """Create master indexer using canonical combinations"""
        print("\n=== CREATING MASTER INDEXER FROM CANONICAL COMBINATIONS ===")
        
        if not hasattr(self, '_canonical_combinations'):
            print("No canonical combinations found - creating them first")
            self.create_canonical_combinations()
        
        master_indexer = {}
        
        print(f"Processing {len(self._canonical_combinations)} canonical combinations")
        
        for _, combo in self._canonical_combinations.iterrows():
            stop_id = str(combo['stop_id'])
            direction_id = str(combo['direction_id'])
            trip_type = combo['trip_type']
            stop_name = combo['stop_name']
            
            # Create indexer key
            indexer_key = self.get_combination_key(stop_id, direction_id, trip_type)
            
            # Get data for this specific combination
            combo_data = self.df_final[
                (self.df_final['stop_id'].astype(str) == stop_id) &
                (self.df_final['direction_id'].astype(str) == direction_id) &
                (self.df_final['trip_type'] == trip_type)
            ]
            
            if len(combo_data) == 0:
                print(f"Warning: No data found for combination {indexer_key}")
                continue
            
            # Get all time_types for this combination
            time_type_data = {}
            total_records = 0
            
            for time_type in combo_data['time_type'].unique():
                time_group = combo_data[combo_data['time_type'] == time_type]
                time_type_data[time_type] = {
                    'record_count': len(time_group),
                    'is_regulatory': bool(time_group['is_regulatory'].iloc[0]) if 'is_regulatory' in time_group.columns else False,
                    'travel_time_valid': bool(time_group['travel_time_valid'].iloc[0]) if 'travel_time_valid' in time_group.columns else True,
                    'topology_flagged': bool(time_group['topology_flagged'].iloc[0]) if 'topology_flagged' in time_group.columns else False,
                    'topology_critical': bool(time_group['topology_critical'].iloc[0]) if 'topology_critical' in time_group.columns else False
                }
                total_records += len(time_group)
            
            # Get analysis information for this combination
            topology_info = self._get_topology_info_for_combination(stop_id, direction_id, trip_type)
            pattern_info = self._get_pattern_info_for_combination(stop_id, direction_id, trip_type)
            regulatory_info = self._get_regulatory_info_for_combination(stop_id, direction_id, trip_type)
            
            # Create master indexer entry
            master_indexer[indexer_key] = {
                # Basic identifiers
                'route_id': str(self.route_id),
                'route_name': self.route_long_name,
                'route_short_name': self.route_short_name,
                'stop_id': stop_id,
                'stop_name': stop_name,
                'direction_id': direction_id,
                'trip_type': trip_type,
                'total_records': total_records,
                
                # Time type breakdown
                'time_types': {
                    'available': sorted(time_type_data.keys(), key=lambda x: ['am_rush', 'day', 'pm_rush', 'night', 'weekend'].index(x) if x in ['am_rush', 'day', 'pm_rush', 'night', 'weekend'] else 999),
                    'data': time_type_data
                },
                
                # Violation flags
                'violation_flags': {
                    'has_topology_violation': topology_info.get('is_flagged', False),
                    'has_pattern_violation': pattern_info.get('has_issues', False),
                    'has_regulatory_violation': regulatory_info.get('has_anomaly', False),
                    'has_gap_before_this_stop': gap_info.get('has_gap', False),
                    'has_any_violation': (
                        topology_info.get('is_flagged', False) or 
                        pattern_info.get('has_issues', False) or 
                        regulatory_info.get('has_anomaly', False) or
                        gap_info.get('has_gap', False)
                    ),
                    'time_types_with_violations': sorted([
                        time_type for time_type in time_type_data.keys()
                    ] if (topology_info.get('is_flagged', False) or 
                        pattern_info.get('has_issues', False) or 
                        regulatory_info.get('has_anomaly', False) or
                        gap_info.get('has_gap', False)) else [], 
                    key=lambda x: ['am_rush', 'day', 'pm_rush', 'night', 'weekend'].index(x) if x in ['am_rush', 'day', 'pm_rush', 'night', 'weekend'] else 999)
                },
                
                # Violation details by time_type
                'violation_details_by_time_type': {
                    time_type: {
                        'topology': {
                            'severity': topology_info.get('severity'),
                            'proportion': 1.0
                        } if topology_info.get('is_flagged', False) else None,
                        
                        'pattern': {
                            'severity': pattern_info.get('severity'),
                            'proportion': 1.0
                        } if pattern_info.get('has_issues', False) else None,
                        
                        'regulatory': {
                            'severity': regulatory_info.get('severity'),
                            'proportion': 1.0 - regulatory_info.get('regulation_ratio', 1.0)
                        } if regulatory_info.get('has_anomaly', False) else None,
                        
                        'gap': {
                            'severity': gap_info.get('severity'),
                            'proportion': 1.0
                        } if gap_info.get('has_gap', False) else None,
                        
                        'travel_time_unreliable': not time_data.get('travel_time_valid', True)
                    }
                    for time_type, time_data in time_type_data.items()
                },
                
                # Overall severity
                'overall_severity': self._determine_overall_severity(
                    topology_info.get('severity'),
                    pattern_info.get('severity'),
                    regulatory_info.get('severity'),
                    gap_info.get('severity')
                )
            }
        
        self._master_indexer = master_indexer
        
        # Create summary statistics
        total_combinations = len(master_indexer)
        flagged_combinations = sum(1 for entry in master_indexer.values() 
                                if entry['violation_flags']['has_any_violation'])
        
        severity_counts = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
        for entry in master_indexer.values():
            severity = entry.get('overall_severity', 'none')
            severity_counts[severity] += 1
        
        print(f"Master indexer created: {total_combinations} combinations")
        print(f"  - Flagged: {flagged_combinations} ({flagged_combinations/total_combinations*100:.1f}%)")
        print(f"  - High severity: {severity_counts['high']}")
        print(f"  - Medium severity: {severity_counts['medium']}")
        print(f"  - Low severity: {severity_counts['low']}")
        
        return master_indexer

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

    def _sort_enhanced_navigation(self, stop_to_combinations, route_to_combinations):
        """Sort all arrays in the enhanced navigation structure"""
        
        time_order = ['am_rush', 'day', 'pm_rush', 'night', 'weekend']
        trip_type_order = ['full', 'partial_1', 'partial_2', 'partial_3', 'partial_4', 'partial_5']
        
        def sort_time_types(time_types_list):
            return sorted(time_types_list, key=lambda x: time_order.index(x) if x in time_order else 999)
        
        def sort_trip_types(trip_types_list):
            return sorted(trip_types_list, key=lambda x: trip_type_order.index(x) if x in trip_type_order else 999)
        
        # Sort route_to_combinations
        for route_id in route_to_combinations:
            # Sort directions array
            if "directions" in route_to_combinations[route_id]:
                route_to_combinations[route_id]["directions"].sort()
            
            for direction_id in route_to_combinations[route_id]:
                if direction_id == "directions":
                    continue
                    
                # Sort stop_ids array
                if "stop_ids" in route_to_combinations[route_id][direction_id]:
                    route_to_combinations[route_id][direction_id]["stop_ids"].sort()
                
                for stop_id in route_to_combinations[route_id][direction_id].get("stop_ids", []):
                    if stop_id in route_to_combinations[route_id][direction_id]:
                        stop_data = route_to_combinations[route_id][direction_id][stop_id]
                        
                        # Sort convenience arrays
                        if "trip_types" in stop_data:
                            stop_data["trip_types"] = sort_trip_types(stop_data["trip_types"])
                        if "time_types" in stop_data:
                            stop_data["time_types"] = sort_time_types(stop_data["time_types"])
                        
                        # Sort time_types arrays within each trip_type
                        for trip_type in stop_data.get("trip_types", []):
                            if trip_type in stop_data and isinstance(stop_data[trip_type], dict):
                                if "time_types" in stop_data[trip_type]:
                                    stop_data[trip_type]["time_types"] = sort_time_types(
                                        stop_data[trip_type]["time_types"]
                                    )
                        
                        # Sort available_trip_types arrays within each time_type
                        for time_type in stop_data.get("time_types", []):
                            if time_type in stop_data and isinstance(stop_data[time_type], dict):
                                if "available_trip_types" in stop_data[time_type]:
                                    stop_data[time_type]["available_trip_types"] = sort_trip_types(
                                        stop_data[time_type]["available_trip_types"]
                                    )
        
        # Sort stop_to_combinations (same logic)
        for stop_id in stop_to_combinations:
            # Sort routes array
            if "routes" in stop_to_combinations[stop_id]:
                stop_to_combinations[stop_id]["routes"].sort()
            
            for route_id in stop_to_combinations[stop_id]:
                if route_id == "routes":
                    continue
                    
                # Sort directions array
                if "directions" in stop_to_combinations[stop_id][route_id]:
                    stop_to_combinations[stop_id][route_id]["directions"].sort()
                
                for direction_id in stop_to_combinations[stop_id][route_id]:
                    if direction_id == "directions":
                        continue
                        
                    direction_data = stop_to_combinations[stop_id][route_id][direction_id]
                    
                    # Sort convenience arrays
                    if "trip_types" in direction_data:
                        direction_data["trip_types"] = sort_trip_types(direction_data["trip_types"])
                    if "time_types" in direction_data:
                        direction_data["time_types"] = sort_time_types(direction_data["time_types"])
                    
                    # Sort time_types arrays within each trip_type
                    for trip_type in direction_data.get("trip_types", []):
                        if trip_type in direction_data and isinstance(direction_data[trip_type], dict):
                            if "time_types" in direction_data[trip_type]:
                                direction_data[trip_type]["time_types"] = sort_time_types(
                                    direction_data[trip_type]["time_types"]
                                )
                    
                    # Sort available_trip_types arrays within each time_type
                    for time_type in direction_data.get("time_types", []):
                        if time_type in direction_data and isinstance(direction_data[time_type], dict):
                            if "available_trip_types" in direction_data[time_type]:
                                direction_data[time_type]["available_trip_types"] = sort_trip_types(
                                    direction_data[time_type]["available_trip_types"]
                                )

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

    def create_analysis_master_indexer(self):
        """Create analysis master indexer with time_type level granularity (parallel to violations master indexer)"""
        print("\n=== CREATING ANALYSIS MASTER INDEXER ===")
        
        if not hasattr(self, 'df_final') or len(self.df_final) == 0:
            print("No final dataframe found")
            return
        
        analysis_master_indexer = {}
        
        # Group by stop-direction-time combinations (aggregating across trip_types)
        analysis_groups = self.df_final.groupby(['stop_id', 'stop_name', 'direction_id', 'time_type'])
        
        print(f"Processing {len(analysis_groups)} unique analysis combinations")
        
        for (stop_id, stop_name, direction_id, time_type), group in analysis_groups:
            route_id = str(self.route_id)
            stop_id = str(stop_id)
            direction_id = str(direction_id)
            
            # Create analysis indexer key at time_type level: route_id_stop_id_direction_id_time_type
            analysis_key = f"{route_id}_{stop_id}_{direction_id}_{time_type}"
            
            # Get all trip_types for this time_type combination
            trip_types_data = {}
            total_records = 0
            
            for trip_type in group['trip_type'].unique():
                trip_group = group[group['trip_type'] == trip_type]
                trip_types_data[trip_type] = {
                    'record_count': len(trip_group),
                    'is_regulatory': bool(trip_group['is_regulatory'].iloc[0]) if 'is_regulatory' in trip_group.columns else False,
                    'travel_time_valid': bool(trip_group['travel_time_valid'].iloc[0]) if 'travel_time_valid' in trip_group.columns else True,
                    'topology_flagged': bool(trip_group['topology_flagged'].iloc[0]) if 'topology_flagged' in trip_group.columns else False,
                    'topology_critical': bool(trip_group['topology_critical'].iloc[0]) if 'topology_critical' in trip_group.columns else False
                }
                total_records += len(trip_group)
            
            # Get histogram and punctuality info for this time_type combination
            histogram_info = self._get_histogram_info_for_combination(stop_id, direction_id, time_type)
            punctuality_info = self._get_punctuality_info_for_combination(stop_id, direction_id, time_type)
            
            # Create analysis master indexer entry
            analysis_master_indexer[analysis_key] = {
                # Basic identifiers (time_type level)
                'route_id': route_id,
                'route_name': self.route_long_name,
                'route_short_name': self.route_short_name,
                'stop_id': stop_id,
                'stop_name': stop_name,
                'direction_id': direction_id,
                'time_type': time_type,
                'total_records': total_records,
                
                # Trip types breakdown (which trip_types contribute to this time_type)
                'trip_types': {
                    'available': sorted(trip_types_data.keys(), key=lambda x: ['full', 'partial_1', 'partial_2', 'partial_3', 'partial_4', 'partial_5'].index(x) if x in ['full', 'partial_1', 'partial_2', 'partial_3', 'partial_4', 'partial_5'] else 999),
                    'data': trip_types_data
                },
                
                # Analysis availability flags
                'analysis_flags': {
                    'has_histogram_data': histogram_info['has_histograms'],
                    'has_punctuality_data': punctuality_info['has_punctuality'],
                    'histogram_types': histogram_info['histogram_types'],
                    'histogram_sample_size': histogram_info['sample_size'],
                    'punctuality_sample_size': punctuality_info['sample_size'],
                    'on_time_performance': punctuality_info['on_time_performance'],
                    'performance_level': punctuality_info['performance_level']
                },
                
                # Aggregated violation indicators (rolled up from trip_types)
                'violation_indicators': {
                    'any_trip_type_flagged': any(data.get('topology_flagged', False) for data in trip_types_data.values()),
                    'any_trip_type_critical': any(data.get('topology_critical', False) for data in trip_types_data.values()),
                    'any_trip_type_regulatory': any(data.get('is_regulatory', False) for data in trip_types_data.values()),
                    'all_trip_types_have_valid_travel': all(data.get('travel_time_valid', True) for data in trip_types_data.values()),
                    'flagged_trip_types': [tt for tt, data in trip_types_data.items() if data.get('topology_flagged', False)],
                    'regulatory_trip_types': [tt for tt, data in trip_types_data.items() if data.get('is_regulatory', False)]
                },
                
                # Direct log keys for easy lookup
                'log_keys': {
                    'histogram_key': f"{route_id}_{stop_id}_{direction_id}_{time_type}" if histogram_info['has_histograms'] else None,
                    'punctuality_key': f"{route_id}_{stop_id}_{direction_id}_{time_type}" if punctuality_info['has_punctuality'] else None,
                    'violation_keys': [f"{route_id}_{stop_id}_{direction_id}_{tt}" for tt in trip_types_data.keys()]  # Links to violation master indexer
                }
            }
        
        # Store analysis master indexer
        self._analysis_master_indexer = analysis_master_indexer
        
        # Create summary statistics
        total_combinations = len(analysis_master_indexer)
        with_histograms = sum(1 for entry in analysis_master_indexer.values() 
                            if entry['analysis_flags']['has_histogram_data'])
        with_punctuality = sum(1 for entry in analysis_master_indexer.values() 
                            if entry['analysis_flags']['has_punctuality_data'])
        with_violations = sum(1 for entry in analysis_master_indexer.values() 
                            if entry['violation_indicators']['any_trip_type_flagged'])
        
        print(f"Analysis master indexer created: {total_combinations} time_type combinations")
        print(f"  - With histograms: {with_histograms} ({with_histograms/total_combinations*100:.1f}%)")
        print(f"  - With punctuality: {with_punctuality} ({with_punctuality/total_combinations*100:.1f}%)")
        print(f"  - With violations: {with_violations} ({with_violations/total_combinations*100:.1f}%)")
        
        return analysis_master_indexer

# 2. ADD MISSING HELPER METHODS FOR ANALYSIS MASTER INDEXER
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


    def create_unified_navigation_maps(self):
        """Create unified navigation maps using canonical combinations for single source of truth"""
        print("\n=== CREATING UNIFIED NAVIGATION MAPS FROM CANONICAL COMBINATIONS ===")
        
        if not hasattr(self, '_canonical_combinations'):
            print("No canonical combinations found - creating them first")
            self.create_canonical_combinations()
        
        stop_to_combinations = {}
        route_to_combinations = {}
        
        # Use clean stop mapping from topology validation
        if hasattr(self, '_stop_name_to_stop_ids') and self._stop_name_to_stop_ids:
            print("Using topology validation results for stop mapping")
            stop_name_to_stop_ids = self._stop_name_to_stop_ids
        else:
            print("Warning: No topology validation results found. Creating basic mapping...")
            stop_name_to_stop_ids = {}
            stop_name_mapping = self.df_final[['stop_name', 'stop_id']].drop_duplicates()
            for _, row in stop_name_mapping.iterrows():
                stop_name = row['stop_name']
                stop_id = str(row['stop_id'])
                
                if stop_name not in stop_name_to_stop_ids:
                    stop_name_to_stop_ids[stop_name] = []
                if stop_id not in stop_name_to_stop_ids[stop_name]:
                    stop_name_to_stop_ids[stop_name].append(stop_id)
            
            for stop_name in stop_name_to_stop_ids:
                stop_name_to_stop_ids[stop_name].sort()
            
            self._stop_name_to_stop_ids = stop_name_to_stop_ids
        
        # Expand canonical combinations to time_type level for navigation
        print(f"Expanding {len(self._canonical_combinations)} canonical combinations to include time_type level")
        expanded_combinations = []
        
        for _, combo in self._canonical_combinations.iterrows():
            stop_id = str(combo['stop_id'])
            direction_id = str(combo['direction_id'])
            trip_type = combo['trip_type']
            time_types_list = combo['time_types_list']
            
            for time_type in time_types_list:
                expanded_combinations.append({
                    'stop_id': stop_id,
                    'stop_name': combo['stop_name'],
                    'direction_id': direction_id,
                    'trip_type': trip_type,
                    'time_type': time_type,
                    'total_records': combo['total_records']
                })
        
        print(f"Expanded to {len(expanded_combinations)} stop-direction-trip_type-time_type combinations")
        
        # Build enhanced navigation structures from expanded combinations
        for combo in expanded_combinations:
            stop_id = combo['stop_id']
            direction_id = combo['direction_id']
            time_type = combo['time_type']
            trip_type = combo['trip_type']
            stop_name = combo['stop_name']
            route_id = str(self.route_id)
            
            # ===========================
            # ROUTE_TO_COMBINATIONS STRUCTURE
            # ===========================
            if route_id not in route_to_combinations:
                route_to_combinations[route_id] = {"directions": []}
            
            if direction_id not in route_to_combinations[route_id]:
                route_to_combinations[route_id][direction_id] = {"stop_ids": []}
                if direction_id not in route_to_combinations[route_id]["directions"]:
                    route_to_combinations[route_id]["directions"].append(direction_id)
            
            if stop_id not in route_to_combinations[route_id][direction_id]:
                route_to_combinations[route_id][direction_id][stop_id] = {}
                if stop_id not in route_to_combinations[route_id][direction_id]["stop_ids"]:
                    route_to_combinations[route_id][direction_id]["stop_ids"].append(stop_id)
            
            route_stop_data = route_to_combinations[route_id][direction_id][stop_id]
            
            # Ensure convenience arrays exist at stop level
            if "trip_types" not in route_stop_data:
                route_stop_data["trip_types"] = []
            if "time_types" not in route_stop_data:
                route_stop_data["time_types"] = []
            
            # Add to convenience arrays
            if trip_type not in route_stop_data["trip_types"]:
                route_stop_data["trip_types"].append(trip_type)
            if time_type not in route_stop_data["time_types"]:
                route_stop_data["time_types"].append(time_type)
            
            # ===========================
            # TRIP_TYPE DATA (VIOLATION INFO) - Use Master Indexer as Source of Truth
            # ===========================
            if trip_type not in route_stop_data:
                # Get data from master indexer (single source of truth)
                master_key = self.get_combination_key(stop_id, direction_id, trip_type)
                master_data = self._master_indexer.get(master_key, {})
                
                # Use the COMPLETE master indexer entry
                route_stop_data[trip_type] = master_data.copy()
                    
                # Add navigation-specific convenience fields
                route_stop_data[trip_type].update({
                    "time_types": [],  # Will be populated as we process
                    # Any other navigation-specific helpers
                })
            
            # Add time_type to trip_type's time_types array
            if time_type not in route_stop_data[trip_type]["time_types"]:
                route_stop_data[trip_type]["time_types"].append(time_type)
            
            # ===========================
            # TIME_TYPE DATA (ANALYSIS INFO)
            # ===========================
            if time_type not in route_stop_data:
                # Get analysis info for this time_type (aggregated across trip_types)
                histogram_info = self._get_histogram_info_for_combination(stop_id, direction_id, time_type)
                punctuality_info = self._get_punctuality_info_for_combination(stop_id, direction_id, time_type)
                
                # Get violation summary for this time_type (rolled up from all trip_types that contribute)
                contributing_trip_types = [
                    tt for combo_inner in expanded_combinations 
                    if (combo_inner['stop_id'] == stop_id and 
                        combo_inner['direction_id'] == direction_id and 
                        combo_inner['time_type'] == time_type)
                    for tt in [combo_inner['trip_type']]
                ]
                
                flagged_trip_types = []
                regulatory_trip_types = []
                any_trip_type_flagged = False
                
                # Check violations across all contributing trip_types
                for tt in set(contributing_trip_types):  # Remove duplicates
                    master_key = self.get_combination_key(stop_id, direction_id, tt)
                    master_data = self._master_indexer.get(master_key, {})
                    violation_flags = master_data.get('violation_flags', {})
                    
                    if (violation_flags.get('has_topology_violation', False) or 
                        violation_flags.get('has_pattern_violation', False)):
                        flagged_trip_types.append(tt)
                        any_trip_type_flagged = True
                    
                    if self._is_combination_regulatory(stop_id, direction_id, tt):
                        regulatory_trip_types.append(tt)
                
                route_stop_data[time_type] = {
                    "time_type": time_type,
                    "available_trip_types": [],  # Will be populated as we process
                    
                    # ANALYSIS AVAILABILITY FLAGS
                    "has_histogram_data": histogram_info.get('has_histograms', False),
                    "has_punctuality_data": punctuality_info.get('has_punctuality', False),
                    "histogram_types": histogram_info.get('histogram_types', []),
                    "histogram_sample_size": histogram_info.get('sample_size', 0),
                    "punctuality_sample_size": punctuality_info.get('sample_size', 0),
                    "on_time_performance": punctuality_info.get('on_time_performance'),
                    "performance_level": punctuality_info.get('performance_level'),
                    
                    # VIOLATION SUMMARY (rolled up from trip_types)
                    "any_trip_type_flagged": any_trip_type_flagged,
                    "flagged_trip_types": sorted(flagged_trip_types),
                    "regulatory_trip_types": sorted(regulatory_trip_types)
                }
            
            # Add trip_type to time_type's available_trip_types array
            if trip_type not in route_stop_data[time_type]["available_trip_types"]:
                route_stop_data[time_type]["available_trip_types"].append(trip_type)
            
            # ===========================
            # STOP_TO_COMBINATIONS STRUCTURE (MIRROR LOGIC)
            # ===========================
            if stop_id not in stop_to_combinations:
                stop_to_combinations[stop_id] = {"routes": []}
            
            if route_id not in stop_to_combinations[stop_id]:
                stop_to_combinations[stop_id][route_id] = {"directions": []}
                if route_id not in stop_to_combinations[stop_id]["routes"]:
                    stop_to_combinations[stop_id]["routes"].append(route_id)
            
            if direction_id not in stop_to_combinations[stop_id][route_id]:
                stop_to_combinations[stop_id][route_id][direction_id] = {}
                if direction_id not in stop_to_combinations[stop_id][route_id]["directions"]:
                    stop_to_combinations[stop_id][route_id]["directions"].append(direction_id)
            
            stop_direction_data = stop_to_combinations[stop_id][route_id][direction_id]
            
            # Ensure convenience arrays exist
            if "trip_types" not in stop_direction_data:
                stop_direction_data["trip_types"] = []
            if "time_types" not in stop_direction_data:
                stop_direction_data["time_types"] = []
            
            # Add to convenience arrays
            if trip_type not in stop_direction_data["trip_types"]:
                stop_direction_data["trip_types"].append(trip_type)
            if time_type not in stop_direction_data["time_types"]:
                stop_direction_data["time_types"].append(time_type)
            
            # Copy the same trip_type and time_type data from route_to_combinations
            # (to maintain consistency between both navigation structures)
            if trip_type not in stop_direction_data:
                stop_direction_data[trip_type] = route_stop_data[trip_type].copy()
            if time_type not in stop_direction_data:
                stop_direction_data[time_type] = route_stop_data[time_type].copy()
        
        # ===========================
        # SORT EVERYTHING
        # ===========================
        self._sort_enhanced_navigation(stop_to_combinations, route_to_combinations)
        
        # Store results
        self._stop_to_combinations = stop_to_combinations
        self._route_to_combinations = route_to_combinations
        
        print(f"Enhanced navigation maps created:")
        print(f"  - {len(stop_to_combinations)} stops")
        print(f"  - {len(route_to_combinations)} routes")
        print(f"  - {len(stop_name_to_stop_ids)} stop names mapped")
        print(f"  - Using master indexer as single source of truth for violation data")
        
        return {
            'stop_to_combinations': stop_to_combinations,
            'route_to_combinations': route_to_combinations,
            'stop_name_to_stop_ids': stop_name_to_stop_ids
        }

    # Helper methods to extract data from logs (single source of truth)
    def _is_combination_regulatory(self, stop_id, direction_id, trip_type):
        """Check if combination is regulatory from regulatory stops log"""
        key = self.get_combination_key(stop_id, direction_id, trip_type)
        return key in getattr(self, '_regulatory_stops_log', {})

    def _get_pattern_issue_type(self, stop_id, direction_id, trip_type):
        """Get pattern issue type from trip types log"""
        key = self.get_combination_key(stop_id, direction_id, trip_type)
        trip_data = getattr(self, '_trip_types_log', {}).get(key, {})
        return trip_data.get('issue_type', 'consecutive')

    def _get_topology_violation_type(self, stop_id, direction_id, trip_type):
        """Get topology violation type from topology violations log"""
        key = self.get_combination_key(stop_id, direction_id, trip_type)
        violation = getattr(self, '_topology_violations_log', {}).get(key, {})
        return violation.get('violation_type')

    def _get_regulation_ratio(self, stop_id, direction_id, trip_type):
        """Get regulation ratio from regulatory stops log"""
        key = self.get_combination_key(stop_id, direction_id, trip_type)
        reg_data = getattr(self, '_regulatory_stops_log', {}).get(key, {})
        return reg_data.get('zero_seconds_ratio')

    def _determine_overall_severity(self, topology_severity, pattern_severity, regulatory_severity, gap_severity=None):
        """Determine overall severity from individual severities"""
        severities = [s for s in [topology_severity, pattern_severity, regulatory_severity, gap_severity] if s is not None]
        
        if not severities:
            return 'none'
        
        severity_order = ['low', 'medium', 'high']
        for severity in reversed(severity_order):
            if severity in severities:
                return severity
        
        return 'none'

    # ======================================= Export All Data =========================================

    def export_all_data(self):
        """Export master indexer and all violation logs as separate JSON files with global merging"""
        print("\n=== EXPORTING ALL DATA WITH GLOBAL MERGING ===")
        
        def make_json_serializable(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        def load_existing_json(file_path):
            """Load existing JSON file if it exists"""
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Warning: Could not load existing {file_path.name}: {e}")
                    return {}
            return {}
        
        def clean_existing_route_data(existing_data, route_id):
            """Remove all entries for a specific route before adding new data"""
            cleaned = {}
            
            for key, value in existing_data.items():
                # Check if this entry belongs to the route being processed
                if isinstance(value, dict) and 'route_id' in value:
                    # Entry has route_id field - check if it matches
                    if str(value['route_id']) != route_id:
                        cleaned[key] = value  # Keep entries from other routes
                elif key.startswith(f"{route_id}_"):
                    # Key starts with route_id - skip (remove old data for this route)
                    continue
                elif f"_{route_id}_" in key:
                    # Route ID appears in key - skip (remove old data for this route)
                    continue
                else:
                    # Doesn't appear to belong to this route - keep it
                    cleaned[key] = value
            
            return cleaned
        
        def merge_navigation_data(existing_nav, new_nav):
            """Merge navigation data (always accumulative - no route cleanup)"""
            
            def deep_merge_nav_structure(existing, new):
                """Deep merge navigation structures"""
                result = existing.copy()
                
                for key, value in new.items():
                    if key in result:
                        if isinstance(result[key], dict) and isinstance(value, dict):
                            if key == "time_types" and isinstance(value, list):
                                # Special case: time_types is a list - merge and deduplicate
                                existing_times = set(result[key])
                                new_times = set(value)
                                result[key] = sorted(list(existing_times | new_times), 
                                                key=lambda x: ['am_rush', 'day', 'pm_rush', 'night', 'weekend'].index(x) if x in ['am_rush', 'day', 'pm_rush', 'night', 'weekend'] else 999)
                            elif "time_types" in value and isinstance(value["time_types"], list):
                                # Structure with time_types list inside
                                merged_struct = deep_merge_nav_structure(result[key], value)
                                if "time_types" in merged_struct and "time_types" in result[key]:
                                    existing_times = set(result[key]["time_types"])
                                    new_times = set(value["time_types"])
                                    merged_struct["time_types"] = sorted(list(existing_times | new_times),
                                                                    key=lambda x: ['am_rush', 'day', 'pm_rush', 'night', 'weekend'].index(x) if x in ['am_rush', 'day', 'pm_rush', 'night', 'weekend'] else 999)
                                if "stop_ids" in merged_struct and "stop_ids" in result[key]:
                                    existing_stops = set(result[key]["stop_ids"])
                                    new_stops = set(value["stop_ids"])
                                    merged_struct["stop_ids"] = sorted(list(existing_stops | new_stops))
                                result[key] = merged_struct
                            else:
                                result[key] = deep_merge_nav_structure(result[key], value)
                        elif isinstance(result[key], list) and isinstance(value, list):
                            # Merge lists and deduplicate
                            result[key] = sorted(list(set(result[key] + value)))
                        else:
                            # Overwrite with new value
                            result[key] = value
                    else:
                        result[key] = value
                
                return result
            
            return deep_merge_nav_structure(existing_nav, new_nav)
        
        def merge_data(existing_data, new_data, data_type):
            """Merge new data with existing data based on data type"""
            if data_type == 'master_indexer':
                # Master indexer: clean existing route data, then merge
                cleaned = clean_existing_route_data(existing_data, str(self.route_id))
                cleaned.update(new_data)
                return cleaned
            
            elif data_type in ['violations', 'stops', 'trip_types']:
                # Violation logs: clean existing route data, then merge
                cleaned = clean_existing_route_data(existing_data, str(self.route_id))
                cleaned.update(new_data)
                return cleaned
            
            elif data_type == 'navigation':
                # Navigation maps: always accumulate (no cleanup)
                return merge_navigation_data(existing_data, new_data)
            
            elif data_type == 'summary':
                # Summary: aggregate statistics across routes
                if not existing_data:
                    return new_data
                
                existing_summary = existing_data.get('data_summary', {})
                new_summary = new_data['data_summary']
                
                aggregated_summary = {
                    # Additive counts
                    'total_routes_analyzed': existing_summary.get('total_routes_analyzed', 0) + new_summary.get('total_routes_analyzed', 0),
                    'total_stop_names': existing_summary.get('total_stop_names', 0) + new_summary.get('total_stop_names', 0),
                    'total_unique_stop_ids': existing_summary.get('total_unique_stop_ids', 0) + new_summary.get('total_unique_stop_ids', 0),
                    'total_combinations': existing_summary.get('total_combinations', 0) + new_summary.get('total_combinations', 0),
                    
                    # Boolean OR for violation flags (if ANY route has violations, global = true)
                    'has_topology_violations': existing_summary.get('has_topology_violations', False) or new_summary.get('has_topology_violations', False),
                    'has_pattern_violations': existing_summary.get('has_pattern_violations', False) or new_summary.get('has_pattern_violations', False),
                    'has_regulatory_violations': existing_summary.get('has_regulatory_violations', False) or new_summary.get('has_regulatory_violations', False),
                    'has_any_violations': existing_summary.get('has_any_violations', False) or new_summary.get('has_any_violations', False)
                }
                
                return {
                    'data_summary': aggregated_summary,
                    'files_exported': new_data['files_exported']
                }
            
            else:
                # Default: simple merge
                merged = existing_data.copy()
                merged.update(new_data)
                return merged
        
        # Use global output folder (not route-specific)
        output_folder = Path('transit_analysis_global')
        output_folder.mkdir(exist_ok=True)
        
        print(f"Using global output folder: {output_folder}")
        
        # Define all files to export
        files_to_export = {
            'master_indexer.json': ('master_indexer', self._master_indexer),
            'log_topology_violations.json': ('violations', self._topology_violations_log),
            'log_pattern_violations.json': ('violations', self._pattern_violations_log),
            'log_regulatory_violations.json': ('violations', self._regulatory_violations_log),
            'log_trip_types.json': ('trip_types', self._trip_types_log),
            'log_regulatory_stops.json': ('stops', self._regulatory_stops_log),
            # Add to files_to_export in export_all_data()
            'log_histograms.json': ('analysis', self._histograms_log),
            'log_punctuality.json': ('analysis', self._punctuality_log),
        }
        
        exported_files = []
        
        # Export each file with merging
        for filename, (data_type, new_data) in files_to_export.items():
            if not new_data:  # Skip empty data
                continue
                
            file_path = output_folder / filename
            
            # Load existing data
            existing_data = load_existing_json(file_path)
            
            # Merge data
            merged_data = merge_data(existing_data, new_data, data_type)
            
            # Write merged data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(make_json_serializable(merged_data), f, indent=2, ensure_ascii=False, default=str)
            
            exported_files.append(filename)
            
            if existing_data:
                old_count = len(existing_data)
                new_count = len(new_data) 
                merged_count = len(merged_data)
                print(f"✅ {filename}: Route cleanup + merge ({old_count} → {merged_count}, +{new_count} new)")
            else:
                print(f"✅ {filename}: Created new file ({len(new_data)} entries)")
        
        # Export GLOBAL navigation files (always accumulative)
        navigation_files = {
            'global_stop_to_combinations.json': ('navigation', self._stop_to_combinations),
            'global_route_to_combinations.json': ('navigation', self._route_to_combinations),
            'global_stop_name_to_stop_ids.json': ('navigation', self._stop_name_to_stop_ids),
            'global_route_short_name_to_info.json': ('navigation', self._create_route_mapping())
        }
        
        for nav_filename, (nav_data_type, nav_data) in navigation_files.items():
            if not nav_data:
                continue
                
            nav_file_path = output_folder / nav_filename
            
            # Load existing navigation data
            existing_nav_data = load_existing_json(nav_file_path)
            
            # Merge navigation data (always accumulative)
            merged_nav_data = merge_data(existing_nav_data, nav_data, nav_data_type)
            
            # Write merged navigation data
            with open(nav_file_path, 'w', encoding='utf-8') as f:
                json.dump(make_json_serializable(merged_nav_data), f, indent=2, ensure_ascii=False)
            
            exported_files.append(nav_filename)
            
            if existing_nav_data:
                print(f"✅ {nav_filename}: Global navigation accumulated (multi-route)")
            else:
                print(f"✅ {nav_filename}: Created global navigation file")
        
        # Create/update global summary
        route_summary = {
            'data_summary': {
                # High-level scope
                'total_routes_analyzed': 1,
                'total_stop_names': len(self._stop_name_to_stop_ids),
                'total_unique_stop_ids': len(set(entry['stop_id'] for entry in self._master_indexer.values())),
                'total_combinations': len(self._master_indexer),
                
                # Simple violation presence flags
                'has_topology_violations': len(self._topology_violations_log) > 0,
                'has_pattern_violations': len(self._pattern_violations_log) > 0,
                'has_regulatory_violations': len(self._regulatory_violations_log) > 0,
                'has_any_violations': (len(self._topology_violations_log) + 
                                    len(self._pattern_violations_log) + 
                                    len(self._regulatory_violations_log)) > 0
            },
            'files_exported': exported_files
        }
        
        summary_file = output_folder / 'global_summary.json'
        existing_summary = load_existing_json(summary_file)
        merged_summary = merge_data(existing_summary, route_summary, 'summary')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(merged_summary), f, indent=2, ensure_ascii=False)
        
        print(f"✅ Global summary updated: {summary_file}")
        
        # Clean final print
        total_routes = merged_summary.get('data_summary', {}).get('total_routes_analyzed', 0)
        total_stop_names = merged_summary.get('data_summary', {}).get('total_stop_names', 0)
        total_combinations = merged_summary.get('data_summary', {}).get('total_combinations', 0)
        has_violations = merged_summary.get('data_summary', {}).get('has_any_violations', False)

        print(f"\n🌍 GLOBAL ANALYSIS STATUS:")
        print(f"  - Routes analyzed: {total_routes}")
        print(f"  - Stop names: {total_stop_names}")
        print(f"  - Total combinations: {total_combinations}")
        print(f"  - Issues detected: {'Yes' if has_violations else 'No'}")
        print(f"  - Global files: {output_folder}")
        print(f"  - Use master_indexer.json for detailed violation analysis")
        
        return {
            'global_output_folder': str(output_folder),
            'global_summary_file': str(summary_file),
            'total_routes': total_routes,
            'total_combinations': total_combinations
        }

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


    def create_unified_navigation_maps(self):
        """Create unified navigation maps with flattened structure and all violation/analysis data"""
        print("\n=== CREATING UNIFIED NAVIGATION MAPS (ENHANCED WITH ALL DATA) ===")
        
        stop_to_combinations = {}
        route_to_combinations = {}
        
        if not hasattr(self, 'df_final') or len(self.df_final) == 0:
            print("No final dataframe found")
            return
        
        # Use clean stop mapping from topology validation
        if hasattr(self, '_stop_name_to_stop_ids') and self._stop_name_to_stop_ids:
            print("Using topology validation results for stop mapping")
            stop_name_to_stop_ids = self._stop_name_to_stop_ids
        else:
            print("Warning: No topology validation results found. Creating basic mapping...")
            stop_name_to_stop_ids = {}
            stop_name_mapping = self.df_final[['stop_name', 'stop_id']].drop_duplicates()
            for _, row in stop_name_mapping.iterrows():
                stop_name = row['stop_name']
                stop_id = str(row['stop_id'])
                
                if stop_name not in stop_name_to_stop_ids:
                    stop_name_to_stop_ids[stop_name] = []
                if stop_id not in stop_name_to_stop_ids[stop_name]:
                    stop_name_to_stop_ids[stop_name].append(stop_id)
            
            for stop_name in stop_name_to_stop_ids:
                stop_name_to_stop_ids[stop_name].sort()
                
            self._stop_name_to_stop_ids = stop_name_to_stop_ids
        
        # Get unique combinations
        unique_combinations = self.df_final.groupby([
            'stop_id', 'stop_name', 'direction_id', 'time_type', 'trip_type'
        ]).size().reset_index(name='count')
        
        print(f"Processing {len(unique_combinations)} unique data combinations")
        
        # Build enhanced navigation structures
        for _, row in unique_combinations.iterrows():
            stop_id = str(row['stop_id'])
            direction_id = str(row['direction_id'])
            time_type = row['time_type']
            trip_type = row['trip_type']
            route_id = str(self.route_id)
            
            # ===========================
            # ROUTE_TO_COMBINATIONS STRUCTURE
            # ===========================
            if route_id not in route_to_combinations:
                route_to_combinations[route_id] = {"directions": []}
            if direction_id not in route_to_combinations[route_id]:
                route_to_combinations[route_id][direction_id] = {"stop_ids": []}
                if direction_id not in route_to_combinations[route_id]["directions"]:
                    route_to_combinations[route_id]["directions"].append(direction_id)
            if stop_id not in route_to_combinations[route_id][direction_id]:
                route_to_combinations[route_id][direction_id][stop_id] = {}
                if stop_id not in route_to_combinations[route_id][direction_id]["stop_ids"]:
                    route_to_combinations[route_id][direction_id]["stop_ids"].append(stop_id)
            
            route_stop_data = route_to_combinations[route_id][direction_id][stop_id]
            
            # Ensure convenience arrays exist at stop level
            if "trip_types" not in route_stop_data:
                route_stop_data["trip_types"] = []
            if "time_types" not in route_stop_data:
                route_stop_data["time_types"] = []
            
            # Add to convenience arrays
            if trip_type not in route_stop_data["trip_types"]:
                route_stop_data["trip_types"].append(trip_type)
            if time_type not in route_stop_data["time_types"]:
                route_stop_data["time_types"].append(time_type)
            
            # ===========================
            # TRIP_TYPE DATA (VIOLATION INFO)
            # ===========================
            if trip_type not in route_stop_data:
                # Get detailed violation information from all logs
                topology_info = self._get_topology_info_for_combination(stop_id, direction_id, trip_type)
                pattern_info = self._get_pattern_info_for_combination(stop_id, direction_id, trip_type)
                regulatory_info = self._get_regulatory_info_for_combination(stop_id, direction_id, trip_type)
                
                # Get total records from master indexer if available
                master_key = self.get_combination_key(stop_id, direction_id, trip_type)
                master_data = self._master_indexer.get(master_key, {})
                
                route_stop_data[trip_type] = {
                    "trip_type": trip_type,
                    "total_records": master_data.get('total_records', 0),
                    "time_types": [],  # Will be populated as we process
                    
                    # VIOLATION FLAGS (from all violation logs)
                    "has_valid_travel_time": not pattern_info.get('invalidates_travel_time', False),
                    "is_regulatory": regulatory_info.get('has_anomaly', False),  # Will be updated
                    "topology_flagged": topology_info.get('is_flagged', False),
                    "topology_critical": topology_info.get('severity') == 'high',
                    "has_pattern_violation": pattern_info.get('has_issues', False),
                    "has_regulatory_violation": regulatory_info.get('has_anomaly', False),
                    "has_gaps_before_stop": pattern_info.get('has_gaps', False),
                    "has_swaps": pattern_info.get('has_swaps', False),
                    "pattern_issue_type": pattern_info.get('issue_type', 'consecutive'),
                    "overall_severity": self._determine_overall_severity(
                        topology_info.get('severity'),
                        pattern_info.get('severity'),
                        regulatory_info.get('severity')
                    ),
                    
                    # DETAILED VIOLATION INFO (optional)
                    "topological_violation_type": topology_info.get('violation_type'),
                    "regulation_ratio": regulatory_info.get('regulation_ratio')
                }
            
            # Add time_type to trip_type's time_types array
            if time_type not in route_stop_data[trip_type]["time_types"]:
                route_stop_data[trip_type]["time_types"].append(time_type)
            
            # ===========================
            # TIME_TYPE DATA (ANALYSIS INFO)
            # ===========================
            if time_type not in route_stop_data:
                # Get analysis info for this time_type
                histogram_info = self._get_histogram_info_for_combination(stop_id, direction_id, time_type)
                punctuality_info = self._get_punctuality_info_for_combination(stop_id, direction_id, time_type)
                
                # Get violation summary for this time_type (rolled up from trip_types)
                flagged_trip_types = []
                regulatory_trip_types = []
                any_trip_type_flagged = False
                
                # Check all trip_types that contribute to this time_type
                for tt in route_stop_data["trip_types"]:
                    if tt in route_stop_data:  # Trip type already processed
                        if route_stop_data[tt].get("topology_flagged", False) or route_stop_data[tt].get("has_pattern_violation", False):
                            flagged_trip_types.append(tt)
                            any_trip_type_flagged = True
                        if route_stop_data[tt].get("is_regulatory", False):
                            regulatory_trip_types.append(tt)
                
                route_stop_data[time_type] = {
                    "time_type": time_type,
                    "available_trip_types": [],  # Will be populated as we process
                    
                    # ANALYSIS AVAILABILITY FLAGS
                    "has_histogram_data": histogram_info.get('has_histograms', False),
                    "has_punctuality_data": punctuality_info.get('has_punctuality', False),
                    "histogram_types": histogram_info.get('histogram_types', []),
                    "histogram_sample_size": histogram_info.get('sample_size', 0),
                    "punctuality_sample_size": punctuality_info.get('sample_size', 0),
                    "on_time_performance": punctuality_info.get('on_time_performance'),
                    "performance_level": punctuality_info.get('performance_level'),
                    
                    # VIOLATION SUMMARY (rolled up from trip_types)
                    "any_trip_type_flagged": any_trip_type_flagged,
                    "flagged_trip_types": flagged_trip_types,
                    "regulatory_trip_types": regulatory_trip_types
                }
            
            # Add trip_type to time_type's available_trip_types array
            if trip_type not in route_stop_data[time_type]["available_trip_types"]:
                route_stop_data[time_type]["available_trip_types"].append(trip_type)
            
            # ===========================
            # STOP_TO_COMBINATIONS STRUCTURE (SAME LOGIC)
            # ===========================
            if stop_id not in stop_to_combinations:
                stop_to_combinations[stop_id] = {"routes": []}
            if route_id not in stop_to_combinations[stop_id]:
                stop_to_combinations[stop_id][route_id] = {"directions": []}
                if route_id not in stop_to_combinations[stop_id]["routes"]:
                    stop_to_combinations[stop_id]["routes"].append(route_id)
            if direction_id not in stop_to_combinations[stop_id][route_id]:
                stop_to_combinations[stop_id][route_id][direction_id] = {}
                if direction_id not in stop_to_combinations[stop_id][route_id]["directions"]:
                    stop_to_combinations[stop_id][route_id]["directions"].append(direction_id)
            
            stop_direction_data = stop_to_combinations[stop_id][route_id][direction_id]
            
            # Ensure convenience arrays exist at stop level
            if "trip_types" not in stop_direction_data:
                stop_direction_data["trip_types"] = []
            if "time_types" not in stop_direction_data:
                stop_direction_data["time_types"] = []
            
            # Add to convenience arrays
            if trip_type not in stop_direction_data["trip_types"]:
                stop_direction_data["trip_types"].append(trip_type)
            if time_type not in stop_direction_data["time_types"]:
                stop_direction_data["time_types"].append(time_type)
            
            # Copy the same trip_type and time_type data from route_to_combinations
            # (to maintain consistency between both navigation structures)
            if trip_type not in stop_direction_data:
                stop_direction_data[trip_type] = route_stop_data[trip_type].copy()
            if time_type not in stop_direction_data:
                stop_direction_data[time_type] = route_stop_data[time_type].copy()
        
        # ===========================
        # UPDATE REGULATORY FLAGS
        # ===========================
        self._update_regulatory_flags_in_navigation(stop_to_combinations, route_to_combinations)
        
        # ===========================
        # SORT EVERYTHING
        # ===========================
        self._sort_enhanced_navigation(stop_to_combinations, route_to_combinations)
        
        # Store results
        self._stop_to_combinations = stop_to_combinations
        self._route_to_combinations = route_to_combinations
        
        print(f"Enhanced navigation maps created:")
        print(f"  - {len(stop_to_combinations)} stops")
        print(f"  - {len(route_to_combinations)} routes")
        print(f"  - {len(stop_name_to_stop_ids)} stop names mapped")
        print(f"  - Complete violation and analysis data integrated")
        
        return {
            'stop_to_combinations': stop_to_combinations,
            'route_to_combinations': route_to_combinations,
            'stop_name_to_stop_ids': stop_name_to_stop_ids
        }

    def _update_regulatory_flags_in_navigation(self, stop_to_combinations, route_to_combinations):
        """Update regulatory flags based on regulatory stops log"""
        
        if not hasattr(self, '_regulatory_stops_log'):
            return
        
        for reg_key, reg_data in self._regulatory_stops_log.items():
            stop_id = str(reg_data['stop_id'])
            route_id = str(reg_data['route_id'])
            direction_id = str(reg_data['direction_id'])
            trip_type = reg_data['trip_type']
            
            # Update route_to_combinations
            if (route_id in route_to_combinations and 
                direction_id in route_to_combinations[route_id] and
                stop_id in route_to_combinations[route_id][direction_id] and
                trip_type in route_to_combinations[route_id][direction_id][stop_id]):
                
                route_to_combinations[route_id][direction_id][stop_id][trip_type]["is_regulatory"] = True
            
            # Update stop_to_combinations
            if (stop_id in stop_to_combinations and 
                route_id in stop_to_combinations[stop_id] and 
                direction_id in stop_to_combinations[stop_id][route_id] and
                trip_type in stop_to_combinations[stop_id][route_id][direction_id]):
                
                stop_to_combinations[stop_id][route_id][direction_id][trip_type]["is_regulatory"] = True

    def _sort_enhanced_navigation(self, stop_to_combinations, route_to_combinations):
        """Sort all arrays in the enhanced navigation structure"""
        
        time_order = ['am_rush', 'day', 'pm_rush', 'night', 'weekend']
        trip_type_order = ['full', 'partial_1', 'partial_2', 'partial_3', 'partial_4', 'partial_5']
        
        def sort_time_types(time_types_list):
            return sorted(time_types_list, key=lambda x: time_order.index(x) if x in time_order else 999)
        
        def sort_trip_types(trip_types_list):
            return sorted(trip_types_list, key=lambda x: trip_type_order.index(x) if x in trip_type_order else 999)
        
        # Sort route_to_combinations
        for route_id in route_to_combinations:
            # Sort directions array
            route_to_combinations[route_id]["directions"].sort()
            
            for direction_id in route_to_combinations[route_id]:
                if direction_id == "directions":
                    continue
                    
                # Sort stop_ids array
                route_to_combinations[route_id][direction_id]["stop_ids"].sort()
                
                for stop_id in route_to_combinations[route_id][direction_id]["stop_ids"]:
                    if stop_id in route_to_combinations[route_id][direction_id]:
                        stop_data = route_to_combinations[route_id][direction_id][stop_id]
                        
                        # Sort convenience arrays
                        if "trip_types" in stop_data:
                            stop_data["trip_types"] = sort_trip_types(stop_data["trip_types"])
                        if "time_types" in stop_data:
                            stop_data["time_types"] = sort_time_types(stop_data["time_types"])
                        
                        # Sort time_types arrays within each trip_type
                        for trip_type in stop_data.get("trip_types", []):
                            if trip_type in stop_data and isinstance(stop_data[trip_type], dict):
                                if "time_types" in stop_data[trip_type]:
                                    stop_data[trip_type]["time_types"] = sort_time_types(
                                        stop_data[trip_type]["time_types"]
                                    )
                        
                        # Sort available_trip_types arrays within each time_type
                        for time_type in stop_data.get("time_types", []):
                            if time_type in stop_data and isinstance(stop_data[time_type], dict):
                                if "available_trip_types" in stop_data[time_type]:
                                    stop_data[time_type]["available_trip_types"] = sort_trip_types(
                                        stop_data[time_type]["available_trip_types"]
                                    )
        
        # Sort stop_to_combinations (same logic)
        for stop_id in stop_to_combinations:
            # Sort routes array
            stop_to_combinations[stop_id]["routes"].sort()
            
            for route_id in stop_to_combinations[stop_id]:
                if route_id == "routes":
                    continue
                    
                # Sort directions array
                stop_to_combinations[stop_id][route_id]["directions"].sort()
                
                for direction_id in stop_to_combinations[stop_id][route_id]:
                    if direction_id == "directions":
                        continue
                        
                    direction_data = stop_to_combinations[stop_id][route_id][direction_id]
                    
                    # Sort convenience arrays
                    if "trip_types" in direction_data:
                        direction_data["trip_types"] = sort_trip_types(direction_data["trip_types"])
                    if "time_types" in direction_data:
                        direction_data["time_types"] = sort_time_types(direction_data["time_types"])
                    
                    # Sort time_types arrays within each trip_type
                    for trip_type in direction_data.get("trip_types", []):
                        if trip_type in direction_data and isinstance(direction_data[trip_type], dict):
                            if "time_types" in direction_data[trip_type]:
                                direction_data[trip_type]["time_types"] = sort_time_types(
                                    direction_data[trip_type]["time_types"]
                                )
                    
                    # Sort available_trip_types arrays within each time_type
                    for time_type in direction_data.get("time_types", []):
                        if time_type in direction_data and isinstance(direction_data[time_type], dict):
                            if "available_trip_types" in direction_data[time_type]:
                                direction_data[time_type]["available_trip_types"] = sort_trip_types(
                                    direction_data[time_type]["available_trip_types"]
                                )

        """Export travel times to separate JSON file with flat route_stop structure"""
        
        if not hasattr(self, '_travel_times_data') or not self._travel_times_data:
            print("No travel times data to export")
            return
        
        # Load existing travel times data
        travel_times_path = os.path.join(output_dir, "travel_times.json")
        
        def load_existing_json(file_path):
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    return {}
            return {}
        
        existing_travel_times = load_existing_json(travel_times_path)
        
        # Add current route's travel times using flat route_stop keys (like histograms)
        existing_travel_times.update(self._travel_times_data)
        
        # Save updated travel times
        try:
            serializable_data = self._make_json_serializable(existing_travel_times)
            with open(travel_times_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            # Count total travel times for summary (same pattern as histograms)
            total_stops = len(existing_travel_times)
            total_combinations = 0
            total_segments = 0
            
            for stop_data in existing_travel_times.values():
                for direction_time_key, combo_data in stop_data['travel_times'].items():
                    total_combinations += 1
                    total_segments += 1  # Each combination is one travel segment
            
            print(f"  ✓ Saved travel_times.json:")
            print(f"    - {total_stops} stops")
            print(f"    - {total_combinations} direction-time combinations")  
            print(f"    - {total_segments} travel segments")
            
        except Exception as e:
            print(f"  ✗ Error saving travel_times.json: {e}")