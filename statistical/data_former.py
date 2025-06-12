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

        self._topology_violations_log = {}
        self._regulatory_stops_log = {}        
        self._pattern_violations_log = {}
        self._regulatory_violations_log = {}
        
        # Master indexer
        self._master_indexer = {}
        
        # Navigation structures
        self._stop_to_combinations = {}      
        self._route_to_combinations = {}

        # Process data through the pipeline
        self.df_before = self.prepare_columns(raw_data)
        self.create_and_validate_stop_topology(self.df_before)
        self.df_classified = self.identify_and_classify_trips(self.df_before)
        self.df_regulatory = self.identify_and_classify_stops(self.df_classified)
        self.df_ready = self.calculate_travel_times_and_delays(self.df_regulatory)
        self.df_final = self.df_ready
        
        # Create master indexer after all processing
        self.create_master_indexer()
        
        # Create navigation and export
        self.create_new_navigation_maps()
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

#   ====================================== Handle Violation Logging ==========================================

    def create_violation_entry(self, violation_type, severity, description, **details):
        """Create standardized violation entry"""
        return {
            'violation_type': violation_type,
            'severity': severity,
            'description': description,
            'route_id': self.route_id,
            'route_name': self.route_long_name,
            'route_short_name': self.route_short_name,
            **details
        }

    def add_violation_to_log(self, log_dict, key, violation_entry):
        """Add violation to specified log with consistent key format"""
        log_dict[key] = violation_entry
        return violation_entry

#   ================ Validating / Logging Violating RouteID-DirectionID-StopID Behaviours =====================

    def create_and_validate_stop_topology(self, df):
            """Create stop-to-direction mapping and detect violations"""
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
            
            violations_log = {}
            
            for stop_name, stop_ids in stop_name_to_stop_ids.items():
                violation = self._detect_topology_violation(df, stop_name, stop_ids)
                
                if violation:
                    log_key = f"topology_{self.route_long_name}_{stop_name}"
                    self.add_violation_to_log(violations_log, log_key, violation)
                    print(f"🚩 {stop_name}: {violation['violation_type']} ({violation['severity']})")
                else:
                    print(f"✅ {stop_name}: Valid mapping")
            
            # Store results
            self._stop_name_to_stop_ids = stop_name_to_stop_ids
            self._topology_violations_log = violations_log
            
            # Add flags to dataframe
            self._add_topology_flags(df, violations_log)
            
            print(f"Validation complete: {len(violations_log)} violations detected")
            return violations_log

    def _detect_topology_violation(self, df, stop_name, stop_ids):
        """Detect topology violation and return standardized format"""
        
        stop_direction_map = {}
        for stop_id in stop_ids:
            stop_data = df[df['stop_id'] == stop_id]
            direction_counts = stop_data['direction_id'].value_counts().to_dict()
            
            stop_direction_map[stop_id] = {
                'directions': list(direction_counts.keys()),
                'counts': direction_counts,
                'is_bidirectional': len(direction_counts) > 1,
                'dominant_direction': max(direction_counts.items(), key=lambda x: x[1])[0] if direction_counts else None
            }
        
        num_stop_ids = len(stop_ids)
        
        if num_stop_ids == 2:
            bidirectional_stops = [sid for sid, data in stop_direction_map.items() if data['is_bidirectional']]
            
            if bidirectional_stops:
                contamination_rate = max([
                    (sum(stop_direction_map[sid]['counts'].values()) - max(stop_direction_map[sid]['counts'].values())) / 
                    sum(stop_direction_map[sid]['counts'].values()) 
                    for sid in bidirectional_stops
                ])
                
                return self.create_violation_entry(
                    violation_type='directional_pair_contamination',
                    severity='high' if contamination_rate > 0.3 else 'medium',
                    description=f'Stop_ids should serve separate directions but {len(bidirectional_stops)} serve both',
                    stop_name=stop_name,
                    affected_stop_ids=bidirectional_stops,
                    contamination_rate=contamination_rate
                )
        
        elif num_stop_ids >= 5:
            return self.create_violation_entry(
                violation_type='unexpected_stop_count',
                severity='high',
                description=f'Stop has {num_stop_ids} stop_ids (expected 1-4)',
                stop_name=stop_name,
                affected_stop_ids=stop_ids
            )
        
        return None
    
    def _add_topology_flags(self, df, violations_log):
        """Add topology violation flags to dataframe"""
        
        flagged_stops = set()
        critical_stops = set()
        
        for violation in violations_log.values():
            stop_name = violation.get('stop_name', '')
            if not stop_name:
                continue
                
            flagged_stops.add(stop_name)
            if violation.get('severity') == 'high':
                critical_stops.add(stop_name)
        
        df['topology_flagged'] = df['stop_name'].isin(flagged_stops)
        df['topology_critical'] = df['stop_name'].isin(critical_stops)
        
        if flagged_stops:
            flagged_count = df['topology_flagged'].sum()
            critical_count = df['topology_critical'].sum()
            print(f"🚩 Flags added: {flagged_count} flagged, {critical_count} critical records")

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
                    trip_type_key = f"{self.route_long_name}_{stop_id}_{direction}_{trip_type}"
                    
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
                        'has_issues': analysis['type'] != 'consecutive',
                        'issue_type': analysis['type'],
                        'travel_reliable': analysis['valid'],
                        'trip_count': trip_count,
                        'canonical_description': canonical_description,
                        'pattern_description': pattern_description
                    }
                    
                    # Create stop-level pattern violation using standard formatter
                    if analysis['type'] != 'consecutive':
                        has_gap_before = self._stop_has_gap_before(stop_id, pattern, canonical)
                        
                        if has_gap_before:
                            severity = 'high' if analysis['type'] == 'has_swaps_and_gaps' else 'medium'
                            
                            # USE STANDARD VIOLATION FORMATTER
                            violation = self.create_violation_entry(
                                violation_type='gap_before_stop',
                                severity=severity,
                                description=f'Gap exists before this stop in {trip_type} trips',
                                stop_id=stop_id,
                                stop_name=stop_name,
                                direction_id=direction,
                                trip_type=trip_type,
                                trip_count=trip_count,
                                original_issue_type=analysis['type'],
                                canonical_description=canonical_description,
                                problematic_description=pattern_description
                            )
                            
                            violation_key = f"pattern_{self.route_long_name}_{stop_id}_{direction}_{trip_type}"
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

    def _stop_has_gap_before(self, stop_id, pattern, canonical):
        """Check if a specific stop has a gap before it in the pattern"""
        pattern_list = list(pattern)
        canonical_list = list(canonical)
        
        try:
            pattern_idx = pattern_list.index(stop_id)
            canonical_idx = canonical_list.index(stop_id)
            
            # First stop can't have gap before it
            if pattern_idx == 0 or canonical_idx == 0:
                return False
            
            # Get previous stops
            pattern_prev = pattern_list[pattern_idx - 1]
            canonical_prev = canonical_list[canonical_idx - 1]
            
            # Gap exists if previous stops are different
            return pattern_prev != canonical_prev
            
        except (ValueError, IndexError):
            return False

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
            violation_key = f"pattern_{self.route_long_name}_{row['stop_id']}_{row['direction_id']}_{row['trip_type']}"
            
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
            key = f"{self.route_long_name}_{row['stop_id']}_{row['direction_id']}_{row['trip_type']}"
            
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
                
                violation_key = f"regulatory_{key}"
                self.add_violation_to_log(regulatory_violations_log, violation_key, violation)
        
        self._regulatory_violations_log = regulatory_violations_log
        
        df = df.drop('departure_seconds', axis=1)
        
        print(f"Regulatory analysis complete: {len(self._regulatory_stops_log)} combinations, {len(regulatory_violations_log)} violations")
        return df
   
#   ======================================= Handle Navigational Maps =========================================
    def create_master_indexer(self):
        """Create master indexer with correct stop-level lookups"""
        print("\n=== CREATING MASTER INDEXER ===")
        
        if not hasattr(self, 'df_final') or len(self.df_final) == 0:
            print("No final dataframe found")
            return
        
        master_indexer = {}
        
        # Group by combination level
        combination_groups = self.df_final.groupby([
            'stop_id', 'stop_name', 'direction_id', 'trip_type'
        ])
        
        print(f"Processing {len(combination_groups)} unique combinations")
        
        for (stop_id, stop_name, direction_id, trip_type), group in combination_groups:
            route_id = str(self.route_id)
            stop_id = str(stop_id)
            direction_id = str(direction_id)
            
            # Create indexer key at combination level
            indexer_key = f"{route_id}_{stop_id}_{direction_id}_{trip_type}"
            
            # Get all time_types for this combination
            time_type_data = {}
            total_records = 0
            
            for time_type in group['time_type'].unique():
                time_group = group[group['time_type'] == time_type]
                time_type_data[time_type] = {
                    'record_count': len(time_group),
                    'is_regulatory': bool(time_group['is_regulatory'].iloc[0]) if 'is_regulatory' in time_group.columns else False,
                    'travel_time_valid': bool(time_group['travel_time_valid'].iloc[0]) if 'travel_time_valid' in time_group.columns else True,
                    'topology_flagged': bool(time_group['topology_flagged'].iloc[0]) if 'topology_flagged' in time_group.columns else False,
                    'topology_critical': bool(time_group['topology_critical'].iloc[0]) if 'topology_critical' in time_group.columns else False
                }
                total_records += len(time_group)
            
            # Get analysis information for this combination
            topology_info = self._get_topology_info_for_stop(stop_name)
            pattern_info = self._get_pattern_info_for_combination(stop_id, direction_id, trip_type)  # FIXED: Added stop_id
            regulatory_info = self._get_regulatory_info_for_combination(stop_id, direction_id, trip_type)
            gap_info = self._get_gap_info_for_combination(stop_id, direction_id, trip_type)
            
            # Create master indexer entry
            master_indexer[indexer_key] = {
                # Basic identifiers
                'route_id': route_id,
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
                
                # Gap information
                'gap_info': gap_info,
                
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
                    gap_info.get('severity')  # FIXED: Now method accepts 4 parameters
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

    def _get_gap_info_for_combination(self, stop_id, direction_id, trip_type):
        """Get gap information from pattern violations log"""
        # Create the key for pattern violations log
        violation_key = f"pattern_{self.route_long_name}_{stop_id}_{direction_id}_{trip_type}"
        
        if hasattr(self, '_pattern_violations_log') and violation_key in self._pattern_violations_log:
            violation = self._pattern_violations_log[violation_key]
            return {
                'has_gap': True,
                'severity': violation.get('severity', 'medium'),
                'violation_type': violation.get('violation_type'),
                'description': violation.get('description')
            }
        
        return {
            'has_gap': False,
            'severity': None,
            'violation_type': None,
            'description': None
        }

    def _get_topology_info_for_stop(self, stop_name):
        """Get topology information for a specific stop"""
        for log_key, violation in self._topology_violations_log.items():
            if violation.get('stop_name') == stop_name:
                return {
                    'is_flagged': True,
                    'violation_type': violation.get('violation_type'),
                    'severity': violation.get('severity')
                }
        
        return {
            'is_flagged': False,
            'violation_type': None,
            'severity': None
        }

    def _get_pattern_info_for_combination(self, stop_id, direction_id, trip_type):
        """Get pattern information for specific stop/direction/trip_type combination"""
        # Create the key for stop-level trip types log
        trip_type_key = f"{self.route_long_name}_{stop_id}_{direction_id}_{trip_type}"
        
        if hasattr(self, '_trip_types_log') and trip_type_key in self._trip_types_log:
            trip_info = self._trip_types_log[trip_type_key]
            return {
                'has_issues': trip_info.get('has_issues', False),
                'issue_type': trip_info.get('issue_type'),
                'severity': 'high' if trip_info.get('issue_type') == 'has_swaps_and_gaps' 
                        else 'medium' if trip_info.get('has_issues') else None
            }
        
        return {
            'has_issues': False,
            'issue_type': None,
            'severity': None
        }

    def _get_regulatory_info_for_combination(self, stop_id, direction_id, trip_type):
        """Get regulatory information for specific stop/direction/trip_type combination"""
        key = f"{self.route_long_name}_{stop_id}_{direction_id}_{trip_type}"
        
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

    def create_new_navigation_maps(self):
        """Create navigation maps using clean topology results"""
        print("\n=== CREATING NEW NAVIGATION MAPS ===")
        
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
        
        # Build main navigation structures
        for _, row in unique_combinations.iterrows():
            stop_id = str(row['stop_id'])
            direction_id = str(row['direction_id'])
            time_type = row['time_type']
            trip_type = row['trip_type']
            route_id = str(self.route_id)
            
            # Structure 1: stop_to_combinations
            if stop_id not in stop_to_combinations:
                stop_to_combinations[stop_id] = {}
            if route_id not in stop_to_combinations[stop_id]:
                stop_to_combinations[stop_id][route_id] = {}
            if direction_id not in stop_to_combinations[stop_id][route_id]:
                stop_to_combinations[stop_id][route_id][direction_id] = {}
            if trip_type not in stop_to_combinations[stop_id][route_id][direction_id]:
                stop_to_combinations[stop_id][route_id][direction_id][trip_type] = {"time_types": []}
            
            time_types_list = stop_to_combinations[stop_id][route_id][direction_id][trip_type]["time_types"]
            if time_type not in time_types_list:
                time_types_list.append(time_type)
            
            # Structure 2: route_to_combinations
            if route_id not in route_to_combinations:
                route_to_combinations[route_id] = {}
            if direction_id not in route_to_combinations[route_id]:
                route_to_combinations[route_id][direction_id] = {}
            if trip_type not in route_to_combinations[route_id][direction_id]:
                route_to_combinations[route_id][direction_id][trip_type] = {"time_types": [], "stop_ids": []}
            
            route_time_types = route_to_combinations[route_id][direction_id][trip_type]["time_types"]
            if time_type not in route_time_types:
                route_time_types.append(time_type)
            
            route_stop_ids = route_to_combinations[route_id][direction_id][trip_type]["stop_ids"]
            if stop_id not in route_stop_ids:
                route_stop_ids.append(stop_id)
        
        # Sort time types logically
        time_order = ['am_rush', 'day', 'pm_rush', 'night', 'weekend']
        def sort_structure(structure):
            if isinstance(structure, dict):
                for key, value in structure.items():
                    if key == "time_types" and isinstance(value, list):
                        value.sort(key=lambda x: time_order.index(x) if x in time_order else 999)
                    elif key == "stop_ids" and isinstance(value, list):
                        value.sort()
                    else:
                        sort_structure(value)
        
        sort_structure(stop_to_combinations)
        sort_structure(route_to_combinations)
        
        # Store results
        self._stop_to_combinations = stop_to_combinations
        self._route_to_combinations = route_to_combinations
        
        print(f"Navigation maps created:")
        print(f"  - {len(stop_to_combinations)} stops in combinations")
        print(f"  - {len(route_to_combinations)} routes") 
        print(f"  - {len(stop_name_to_stop_ids)} stop names mapped")
        
        return {
            'stop_to_combinations': stop_to_combinations,
            'route_to_combinations': route_to_combinations,
            'stop_name_to_stop_ids': stop_name_to_stop_ids
        }

    def export_new_navigation_maps(self):
        """Export the new navigation maps to JSON files"""
        print("\n=== EXPORTING NEW NAVIGATION MAPS ===")
        
        if not hasattr(self, '_stop_to_combinations') or not hasattr(self, '_route_to_combinations'):
            print("Navigation maps not found. Creating them first...")
            self.create_new_navigation_maps()
        
        def make_json_serializable(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        safe_route_name = self.route_long_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        output_folder = Path(f'route_analysis_{safe_route_name}_{self.route_id}')
        output_folder.mkdir(exist_ok=True)
        navigation_folder = output_folder / 'navigation_maps'
        navigation_folder.mkdir(exist_ok=True)
        
        print(f"Created output folder: {output_folder}")
        
        # Export the three navigation structures
        stop_to_comb_file = navigation_folder / 'stop_to_combinations.json'
        route_to_comb_file = navigation_folder / 'route_to_combinations.json'
        stop_name_to_ids_file = navigation_folder / 'stop_name_to_stop_ids.json'
        
        with open(stop_to_comb_file, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(self._stop_to_combinations), f, indent=2, ensure_ascii=False)
        
        with open(route_to_comb_file, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(self._route_to_combinations), f, indent=2, ensure_ascii=False)
        
        with open(stop_name_to_ids_file, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(self._stop_name_to_stop_ids), f, indent=2, ensure_ascii=False)
        
        print(f"Navigation maps exported to: {navigation_folder}")
        return output_folder

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
                    print(f"    Creating new summary file for route {list(new_data['route_info'].keys())[0]}")
                    return new_data
                
                # FIXED: Properly merge route_info (keep all routes)
                merged_routes = existing_data.get('route_info', {}).copy()
                existing_route_count = len(merged_routes)
                
                # Get the current route being processed and merge it
                current_route_id = None
                for route_id, route_info in new_data['route_info'].items():
                    current_route_id = route_id
                    was_already_processed = route_id in merged_routes
                    merged_routes[route_id] = route_info
                    break  # Should only be one route in new_data
                
                new_route_count = len(merged_routes)
                
                print(f"    Route summary: {existing_route_count} → {new_route_count} routes")
                if current_route_id:
                    if was_already_processed:
                        print(f"    Route {current_route_id} was reprocessed (replacing existing data)")
                    else:
                        print(f"    Route {current_route_id} added as new route")
                
                # Aggregate data_summary
                existing_summary = existing_data.get('data_summary', {})
                new_summary = new_data['data_summary']
                
                if was_already_processed:
                    print(f"    Warning: Route {current_route_id} reprocessed - summary stats may be inflated")
                    print(f"    Consider cleaning the summary file if this is unexpected")
                
                aggregated_summary = {
                    'total_combinations': existing_summary.get('total_combinations', 0) + new_summary.get('total_combinations', 0),
                    'flagged_combinations': existing_summary.get('flagged_combinations', 0) + new_summary.get('flagged_combinations', 0),
                    'severity_breakdown': {
                        'high': existing_summary.get('severity_breakdown', {}).get('high', 0) + new_summary.get('severity_breakdown', {}).get('high', 0),
                        'medium': existing_summary.get('severity_breakdown', {}).get('medium', 0) + new_summary.get('severity_breakdown', {}).get('medium', 0),
                        'low': existing_summary.get('severity_breakdown', {}).get('low', 0) + new_summary.get('severity_breakdown', {}).get('low', 0),
                        'none': existing_summary.get('severity_breakdown', {}).get('none', 0) + new_summary.get('severity_breakdown', {}).get('none', 0)
                    },
                    'violation_type_counts': {
                        'topology': existing_summary.get('violation_type_counts', {}).get('topology', 0) + new_summary.get('violation_type_counts', {}).get('topology', 0),
                        'pattern': existing_summary.get('violation_type_counts', {}).get('pattern', 0) + new_summary.get('violation_type_counts', {}).get('pattern', 0),
                        'regulatory': existing_summary.get('violation_type_counts', {}).get('regulatory', 0) + new_summary.get('violation_type_counts', {}).get('regulatory', 0)
                    }
                }
                
                return {
                    'route_info': merged_routes,
                    'data_summary': aggregated_summary,
                    'files_exported': new_data['files_exported']  # Keep current export info
                }
            
            else:
                # Default: simple merge
                merged = existing_data.copy()
                merged.update(new_data)
                return merged
        
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
        
        # Use global output folder (not route-specific)
        output_folder = Path('transit_analysis_global')
        output_folder.mkdir(exist_ok=True)
        
        print(f"Using global output folder: {output_folder}")
        
        # Define all files to export
        files_to_export = {
            'master_indexer.json': ('master_indexer', self._master_indexer),
            'topology_violations.json': ('violations', self._topology_violations_log),
            'pattern_violations.json': ('violations', self._pattern_violations_log),
            'regulatory_violations.json': ('violations', self._regulatory_violations_log),
            'trip_types.json': ('trip_types', self._trip_types_log),
            'regulatory_stops.json': ('stops', self._regulatory_stops_log)
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
            'global_stop_name_to_stop_ids.json': ('navigation', self._stop_name_to_stop_ids)
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
            'route_info': {
                str(self.route_id): {  # ← KEY FIX: Use route_id as dictionary key
                    'route_id': str(self.route_id),
                    'route_name': str(self.route_long_name),
                    'route_short_name': str(self.route_short_name)
                }
            },
            'data_summary': {
                'total_combinations': len(self._master_indexer),
                'flagged_combinations': sum(1 for entry in self._master_indexer.values() 
                                        if entry['violation_flags']['has_any_violation']),
                'severity_breakdown': {
                    'high': sum(1 for entry in self._master_indexer.values() if entry.get('overall_severity') == 'high'),
                    'medium': sum(1 for entry in self._master_indexer.values() if entry.get('overall_severity') == 'medium'),
                    'low': sum(1 for entry in self._master_indexer.values() if entry.get('overall_severity') == 'low'),
                    'none': sum(1 for entry in self._master_indexer.values() if entry.get('overall_severity') == 'none')
                },
                'violation_type_counts': {
                    'topology': len(self._topology_violations_log),
                    'pattern': len(self._pattern_violations_log),
                    'regulatory': len(self._regulatory_violations_log)
                }
            },
            'files_exported': exported_files
        }
        
        summary_file = output_folder / 'global_summary.json'
        existing_summary = load_existing_json(summary_file)
        merged_summary = merge_data(existing_summary, route_summary, 'summary')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(merged_summary), f, indent=2, ensure_ascii=False)
        
        print(f"✅ Global summary updated: {summary_file}")
        
        total_routes_in_summary = len(merged_summary.get('route_info', {}))
        total_global_combinations = merged_summary.get('data_summary', {}).get('total_combinations', 0)
        
        print(f"\n🌍 GLOBAL ANALYSIS STATUS:")
        print(f"  - Total routes processed: {total_routes_in_summary}")
        print(f"  - Total combinations: {total_global_combinations}")
        print(f"  - Global files: {output_folder}")
        print(f"  - All data available in global files")
        
        return {
            'global_output_folder': str(output_folder),
            'global_summary_file': str(summary_file),
            'total_routes': total_routes_in_summary,
            'total_combinations': total_global_combinations
        }




































#OLD OLD OLD










#   ========================== Basic Stop Analysis ===============================

    def find_dir0_stop_sequence_for_order(self):

        df = self.df_final.copy()

        # First, get the Direction 0 sequence mapping for sorting
        dir0_data = df[df['direction_id'] == 0][['stop_name', 'stop_sequence']]
    
        stop_sequence_map = {}
        for stop_name, group in dir0_data.groupby('stop_name'):
            sequence_counts = group['stop_sequence'].value_counts()
            most_common_sequence = sequence_counts.index[0]  # Most frequent sequence
            stop_sequence_map[stop_name] = most_common_sequence
            # Log if there were multiple sequences for this stop
            if len(sequence_counts) > 1:
                print(f"  Warning: {stop_name} has multiple sequences in Dir 0: {dict(sequence_counts)}, using {most_common_sequence}")
        
        return stop_sequence_map
    
    def create_stop_analysis(self):
        """
        Categorize stops as directional or shared, and flag problematic ones
        """
        print(f"\n=== CATEGORIZING STOPS ===")
        stop_analysis = {}  # FIX: Use dictionary, not list
        df = self.df_final.copy()

        dir_zero_stop_seq = self.find_dir0_stop_sequence_for_order()

        for stop_name, group in df.groupby('stop_name'):
            stop_ids = group['stop_id'].unique()
            
            print(f"\nAnalyzing: {stop_name}")
            print(f"  Mapped to stop_ids: {list(stop_ids)}")
            
            # Check how each stop_id behaves
            shared_stop_ids = []
            directional_stop_ids = []
            
            for stop_id in stop_ids:
                stop_id_data = group[group['stop_id'] == stop_id]
                directions_for_this_stop_id = stop_id_data['direction_id'].unique()
                
                if len(directions_for_this_stop_id) > 1:
                    shared_stop_ids.append(stop_id)
                    behavior = "shared"
                else:
                    directional_stop_ids.append(stop_id)
                    behavior = "directional"
                    
                print(f"    {stop_id}: {behavior} (directions: {list(directions_for_this_stop_id)})")
            
            # Check which directions this stop serves
            directions_served = group['direction_id'].unique()

            # Determine stop type - more detailed classification
            if len(shared_stop_ids) > 0:
                stop_type = "Shared"
            else:
                # All stop_ids are directional, but check how many directions
                if len(directions_served) == 1:
                    stop_type = "Unidirectional"
                else:
                    stop_type = "Bidirectional"
            
            # Check what logs this stop appears in
            route_name = self.route_long_name
            composite_key = f"{route_name}_{stop_name}"

            in_sequence_log = composite_key in self._sequence_corrections_log
            in_direction_log = composite_key in self._direction_pair_corrections_log
            in_root_cause_log = composite_key in self._sequence_root_causes_log

            # SIMPLIFIED CLASSIFICATION LOGIC
            # Check if issues were resolved (using 'resolved' key from consistent log structure)
            sequence_resolved = True  # Default to resolved if not in log
            direction_resolved = True  # Default to resolved if not in log
            root_cause_resolved = True  # Root causes are detected, not resolved, so always False if present

            if in_sequence_log:
                sequence_resolved = self._sequence_corrections_log[composite_key].get('resolved', False)

            if in_direction_log:
                direction_resolved = self._direction_pair_corrections_log[composite_key].get('resolved', False)

            if in_root_cause_log:
                root_cause_resolved = self._sequence_root_causes_log[composite_key].get('resolved', False)  # Always False

            # Determine if stop appears in any log
            appears_in_any_log = in_sequence_log or in_direction_log or in_root_cause_log
            
            # Determine if all issues were resolved
            all_issues_resolved = sequence_resolved and direction_resolved and root_cause_resolved

            # SIMPLIFIED SEVERITY AND STATUS
            if not appears_in_any_log:
                # Not apparent in any log = severity 0
                severity = 0
                problematic_status = "Normal"
                problematic_description = "No data quality issues detected."
                
            elif appears_in_any_log and all_issues_resolved:
                # Apparent in log AND resolved = severity 1
                severity = 1
                problematic_status = "Minor"
                fixed_issues = []
                if in_sequence_log and sequence_resolved:
                    fixed_issues.append("sequence standardized")
                if in_direction_log and direction_resolved:
                    fixed_issues.append("direction reassigned")
                problematic_description = f"Issues detected and successfully resolved: {', '.join(fixed_issues)}."
                
            else:
                # Apparent in log AND NOT resolved = severity 2
                severity = 2
                problematic_status = "Severe"
                unresolved_issues = []
                if in_sequence_log and not sequence_resolved:
                    unresolved_issues.append("multiple stop sequences")
                if in_direction_log and not direction_resolved:
                    unresolved_issues.append("direction inconsistencies")
                if in_root_cause_log:
                    unresolved_issues.append("sequence root cause")
                problematic_description = f"Unresolved issues: {', '.join(unresolved_issues)}."

            print(f"  Final Classification: {stop_type} - {problematic_status} (Severity: {severity})")

            # Determine problematic type based on which logs the stop appears in
            problematic_type = "None"  # FIX: Use string instead of False

            if (in_sequence_log or in_root_cause_log) and in_direction_log:
                problematic_type = "Complex"
            elif in_sequence_log or in_root_cause_log:
                problematic_type = "Sequential"
            elif in_direction_log:
                problematic_type = "Directional"

            if severity > 0:
                print(f"    Issue: {problematic_description}")

            # Generate stop type description  
            if stop_type == "Shared":
                stop_type_description = f"uses the same stop_id for both directions (has {len(shared_stop_ids)} shared stop_id{'s' if len(shared_stop_ids) != 1 else ''})"
            elif stop_type == "Bidirectional":
                stop_type_description = f"serves both directions using separate stop_ids ({len(directional_stop_ids)} directional stop_ids total)"
            elif stop_type == "Unidirectional":
                stop_type_description = f"only serves one direction ({len(directional_stop_ids)} directional stop_id{'s' if len(directional_stop_ids) != 1 else ''})"

            # Generate severity description
            if severity == 0:
                severity_description = "no data quality issues were detected"
            elif severity == 1:
                severity_description = "data quality issues were detected but successfully resolved"
            elif severity == 2:
                severity_description = "data quality issues were detected but remain unresolved"

            # Create full description
            full_description = f"On route {route_name}, the stop {stop_name} is a {stop_type.lower()} stop, since it {stop_type_description}. It is marked as {problematic_status.lower()} since {severity_description}."

            # Store with only the entries you want
            stop_analysis[composite_key] = {
                'stop_name': stop_name,
                'route_long_name': route_name,
                'description': full_description,
                'stop_type': stop_type,
                'problematic_status': problematic_status,  # FIX: Use status not severity
                'problematic_type': problematic_type,
                'problematic_description': problematic_description,
                'shared_ids': shared_stop_ids,
                'directional_ids': directional_stop_ids,
                'print_order': dir_zero_stop_seq.get(stop_name, 999)
            }

        # Sort the dictionary by Direction 0 stop sequence
        sorted_stop_analysis = dict(sorted(
            stop_analysis.items(), 
            key=lambda item: item[1]['print_order']
        ))

        # Remove the temporary stop_sequence field from the final dictionary
        for entry in sorted_stop_analysis.values():
            entry.pop('print_order', None)

        print(f"\nStop analysis complete: {len(sorted_stop_analysis)} stops analyzed and sorted by Direction 0 sequence")
        
        return sorted_stop_analysis

    def add_analysis_availability_to_stop_analysis(self):
        """Add availability flags for histograms, travel times, punctuality, and tables to stop_analysis_dict"""
        
        # Update stop_analysis_dict with analysis availability flags
        for composite_key in self.stop_analysis_dict.keys():
            # Extract route name from composite key for table checking
            route_name = composite_key.split('_')[0]  # Assumes format: route_stopname
            
            # Check histogram availability
            has_histograms = hasattr(self, '_delay_histograms') and composite_key in self._delay_histograms
            
            # Check travel times availability
            has_travel_times = hasattr(self, '_travel_times_data') and composite_key in self._travel_times_data
            
            # Check punctuality availability
            has_punctuality = hasattr(self, '_punctuality_data') and composite_key in self._punctuality_data
            
            # Check tables availability (route-level, so check if route has before/after tables)
            has_tables = (hasattr(self, '_direction_tables_before') and 
                        hasattr(self, '_direction_tables_after'))
            
            # Add all flags to stop analysis
            self.stop_analysis_dict[composite_key].update({
                'has_histograms': has_histograms,
                'has_travel_times': has_travel_times,
                'has_punctuality_data': has_punctuality,
                'has_tables': has_tables
            })

#   ====================== Additional Route-wise Analysis =========================
    def create_direction_tables(self, df):
        """
        Create separate tables for each direction, with extra column showing other direction
        """
        # Get direction distribution  
        direction_dist = df.groupby(['stop_name', 'stop_id', 'direction_id', 'stop_sequence']).size().reset_index(name='count')
        
        # Create pivot to get both directions for each stop
        pivot = direction_dist.pivot_table(
            index=['stop_name', 'stop_id', 'stop_sequence'],
            columns='direction_id', 
            values='count',
            fill_value=0
        )
        pivot.columns = ['Dir_0', 'Dir_1']
        
        # Direction 0 table: stops with dir 0 records, sorted by stop_sequence
        dir0_data = pivot[pivot['Dir_0'] > 0].copy().sort_values('stop_sequence')
        
        # Direction 1 table: stops with dir 1 records, sorted by stop_sequence  
        dir1_data = pivot[pivot['Dir_1'] > 0].copy().sort_values('stop_sequence')
        
        dir_table = {
            'direction_0': dir0_data[['Dir_0', 'Dir_1']].reset_index().to_dict('records'),
            'direction_1': dir1_data[['Dir_0', 'Dir_1']].reset_index().to_dict('records')
        }
        return dir_table

#   ====================== Additional Route-Stop-wise Analysis =========================
    def generate_delay_histograms(self, bins=20):
        """
        Generate normalized delay histograms organized by stop with directional-time sub-keys
        """
        print("\n=== GENERATING DELAY HISTOGRAMS (STOP-ORGANIZED) ===")
        
        route_name = self.route_long_name
        delay_histograms = {}
        
        dir_zero_stop_seq = self.find_dir0_stop_sequence_for_order()

        # Generate histograms for each stop, then by direction-time combinations
        for stop_name, stop_group in self.df_final.groupby('stop_name'):
            composite_key = f"{route_name}_{stop_name}"
            
            print(f"Generating histograms for stop: {stop_name}")
            
            # Initialize stop data structure
            stop_histogram_data = {
                'route_name': route_name,
                'stop_name': stop_name,
                'histograms': {},
                'metadata': {
                    'bins_used': bins,
                    'histograms_count': 0
                },
                'print_order': dir_zero_stop_seq.get(stop_name, 999)
            }
            
            # Generate histograms for each direction-time combination within this stop
            for (direction_id, time_type), group in stop_group.groupby(['direction_id', 'time_type']):
                sub_key = f"{direction_id}_{time_type}"
                
                print(f"  Processing: Dir {direction_id}, {time_type}")
                
                if len(group) >= 10:  # Minimum sample size
                    
                    # Filter out invalid data and get clean delays
                    total_delays = group['departure_delay'].dropna()
                    
                    # For incremental delays, only use non-NaN values (automatically excludes sequence gaps)
                    incremental_delays = group['incremental_delay'].dropna()
                    # Additional safety: remove any remaining NaT values if column is object type
                    if group['incremental_delay'].dtype == 'object':
                        incremental_delays = incremental_delays[incremental_delays != pd.NaT]
                    
                    # Generate histograms if we have enough data
                    direction_time_histograms = {}
                    
                    # Total delay histogram
                    if len(total_delays) >= 5:
                        direction_time_histograms['total_delay'] = self._create_normalized_histogram(
                            total_delays, bins, f'Total Delay - {stop_name} (Dir {direction_id}, {time_type.replace("_", " ").title()})'
                        )
                    
                    # Incremental delay histogram  
                    if len(incremental_delays) >= 5:
                        direction_time_histograms['incremental_delay'] = self._create_normalized_histogram(
                            incremental_delays, bins, f'Incremental Delay - {stop_name} (Dir {direction_id}, {time_type.replace("_", " ").title()})'
                        )
                    
                    # Store if we have any histograms for this direction-time combination
                    if direction_time_histograms:
                        stop_histogram_data['histograms'][sub_key] = {
                            'direction_id': direction_id,
                            'time_type': time_type,
                            'histograms': direction_time_histograms,
                            'metadata': {
                                'total_delay_sample_size': len(total_delays),
                                'incremental_delay_sample_size': len(incremental_delays)
                            }
                        }
                        stop_histogram_data['metadata']['histograms_count'] += len(direction_time_histograms)
                        print(f"    ✓ Generated {len(direction_time_histograms)} histogram types")
                    else:
                        print(f"    ✗ Insufficient data for histograms")
                else:
                    print(f"    ✗ Insufficient sample size ({len(group)} < 10)")
            
            # Only store stop if it has any histograms
            if stop_histogram_data['histograms']:
                delay_histograms[composite_key] = stop_histogram_data
                histograms_count = stop_histogram_data['metadata']['histograms_count']
                print(f"  ✓ Stop complete: {histograms_count} direction-time combinations")
            else:
                print(f"  ✗ No histograms generated for this stop")
        
        print(f"\nGenerated histograms for {len(delay_histograms)} stops")
        # Sort the dictionary by Direction 0 stop sequence
        sorted_histograms = dict(sorted(
            delay_histograms.items(), 
            key=lambda item: item[1]['print_order']
        ))

        # Remove the temporary stop_sequence field from the final dictionary
        for entry in sorted_histograms.values():
            entry.pop('print_order', None)

        # Print summary breakdown
        total_combinations = 0
        total_histograms = 0
        direction_breakdown = {0: 0, 1: 0}
        time_breakdown = {}
        
        for stop_data in sorted_histograms.values():
            for sub_key, combo_data in stop_data['histograms'].items():
                total_combinations += 1  # Count each direction-time combination
                total_histograms += len(combo_data['histograms'])
                
                direction_id = combo_data['direction_id']
                time_type = combo_data['time_type']
                
                direction_breakdown[direction_id] += 1
                time_breakdown[time_type] = time_breakdown.get(time_type, 0) + 1
        
        print(f"Total direction-time combinations: {total_combinations}")
        print(f"Total individual histograms: {total_histograms}")
        print(f"Direction breakdown: Dir 0: {direction_breakdown[0]}, Dir 1: {direction_breakdown[1]}")
        print(f"Time type breakdown: {time_breakdown}")
        
        return sorted_histograms

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

    def generate_travel_times_data(self):
        """
        Generate travel times statistics organized by stop with directional-time sub-keys
        """
        print("\n=== GENERATING TRAVEL TIMES DATA (STOP-ORGANIZED) ===")
        
        route_name = self.route_long_name
        travel_times_data = {}
        df_final = self.df_final.copy()
        dir_zero_stop_seq = self.find_dir0_stop_sequence_for_order()

        # Simple filtering - just need previous_stop and valid travel times
        valid_data = df_final[
            df_final['previous_stop'].notna() &              # Must have a previous stop
            df_final['travel_time_valid'] == True            # Must be valid (no sequence gaps)
        ].copy()
        
        print(f"Valid travel time records: {len(valid_data)}/{len(df_final)} rows")
        
        # Convert to seconds
        valid_data['observed_travel_time_seconds'] = valid_data['observed_travel_time'].dt.total_seconds()
        valid_data['scheduled_travel_time_seconds'] = valid_data['scheduled_travel_time'].dt.total_seconds()
        
        # Group by to_stop (arrival stop), then by direction and time category
        for to_stop, stop_group in valid_data.groupby('stop_name'):
            composite_key = f"{route_name}_{to_stop}"
            
            print(f"Generating travel times for stop: {to_stop}")
            
            # Initialize stop data structure
            stop_travel_data = {
                'route_name': route_name,
                'stop_name': to_stop,
                'travel_times': {},
                'metadata': {
                    'total_segments': 0
                },
                'print_order': dir_zero_stop_seq.get(to_stop, 999)
            }
            
            # Generate travel times for each direction-time combination within this stop
            for (direction_id, time_type), group in stop_group.groupby(['direction_id', 'time_type']):
                sub_key = f"{direction_id}_{time_type}"
                
                print(f"  Processing: Dir {direction_id}, {time_type}")
                
                if len(group) >= 3:  # Minimum sample size
                    
                    # Get the from_stop for this direction-time combination
                    # (should be consistent within each direction)
                    from_stops = group['previous_stop'].unique()
                    if len(from_stops) == 1:
                        from_stop = from_stops[0]
                    else:
                        # Multiple from_stops - this could happen at transfer points
                        # Take the most common one
                        from_stop = group['previous_stop'].mode().iloc[0]
                        print(f"    Warning: Multiple from_stops detected, using most common: {from_stop}")
                    
                    observed_times = group['observed_travel_time_seconds']
                    scheduled_times = group['scheduled_travel_time_seconds'] 
                    incremental_delays = group['incremental_delay']
                    
                    direction_time_stats = {
                        'direction_id': direction_id,
                        'time_type': time_type,
                        'from_stop': from_stop,
                        'to_stop': to_stop,
                        'stop_sequence': group['stop_sequence'].iloc[0],
                        'segment_name': f"{from_stop} → {to_stop}",
                        'statistics': {
                            'observed_travel_time': self._calculate_time_stats(observed_times, 'observed'),
                            'scheduled_travel_time': self._calculate_time_stats(scheduled_times, 'scheduled'),
                            'incremental_delay': self._calculate_time_stats(incremental_delays, 'incremental_delay'),
                            'sample_size': len(group),
                            'comparison': {
                                'mean_difference_seconds': float(observed_times.mean() - scheduled_times.mean()),
                                'mean_incremental_delay': float(incremental_delays.mean())
                            }
                        }
                    }
                    # Store the direction-time combination
                    stop_travel_data['travel_times'][sub_key] = direction_time_stats
                    stop_travel_data['metadata']['total_segments'] += 1
                    print(f"    ✓ Generated travel time stats for {from_stop} → {to_stop}")
                else:
                    print(f"    ✗ Insufficient sample size ({len(group)} < 3)")
            
            # Only store stop if it has any travel time data
            if stop_travel_data['travel_times']:
                travel_times_data[composite_key] = stop_travel_data
                total_segments = stop_travel_data['metadata']['total_segments']
                print(f"  ✓ Stop complete: {total_segments} direction-time combinations")
            else:
                print(f"  ✗ No travel time data generated for this stop")
        
        print(f"\nGenerated travel time data for {len(travel_times_data)} stops")

        # Sort the dictionary by Direction 0 stop sequence
        sorted_travel_times = dict(sorted(
            travel_times_data.items(), 
            key=lambda item: item[1]['print_order']
        ))

        # Remove the temporary stop_sequence field from the final dictionary
        for entry in sorted_travel_times.values():
            entry.pop('print_order', None)

        # Print summary breakdown
        total_segments = 0
        direction_breakdown = {0: 0, 1: 0}
        time_breakdown = {}
        
        for stop_data in sorted_travel_times.values():
            for sub_key, segment_data in stop_data['travel_times'].items():
                total_segments += 1
                
                direction_id = segment_data['direction_id']
                time_type = segment_data['time_type']
                
                direction_breakdown[direction_id] += 1
                time_breakdown[time_type] = time_breakdown.get(time_type, 0) + 1
        
        print(f"Total direction-time segments: {total_segments}")
        print(f"Direction breakdown: Dir 0: {direction_breakdown[0]}, Dir 1: {direction_breakdown[1]}")
        print(f"Time type breakdown: {time_breakdown}")
        
        return sorted_travel_times

    def _calculate_time_stats(self, times, data_type=None):
        """Calculate statistics for a set of times with proper validation"""
        
        # First check: empty input
        if len(times) == 0:
            return None
        
        # DATA VALIDATION: Remove NaN/NaT values
        clean_times = times.dropna()
        
        # Second check: no valid data after cleaning
        if len(clean_times) == 0:
            return {
                'data_type': data_type,
                'sample_size': 0,
                'error': 'All values were NaN/NaT'
            }
        
        # SAFE STATISTICAL CALCULATIONS with validation
        try:
            # Basic stats (always safe with clean data)
            mean_val = float(clean_times.mean())
            median_val = float(clean_times.median())
            min_val = float(clean_times.min())
            max_val = float(clean_times.max())
            
            # Standard deviation (safe only with 2+ values)
            std_val = float(clean_times.std()) if len(clean_times) > 1 else 0.0
            
            # Percentiles (need sufficient data points)
            if len(clean_times) >= 4:
                p25 = float(np.percentile(clean_times, 25))
                p75 = float(np.percentile(clean_times, 75))
            else:
                p25 = min_val  # Use min/max for small samples
                p75 = max_val
            
            if len(clean_times) >= 20:
                p5 = float(np.percentile(clean_times, 5))
                p95 = float(np.percentile(clean_times, 95))
            else:
                p5 = min_val   # Use min/max for small samples
                p95 = max_val
            
            return {
                'data_type': data_type,
                'mean_time_seconds': mean_val,
                'std_time_seconds': std_val,
                'percentile_25': p25,
                'percentile_75': p75,
                'percentile_5': p5,
                'percentile_95': p95,
                'median_time_seconds': median_val,
                'min_time_seconds': min_val,
                'max_time_seconds': max_val,
                'sample_size': len(clean_times),
                'original_sample_size': len(times),  # Track how many were removed
                'nan_count': len(times) - len(clean_times)
            }
            
        except Exception as e:
            # ERROR HANDLING: Catch any unexpected issues
            print(f"Warning: Error calculating statistics for {data_type}: {e}")
            return {
                'data_type': data_type,
                'sample_size': len(clean_times),
                'original_sample_size': len(times),
                'error': str(e)
            }
#   ========================== Converting to JSON ================================

    def _convert_table_to_json(self, table_data):
        """Convert pandas DataFrame or dict to simple JSON format"""
        if table_data is None:
            return {"columns": [], "data": []}
        
        # If it's already a dict, return as-is (assuming it's already in correct format)
        if isinstance(table_data, dict):
            return table_data
        
        # If it's a DataFrame, convert it
        if hasattr(table_data, 'empty'):  # pandas DataFrame
            if table_data.empty:
                return {"columns": [], "data": []}
            return {
                "columns": table_data.columns.tolist(),
                "data": table_data.values.tolist()
            }
        
        # If it's something else, try to handle it
        return {"columns": [], "data": []}

    def _make_json_serializable(self, obj):
        """Convert pandas/numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):  # Handle NaN/NaT values
            return None
        elif isinstance(obj, np.ndarray):  # Handle numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # pandas/numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # pandas/numpy arrays
            return obj.tolist()
        else:
            return obj
        
#   ========================== Exporting to JSON =================================    

    def load_existing_json(self, file_path):
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    # Renamed and simplified basic export:
    def export_basic_json_files(self, output_dir="./analysis_output"):
        """
        Export basic analysis to 4 core JSON files (no detailed data)
        """
        print(f"\n=== EXPORTING BASIC JSON FILES ===")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Define file paths for basic files only
        file_paths = {
            "route_stops": os.path.join(output_dir, "route_stops.json"),
            "stop_routes": os.path.join(output_dir, "stop_routes.json"),
            "stop_analysis": os.path.join(output_dir, "stop_analysis.json"),
            "logs_details": os.path.join(output_dir, "logs_details.json")
        }
        
        # Load existing data
        route_stops_data = self.load_existing_json(file_paths["route_stops"])
        stop_routes_data = self.load_existing_json(file_paths["stop_routes"])
        stop_analysis_data = self.load_existing_json(file_paths["stop_analysis"])
        logs_details_data = self.load_existing_json(file_paths["logs_details"])
        route_name = self.route_long_name
            
        print(f"Processing route: {route_name}")

        # 1. UPDATE ROUTE-TO-STOPS MAPPING
        if hasattr(self, 'stop_analysis_dict') and self.stop_analysis_dict:
            stops_on_route = list(set(data['stop_name'] for data in self.stop_analysis_dict.values()))
            
            # Order stops by direction 0 sequence using centralized function
            dir_zero_stop_seq = self.find_dir0_stop_sequence_for_order()
            stops_on_route.sort(key=lambda stop: dir_zero_stop_seq.get(stop, 999))
                
            # Calculate summary statistics
            summary_stats = {"normal_stops_count": 0, "minor_stops_count": 0, "severe_stops_count": 0}
            for data in self.stop_analysis_dict.values():
                status = data.get('problematic_status', 'Normal').lower()
                if status == 'normal':
                    summary_stats["normal_stops_count"] += 1
                elif status == 'minor':
                    summary_stats["minor_stops_count"] += 1
                elif status == 'severe':
                    summary_stats["severe_stops_count"] += 1
            
            route_stops_data[route_name] = {
                "route_name": route_name,
                "stops": stops_on_route,
                "total_stops": len(stops_on_route),
                "summary": summary_stats,
                "analysis_availability": {
                    "before_and_after_tables": hasattr(self, '_direction_tables_before') and hasattr(self, '_direction_tables_after'),
                    "delay_histograms": hasattr(self, '_delay_histograms') and bool(self._delay_histograms),
                    "travel_times": hasattr(self, '_travel_times_data') and bool(self._travel_times_data),
                    "punctuality": hasattr(self, '_punctuality_data') and bool(self._punctuality_data)
                }
            }
            print(f"  Updated route_stops for {route_name} ({len(stops_on_route)} stops)")
        
        # 2. UPDATE STOP-TO-ROUTES MAPPING
        if hasattr(self, 'stop_analysis_dict') and self.stop_analysis_dict:
            for data in self.stop_analysis_dict.values():
                stop_name = data['stop_name']
                
                # Initialize stop entry if it doesn't exist
                if stop_name not in stop_routes_data:
                    stop_routes_data[stop_name] = {
                        "stop_name": stop_name,
                        "routes": [],
                        "total_routes": 0,
                        "summary": {"normal_routes_count": 0, "minor_routes_count": 0, "severe_routes_count": 0}
                    }
                
                # Add route if not already present
                if route_name not in stop_routes_data[stop_name]["routes"]:
                    stop_routes_data[stop_name]["routes"].append(route_name)
                    stop_routes_data[stop_name]["total_routes"] = len(stop_routes_data[stop_name]["routes"])
                    
                    # Update summary based on this route's status for this stop
                    status = data.get('problematic_status', 'Normal').lower()
                    if status == 'normal':
                        stop_routes_data[stop_name]["summary"]["normal_routes_count"] += 1
                    elif status == 'minor':
                        stop_routes_data[stop_name]["summary"]["minor_routes_count"] += 1
                    elif status == 'severe':
                        stop_routes_data[stop_name]["summary"]["severe_routes_count"] += 1
            
            print(f"  Updated stop_routes for {len(set(data['stop_name'] for data in self.stop_analysis_dict.values()))} stops")
        
        # 3. UPDATE STOP ANALYSIS
        if hasattr(self, 'stop_analysis_dict') and self.stop_analysis_dict:
            for composite_key, analysis in self.stop_analysis_dict.items():
                # Check if this stop has detailed logs
                has_details = (
                    composite_key in getattr(self, '_sequence_corrections_log', {}) or
                    composite_key in getattr(self, '_direction_pair_corrections_log', {}) or
                    composite_key in getattr(self, '_sequence_root_causes_log', {})
                )
                
                stop_analysis_data[composite_key] = {
                    **analysis,
                    "has_details": has_details
                }
            
            print(f"  Updated stop_analysis for {len(self.stop_analysis_dict)} route-stop combinations")
        
        # 4. UPDATE DETAILED LOGS (only for stops with issues)
        logs_added = 0
        if hasattr(self, 'stop_analysis_dict') and self.stop_analysis_dict:
            for composite_key in self.stop_analysis_dict.keys():
                # Check if any logs exist for this composite key
                has_sequence_log = composite_key in getattr(self, '_sequence_corrections_log', {})
                has_direction_log = composite_key in getattr(self, '_direction_pair_corrections_log', {})
                has_root_cause_log = composite_key in getattr(self, '_sequence_root_causes_log', {})
                
                if has_sequence_log or has_direction_log or has_root_cause_log:
                    logs_details_data[composite_key] = {
                        "sequence_corrections": getattr(self, '_sequence_corrections_log', {}).get(composite_key),
                        "direction_corrections": getattr(self, '_direction_pair_corrections_log', {}).get(composite_key),
                        "root_causes": getattr(self, '_sequence_root_causes_log', {}).get(composite_key)
                    }
                    logs_added += 1
            
            print(f"  Updated logs_details for {logs_added} problematic route-stop combinations")

        # SAVE BASIC FILES ONLY
        def save_json_file(data, file_path, description):
            serializable_data = self._make_json_serializable(data)
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, indent=2, ensure_ascii=False)
                print(f"  ✓ Saved {description}: {len(data)} entries")
            except Exception as e:
                print(f"  ✗ Error saving {description}: {e}")
        
        save_json_file(route_stops_data, file_paths["route_stops"], "route_stops.json")
        save_json_file(stop_routes_data, file_paths["stop_routes"], "stop_routes.json") 
        save_json_file(stop_analysis_data, file_paths["stop_analysis"], "stop_analysis.json")
        save_json_file(logs_details_data, file_paths["logs_details"], "logs_details.json")
        
        print(f"\nBasic export complete! Files saved to: {output_dir}")
        
        return {
            "route_stops_count": len(route_stops_data),
            "stop_routes_count": len(stop_routes_data),
            "stop_analysis_count": len(stop_analysis_data),
            "logs_details_count": len(logs_details_data),
            "output_directory": output_dir
        }

    # New separate method for tables:
    def export_tables_to_json(self, output_dir="./analysis_output"):
        """Export before/after direction tables to separate JSON file"""
        
        if not (hasattr(self, '_direction_tables_before') and hasattr(self, '_direction_tables_after')):
            print("No direction tables to export")
            return
        
        # Load existing tables data
        tables_path = os.path.join(output_dir, "route_tables.json")
        
        def load_existing_json(file_path):
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    return {}
            return {}
        
        existing_tables = load_existing_json(tables_path)
        route_name = self.route_long_name
        
        # Add current route's tables
        existing_tables[route_name] = {
            "route_name": route_name,
            "before_corrections": self._convert_table_to_json(self._direction_tables_before),
            "after_corrections": self._convert_table_to_json(self._direction_tables_after),
            "metadata": {
                "last_updated": pd.Timestamp.now().isoformat()
            }
        }
        
        # Save updated tables
        try:
            serializable_data = self._make_json_serializable(existing_tables)
            with open(tables_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved route_tables.json: {route_name} before/after tables")
        except Exception as e:
            print(f"  ✗ Error saving route_tables.json: {e}")

    def export_histograms_to_json(self, output_dir="./analysis_output"):
        """Export histograms to separate JSON file"""
        
        if not hasattr(self, '_delay_histograms') or not self._delay_histograms:
            print("No histograms to export")
            return
        
        histograms_path = os.path.join(output_dir, "delay_histograms.json")
        
        existing_histograms = self.load_existing_json(histograms_path)
        
        # Add current route's histograms
        existing_histograms.update(self._delay_histograms)
        
        # Save updated histograms
        try:
            serializable_data = self._make_json_serializable(existing_histograms)
            with open(histograms_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            # Count total histograms for summary (FIXED)
            total_stops = len(existing_histograms)
            total_combinations = 0
            total_histograms = 0
            
            for stop_data in existing_histograms.values():
                for sub_key, combo_data in stop_data['histograms'].items():  # ← Fixed: use 'histograms'
                    total_combinations += 1
                    total_histograms += len(combo_data['histograms'])
            
            print(f"  ✓ Saved delay_histograms.json:")
            print(f"    - {total_stops} stops")
            print(f"    - {total_combinations} direction-time combinations")  
            print(f"    - {total_histograms} individual histograms")
            
        except Exception as e:
            print(f"  ✗ Error saving delay_histograms.json: {e}")

    def export_travel_times_to_json(self, output_dir="./analysis_output"):
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