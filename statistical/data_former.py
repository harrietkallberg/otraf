import pandas as pd
import numpy as np
import json as json 
import os
from pathlib import Path

class DataFormer:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        
        # FIX: Proper way to check if column exists and has non-null values
        if 'route_long_name' in raw_data.columns and not raw_data['route_long_name'].empty:
            # Get first non-null value
            first_valid = raw_data['route_long_name'].dropna()
            if not first_valid.empty:
                self.route_long_name = first_valid.iloc[0]
            else:
                self.route_long_name = 'Unknown'
        else:
            self.route_long_name = 'Unknown'

        self._sequence_corrections_log = {}
        self._direction_pair_corrections_log = {}
        self._sequence_root_causes_log = {}

        self.stop_analysis = pd.DataFrame()
        self._delay_histograms = {}
        self.direction_tables_before = None
        self.direction_tables_after = None
        
        self.form_data = self._process_data()

#   ================================ Wrapper =====================================
    def _process_data(self):
        df = self.raw_data.copy()
        
        df = self.prepare_columns(df)

        # Create and store the BEFORE correction tables
        dir0_table, dir1_table = self.create_direction_tables(df)
        
        # Store as JSON-ready format for export
        self.direction_tables_before = {
            'direction_0': dir0_table.reset_index().to_dict('records'),
            'direction_1': dir1_table.reset_index().to_dict('records')
        }

        print('before')
        print("=== DIRECTION 0 (ordered by stop_sequence) ===")
        print(dir0_table.to_string())

        print("\n=== DIRECTION 1 (ordered by stop_sequence) ===") 
        print(dir1_table.to_string())

        df_seq, fixed_stop_sequence_stops = self.deduplicate_stop_sequences(df.copy()) # fix wrongly tracked stop_sequence for trips consecutive to missing records stops
        df_fixed, fixed_directions_stops = self.fix_stop_pairs(df_seq.copy()) # fix wrongly tracked stop_ids for trips on directional stops
        df_final,found_sequence_root_stops = self.detect_sequence_root_causes(df_fixed.copy())

        # Process trips to calculate incremental delays and travel times
        processed_trips = []
        for key, trip in df_final.groupby(['trip_id', 'direction_id', 'start_date']):
            trip = trip.sort_values('stop_sequence')
            # Calculate sequence differences to detect gaps
            trip['prev_sequence'] = trip['stop_sequence'].shift(1)
            trip['sequence_diff'] = trip['stop_sequence'] - trip['prev_sequence'] 
            trip['has_sequence_gap'] = trip['sequence_diff'] > 1

            # Calculate incremental delays
            trip['prev_delay'] = trip['departure_delay'].shift(1)
            trip['incremental_delay'] = trip['departure_delay'] - trip['prev_delay']
            trip['incremental_delay'] = trip['incremental_delay'].fillna(0)

            # Mark incremental delays as invalid where there's a sequence gap
            trip.loc[trip['has_sequence_gap'], 'incremental_delay'] = np.nan
            
            # Calculate scheduled travel times
            trip['prev_sched'] = trip['scheduled_departure_time'].shift(1)
            trip['scheduled_travel_time'] = trip['scheduled_departure_time'] - trip['prev_sched']
            trip['scheduled_travel_time'] = trip['scheduled_travel_time'].fillna(pd.Timedelta(0))
            
            # Mark travel times as None where there's a sequence gap
            trip.loc[trip['has_sequence_gap'], 'scheduled_travel_time'] = pd.NaT
            
            # Calculate observed travel times
            trip['prev_observed'] = trip['observed_departure_time'].shift(1)
            trip['observed_travel_time'] = trip['observed_departure_time'] - trip['prev_observed']
            trip['observed_travel_time'] = trip['observed_travel_time'].fillna(pd.Timedelta(0))
            
            # Mark travel times as None where there's a sequence gap
            trip.loc[trip['has_sequence_gap'], 'observed_travel_time'] = pd.NaT
            
            # Add metadata for analysis
            trip['travel_time_valid'] = ~trip['has_sequence_gap']
            
            # Clean up temporary columns
            trip = trip.drop(columns=['prev_delay', 'prev_sched', 'prev_observed', 'prev_sequence', 'sequence_diff'])
            
            processed_trips.append(trip)
                    
        # Concatenate and store processed data
        df_final = pd.concat(processed_trips, ignore_index=True)
        self.df_final = df_final
        # Run the analysis
        self.categorize_stops_correctly()
        
        # 3. Generate histograms
        self.generate_delay_histograms()
        
        # 4. Add histogram info to stop analysis
        self.add_histograms_to_stop_analysis()

        # 4. Generate travel times data
        self.generate_travel_times_data()
        
        # Create and store the AFTER correction tables
        dir0_table_after, dir1_table_after = self.create_direction_tables(df_final)
        
        # Store as JSON-ready format for export
        self.direction_tables_after = {
            'direction_0': dir0_table_after.reset_index().to_dict('records'),
            'direction_1': dir1_table_after.reset_index().to_dict('records')
        }

        print('after')
        print("=== DIRECTION 0 (ordered by stop_sequence) ===")
        print(dir0_table_after.to_string())

        print("\n=== DIRECTION 1 (ordered by stop_sequence) ===") 
        print(dir1_table_after.to_string())

        # 5. Export everything
        self.export_to_json_files()
        self.export_histograms_to_json()
        self.export_travel_times_to_json()

        # Check and report memory usage reduction
        initial_memory = self.raw_data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        optimized_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        reduction = (1 - optimized_memory / initial_memory) * 100 if initial_memory > 0 else 0
        
        print(f"Memory optimization: {initial_memory:.2f} MB → {optimized_memory:.2f} MB ({reduction:.1f}% reduction)")
        
        return df_final

#   ============================== Preparation ===================================
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
            
        # Add time_type based on hour of first stop
        def categorize_time(row):
            if pd.isna(row['scheduled_departure_time']):
                return 'unknown'
                
            if row['day_type'] != 'weekday':
                return 'weekend'
                
            hour = row['scheduled_departure_time'].hour
            if 6 <= hour < 8: 
                return 'am_rush'
            elif 8 <= hour < 15: 
                return 'day'
            elif 15 <= hour < 17: 
                return 'pm_rush'
            else: 
                return 'night'
            
        df['time_type'] = df.apply(categorize_time, axis=1)

        if all(col in df.columns for col in ['trip_id', 'stop_id', 'direction_id', 'start_date']):
            # Count duplicates before dropping
            duplicate_count = len(df) - len(df.drop_duplicates(subset=['trip_id', 'stop_id', 'direction_id','start_date']))
            print(f'Removing {duplicate_count} duplicates.')
            df = df.drop_duplicates(subset=['trip_id', 'stop_id', 'direction_id', 'start_date'])

        return df

#   ======================== Correction and Detection ============================ 
    def deduplicate_stop_sequences(self, df):
        """Create one log entry per stop with success flag"""
        print("\n=== DEDUPLICATING STOP SEQUENCES ===")
        
        # Initialize the log dictionary at the start
        sequence_corrections_log = {}
        
        # Check for stop names with multiple sequences per direction (the real issue)
        sequence_stats = df.groupby(['stop_name', 'direction_id'])['stop_sequence'].agg(['count', 'nunique']).reset_index()
        problematic_combinations = sequence_stats[sequence_stats['nunique'] > 1]
        
        if len(problematic_combinations) == 0:
            print("No stop sequence duplicates found.")
            self._sequence_corrections_log = sequence_corrections_log  # Assign empty dict
            return df, {}
        
        print(f"Found {len(problematic_combinations)} stop combinations with multiple sequences:")
        
        corrections_made = 0
        df_corrected = df.copy()
        
        # Group by stop_name to create one entry per stop
        for stop_name in problematic_combinations['stop_name'].unique():
            # Get all problematic (stop_name, direction_id) combinations for this stop
            stop_direction_problems = problematic_combinations[problematic_combinations['stop_name'] == stop_name]
            
            total_records_corrected = 0
            correction_success = True  # Start optimistic, set to False if any direction fails
            affected_directions = []  # Track which directions had issues
            
            # Capture what we print for the log
            direction_details = {}
            
            # Process each (stop_name, direction_id) combination that has multiple sequences
            for _, row in stop_direction_problems.iterrows():
                direction_id = row['direction_id']
                
                print(f"\n  {stop_name} (Dir: {direction_id})")
                
                # Add this direction to affected list
                if direction_id not in affected_directions:
                    affected_directions.append(direction_id)
                
                # Get all records for this stop_name + direction_id combination
                direction_mask = (df_corrected['stop_name'] == stop_name) & \
                                (df_corrected['direction_id'] == direction_id)
                
                direction_data = df_corrected[direction_mask]
                
                # Find the most common sequence for this stop_name + direction
                sequence_counts = direction_data['stop_sequence'].value_counts()
                most_common_sequence = sequence_counts.index[0]
                
                sequences_found = dict(sequence_counts)
                print(f"    Sequences found: {sequences_found}")
                
                # Check if there's a clear majority (not a tie)
                has_clear_majority = len(sequence_counts) == 1 or sequence_counts.iloc[0] > sequence_counts.iloc[1]
                
                if has_clear_majority:
                    print(f"    Standardizing all to sequence: {most_common_sequence}")
                    records_to_correct = len(direction_data[direction_data['stop_sequence'] != most_common_sequence])
                    corrections_made += records_to_correct
                    total_records_corrected += records_to_correct
                    df_corrected.loc[direction_mask, 'stop_sequence'] = most_common_sequence
                    
                    # Capture exactly what we printed
                    direction_details[direction_id] = {
                        'sequences_found': sequences_found,
                        'action': f"Standardizing all to sequence: {most_common_sequence}",
                        'records_corrected': records_to_correct
                    }
                    
                else:
                    print(f"    FAILED: Equal counts detected - leaving sequences unchanged")
                    correction_success = False  # Mark as failed immediately
                    
                    # Capture exactly what we printed
                    direction_details[direction_id] = {
                        'sequences_found': sequences_found,
                        'action': "FAILED: Equal counts detected - leaving sequences unchanged",
                        'records_corrected': 0
                    }
            
            route_name = df['route_long_name'].iloc[0] if 'route_long_name' in df.columns else 'unknown'
            composite_key = f"{route_name}_{stop_name}"
            
            # Simple log entry that mirrors the print output
            sequence_corrections_log[composite_key] = {
                'stop_name': stop_name,
                'route_long_name': route_name,
                'issue_type': 'multiple_stop_sequences',
                'resolved': correction_success,
                'affected_directions': sorted(affected_directions),
                'details': direction_details
            }

        # Always assign the log (whether empty or populated)
        self._sequence_corrections_log = sequence_corrections_log
        
        print(f"\nCorrected {corrections_made} stop sequence values")
        if sequence_corrections_log:
            # FIX: Access correction_success through the 'details' key
            failed_stops = len([stop for stop in sequence_corrections_log.values() 
                            if not stop['resolved']])
            if failed_stops > 0:
                print(f"WARNING: {failed_stops} stops still have sequence problems")

        return df_corrected, list(sequence_corrections_log.keys())

    def fix_stop_pairs(self, df):
        """Create one log entry per stop with success flag"""
        
        # Initialize the log dictionary at the start
        direction_corrections_log = {}
        df_corrected = df.copy()  # Work on corrected copy

        # Find stop pairs
        name_to_ids = {}
        for _, row in df_corrected[['stop_id', 'stop_name']].drop_duplicates().iterrows():
            name_to_ids.setdefault(row['stop_name'], []).append(row['stop_id'])
        
        name_to_ids = {name: ids for name, ids in name_to_ids.items() if len(ids) > 1}
        
        if not name_to_ids:
            print("\nNo stop pairs found to process")
            self._direction_pair_corrections_log = direction_corrections_log
            return df_corrected, []
        
        print(f"\nFound {len(name_to_ids)} potential stop pairs to process")
        total_corrections = 0
        
        for stop_name, stop_ids in name_to_ids.items():
            # Analyze directions first - use corrected dataframe
            id_direction_stats = {}
            needs_correction = False
            
            for stop_id in stop_ids:
                dir_counts = df_corrected[df_corrected['stop_id'] == stop_id]['direction_id'].value_counts()
                id_direction_stats[stop_id] = dir_counts
                # Check if this stop_id has mixed directions
                if len(dir_counts) > 1:
                    needs_correction = True
            
            # Only print and process stops that actually need correction
            if not needs_correction:
                continue
                
            print(f"\n  {stop_name}")
            
            # Capture what we print for the log
            stop_id_details = {}
            
            # Print direction stats for problematic stops
            for stop_id in stop_ids:
                dir_counts = id_direction_stats[stop_id]
                dir_0_count = dir_counts.get(0, 0)
                dir_1_count = dir_counts.get(1, 0)
                direction_stats = {0: dir_0_count, 1: dir_1_count}
                print(f"    {stop_id}: {direction_stats}")
                
                # Capture exactly what we printed
                stop_id_details[stop_id] = {
                    'direction_stats': direction_stats
                }
            
            # Continue with correction logic...
            id_to_dominant_dir = {}
            for stop_id, dir_counts in id_direction_stats.items():
                if len(dir_counts) > 0:
                    dominant_dir = dir_counts.idxmax()
                    id_to_dominant_dir[stop_id] = dominant_dir
            
            # Check if there's a clear direction mapping possible
            has_clear_mapping = len(set(id_to_dominant_dir.values())) == len(id_to_dominant_dir)
            
            if not has_clear_mapping:
                print(f"    FAILED: No clear direction mapping possible - leaving assignments unchanged")
                correction_success = False
                pair_corrections = 0
                
                # Add failure info to each stop_id
                for stop_id in stop_ids:
                    stop_id_details[stop_id]['action'] = "FAILED: No clear direction mapping possible - leaving assignments unchanged"
                    stop_id_details[stop_id]['records_corrected'] = 0
                    
            else:
                print(f"    Direction mapping: {id_to_dominant_dir}")
                
                dir_to_stop_id = {}
                for stop_id, dominant_dir in id_to_dominant_dir.items():
                    if dominant_dir in dir_to_stop_id:
                        existing_id = dir_to_stop_id[dominant_dir]
                        existing_count = id_direction_stats[existing_id][dominant_dir]
                        current_count = id_direction_stats[stop_id][dominant_dir]
                        if current_count > existing_count:
                            dir_to_stop_id[dominant_dir] = stop_id
                    else:
                        dir_to_stop_id[dominant_dir] = stop_id
                
                # Apply corrections - use df_corrected consistently
                pair_corrections = 0
                correction_details = []
                
                for stop_id in stop_ids:
                    if stop_id not in id_to_dominant_dir:
                        continue
                    
                    correct_dir = id_to_dominant_dir[stop_id]
                    mismatched_mask = (df_corrected['stop_id'] == stop_id) & (df_corrected['direction_id'] != correct_dir)
                    mismatched_indices = df_corrected[mismatched_mask].index
                    
                    if len(mismatched_indices) > 0:
                        records_fixed = 0
                        
                        for idx in mismatched_indices:
                            actual_dir = df_corrected.at[idx, 'direction_id']
                            correct_stop_id = dir_to_stop_id.get(actual_dir)
                            
                            if correct_stop_id is not None and correct_stop_id != stop_id:
                                df_corrected.at[idx, 'stop_id'] = correct_stop_id
                                records_fixed += 1
                        
                        if records_fixed > 0:
                            pair_corrections += records_fixed
                            # Find target stop_id more safely
                            other_directions = set(id_direction_stats[stop_id].index) - {correct_dir}
                            if other_directions:
                                target_stop_id = dir_to_stop_id.get(list(other_directions)[0], 'unknown')
                                correction_detail = f"Moved {records_fixed} records from {stop_id} → {target_stop_id}"
                                correction_details.append(f"    {correction_detail}")
                                
                                # Capture correction info
                                stop_id_details[stop_id]['action'] = correction_detail
                                stop_id_details[stop_id]['records_corrected'] = records_fixed
                            else:
                                stop_id_details[stop_id]['action'] = f"Corrected {records_fixed} records"
                                stop_id_details[stop_id]['records_corrected'] = records_fixed
                    else:
                        # No corrections needed for this stop_id
                        stop_id_details[stop_id]['action'] = "No corrections needed"
                        stop_id_details[stop_id]['records_corrected'] = 0
                
                if pair_corrections > 0:
                    print(f"    Corrections made:")
                    for detail in correction_details:
                        print(detail)
                
                # Check success immediately after correction attempt - use df_corrected
                correction_success = True
                problematic_stop_ids = []
                
                for stop_id in stop_ids:
                    dir_counts = df_corrected[df_corrected['stop_id'] == stop_id]['direction_id'].value_counts()
                    if len(dir_counts) > 1:  # Still mixed after correction = FAILURE
                        correction_success = False
                        problematic_stop_ids.append(f"{stop_id}: {dict(dir_counts)}")
                
                if not correction_success:
                    print(f"    FAILED: Still has mixed directions after correction")
                    for problem in problematic_stop_ids:
                        print(f"      - {problem}")
            
            # Always create an entry for stops that had corrections OR still have problems
            if pair_corrections > 0 or not correction_success:
                # Create composite key: route_stop_name - use df_corrected
                route_name = df_corrected['route_long_name'].iloc[0] if 'route_long_name' in df_corrected.columns else 'unknown'
                composite_key = f"{route_name}_{stop_name}"
                
                # Determine which directions actually had corrections applied or problems
                affected_directions = []
                for stop_id in stop_ids:
                    if stop_id in id_direction_stats:
                        # If this stop_id had mixed directions, both directions were affected
                        dir_counts = id_direction_stats[stop_id]
                        if len(dir_counts) > 1:  # Mixed directions = both affected
                            for direction in dir_counts.index:
                                if direction not in affected_directions:
                                    affected_directions.append(direction)
                
                # CONSISTENT STRUCTURE - same as deduplicate_stop_sequences
                direction_corrections_log[composite_key] = {
                    'stop_name': stop_name,
                    'route_long_name': route_name,
                    'issue_type': 'incorrect_direction_pair_assignment',
                    'resolved': correction_success,
                    'affected_directions': sorted(affected_directions),
                    'details': stop_id_details  # Same pattern as direction_details
                }
                
                total_corrections += pair_corrections
        
        # Always assign the log
        self._direction_pair_corrections_log = direction_corrections_log
        
        # Final summary - consistent with sequence function
        total_corrected_stops = len([stop for stop in direction_corrections_log.values() if any(detail.get('records_corrected', 0) > 0 for detail in stop['details'].values())])
        print(f"\nCorrected a total of {total_corrections} direction assignments for {total_corrected_stops} stop{'s' if total_corrected_stops != 1 else ''}")
        if direction_corrections_log:
            failed_stops = len([stop for stop in direction_corrections_log.values() if not stop['resolved']])
            if failed_stops > 0:
                print(f"WARNING: {failed_stops} stops still have direction problems")
        
        return df_corrected, list(direction_corrections_log.keys())

    def detect_sequence_root_causes(self, df, drop_threshold=0.1):
        """
        Detect stops that are root causes of sequence tracking issues
        """
        print("\n=== DETECTING SEQUENCE ROOT CAUSES ===")
        
        # Initialize the log dictionary at the start
        sequence_root_causes_log = {}
        
        if not self._sequence_corrections_log:
            print("No sequence corrections found - skipping root cause analysis")
            self._sequence_root_causes_log = sequence_root_causes_log
            return df, []
        
        total_root_causes = 0
        
        # Analyze each direction separately
        for direction in [0, 1]:
            print(f"\n=== ANALYZING SEQUENCE ROOT CAUSES - DIRECTION {direction} ===")
            
            # Get stops ordered by sequence for this direction
            dir_data = df[df['direction_id'] == direction].groupby(['stop_name', 'stop_sequence']).size().reset_index(name='record_count')
            dir_data = dir_data.sort_values('stop_sequence')
            
            if len(dir_data) < 3:  # Need at least 3 stops to detect patterns
                continue
                
            # Calculate record count changes between consecutive stops
            dir_data['prev_count'] = dir_data['record_count'].shift(1)
            dir_data['count_change'] = dir_data['record_count'] - dir_data['prev_count'] 
            dir_data['percent_change'] = dir_data['count_change'] / dir_data['prev_count']
            
            # Detect significant drops followed by recovery
            for i in range(1, len(dir_data) - 1):
                current_row = dir_data.iloc[i]
                next_row = dir_data.iloc[i + 1]
                
                current_change = current_row['percent_change']
                next_change = next_row['percent_change']
                route_name = df['route_long_name'].iloc[0] if 'route_long_name' in df.columns else 'unknown'
                composite_key = f"{route_name}_{current_row['stop_name']}"
                next_composite_key = f"{route_name}_{next_row['stop_name']}"
                
                # Check if this stop has a significant drop AND next stops recover AND next stop had sequence corrections
                if (current_change < -drop_threshold and  # Significant drop at current stop
                    next_change > 0.05 and  # Recovery at next stop
                    next_composite_key in self._sequence_corrections_log):  # Next stop had sequence issues
                    
                    stop_name = current_row['stop_name']
                    prev_count = current_row['prev_count']
                    current_count = current_row['record_count']
                    recovery_count = next_row['record_count']
                    
                    print(f"  ROOT CAUSE DETECTED: {stop_name} (Direction {direction})")
                    print(f"    Sequence {current_row['stop_sequence']}: {prev_count:.0f} → {current_count:.0f} → {recovery_count:.0f}")
                    
                    proportion_passes_current = current_count / prev_count if prev_count > 0 else 0
                    proportion_passes_next = recovery_count / prev_count if prev_count > 0 else 0
                    
                    print(f"    Proportion Passes first: {proportion_passes_current}, Proportion Passes Second: {proportion_passes_next}")
                    print(f"    Next stop '{next_row['stop_name']}' had sequence corrections")

                    # CONSISTENT STRUCTURE - same as other functions
                    if composite_key not in sequence_root_causes_log:
                        # Create new entry following the same pattern
                        sequence_root_causes_log[composite_key] = {
                            'stop_name': stop_name,
                            'route_long_name': route_name,
                            'issue_type': 'sequence_root_cause_detection',
                            'resolved': False,  # Root causes are detected, not resolved
                            'affected_directions': [direction],
                            'details': {
                                direction: {  # Use direction as key like direction_details
                                    'sequence_number': current_row['stop_sequence'],
                                    'count_progression': f"{prev_count:.0f} → {current_count:.0f} → {recovery_count:.0f}",
                                    'proportion_passes_current': proportion_passes_current,
                                    'proportion_passes_next': proportion_passes_next,
                                    'next_stop_name': next_row['stop_name'],
                                    'next_stop_had_corrections': True
                                }
                            }
                        }
                        total_root_causes += 1
                    else:
                        # Add direction if not already present
                        if direction not in sequence_root_causes_log[composite_key]['affected_directions']:
                            sequence_root_causes_log[composite_key]['affected_directions'].append(direction)
                        
                        # Add details for this direction
                        sequence_root_causes_log[composite_key]['details'][direction] = {
                            'sequence_number': current_row['stop_sequence'],
                            'count_progression': f"{prev_count:.0f} → {current_count:.0f} → {recovery_count:.0f}",
                            'proportion_passes_current': proportion_passes_current,
                            'proportion_passes_next': proportion_passes_next,
                            'next_stop_name': next_row['stop_name'],
                            'next_stop_had_corrections': True
                        }
        
        # Always assign the log
        self._sequence_root_causes_log = sequence_root_causes_log

        # Final summary - consistent with other functions
        print(f"\nIdentified {total_root_causes} root cause stop{'s' if total_root_causes != 1 else ''}")
        if sequence_root_causes_log:
            affected_stops = len(sequence_root_causes_log)
            stop_names = [stop['stop_name'] for stop in sequence_root_causes_log.values()]
            print(f"WARNING: {affected_stops} stops identified as root causes: {', '.join(stop_names)}")

        return df, list(sequence_root_causes_log.keys())

#   ========================== Basic Stop Analysis ===============================
    def categorize_stops_correctly(self):
        """
        Categorize stops as directional or shared, and flag problematic ones
        """
        print(f"\n=== CATEGORIZING STOPS ===")
        stop_analysis = {}  # FIX: Use dictionary, not list
        df = self.df_final.copy()

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
            route_name = group['route_long_name'].iloc[0] if 'route_long_name' in df.columns else 'unknown'
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
                'directional_ids': directional_stop_ids
            }

        # Convert dictionary to DataFrame for analysis
        stop_analysis_df = pd.DataFrame(list(stop_analysis.values()))
        
        # Sort by Direction 0 sequence for easier reading
        dir0_sequences = df[df['direction_id'] == 0][['stop_name', 'stop_sequence']].drop_duplicates()
        
        self.stop_analysis = stop_analysis_df.merge(
            dir0_sequences, 
            on='stop_name', 
            how='left'
        ).sort_values('stop_sequence').drop(columns=['stop_sequence'])
        
        # Store dictionary version for consistency with logs
        self.stop_analysis_dict = stop_analysis

        return self.stop_analysis

#   ====================== Additional Routewise Analysis =========================
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
        
        return dir0_data[['Dir_0', 'Dir_1']], dir1_data[['Dir_0', 'Dir_1']]

    def generate_delay_histograms(self, bins=20):
        """
        Generate normalized delay histograms for each route-stop combination by time type
        """
        print("\n=== GENERATING DELAY HISTOGRAMS BY TIME TYPE ===")
        
        route_name = self.route_long_name
        
        self._delay_histograms = {}
        
        # Define time categories (same as travel times)
        time_categories = ['am_rush', 'day', 'pm_rush', 'night', 'weekend']
        
        # Generate histograms for each stop
        for stop_name, group in self.df_final.groupby('stop_name'):
            composite_key = f"{route_name}_{stop_name}"
            
            print(f"Generating histograms for: {stop_name}")
            
            # Initialize stop data structure
            stop_histogram_data = {
                'stop_name': stop_name,
                'route_name': route_name,
                'time_categories': {},
                'metadata': {
                    'bins_used': bins,
                    'generated_date': pd.Timestamp.now().isoformat()
                }
            }
            
            # Process each time category
            for time_category in time_categories:
                # Filter for this time category
                if 'time_type' in group.columns:
                    time_data = group[group['time_type'] == time_category]
                else:
                    # If no time_type column, skip this category
                    continue
                
                if len(time_data) >= 10:  # Minimum sample size per time category
                    
                    # Filter out invalid data and get clean delays
                    total_delays = time_data['departure_delay'].dropna()
                    
                    # For incremental delays, only use non-NaN values (automatically excludes sequence gaps)
                    incremental_delays = time_data['incremental_delay'].dropna()
                    # Additional safety: remove any remaining NaT values if column is object type
                    if time_data['incremental_delay'].dtype == 'object':
                        incremental_delays = incremental_delays[incremental_delays != pd.NaT]
                    
                    # Generate histograms if we have enough data for this time category
                    category_histograms = {}
                    
                    # Total delay histogram
                    if len(total_delays) >= 5:  # Lower threshold for time categories
                        category_histograms['total_delay'] = self._create_normalized_histogram(
                            total_delays, bins, f'Total Delay Distribution - {time_category.replace("_", " ").title()}'
                        )
                    
                    # Incremental delay histogram  
                    if len(incremental_delays) >= 5:  # Lower threshold for time categories
                        category_histograms['incremental_delay'] = self._create_normalized_histogram(
                            incremental_delays, bins, f'Incremental Delay Distribution - {time_category.replace("_", " ").title()}'
                        )
                    
                    # Store if we have any histograms for this time category
                    if category_histograms:
                        stop_histogram_data['time_categories'][time_category] = {
                            'histograms': category_histograms,
                            'metadata': {
                                'total_delay_sample_size': len(total_delays),
                                'incremental_delay_sample_size': len(incremental_delays),
                                'time_category': time_category
                            }
                        }
                        print(f"  ✓ Generated histograms for {time_category}: {len(category_histograms)} types")
            
            # Only store if we have data for at least one time category
            if stop_histogram_data['time_categories']:
                self._delay_histograms[composite_key] = stop_histogram_data
                total_categories = len(stop_histogram_data['time_categories'])
                print(f"  ✓ Complete: {total_categories} time categories with histograms")
            else:
                print(f"  ✗ No sufficient data for any time category")
        
        print(f"\nGenerated histograms for {len(self._delay_histograms)} route-stop combinations")
        
        # Print summary of what was generated
        total_histograms = 0
        for stop_data in self._delay_histograms.values():
            for time_cat_data in stop_data['time_categories'].values():
                total_histograms += len(time_cat_data['histograms'])
        
        print(f"Total individual histograms generated: {total_histograms}")
        return self._delay_histograms

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

    def add_histograms_to_stop_analysis(self):
        """Add histogram availability flags to stop_analysis_dict"""
        
        if not hasattr(self, '_delay_histograms'):
            return
        
        # Update stop_analysis_dict with histogram availability
        for composite_key in self.stop_analysis_dict.keys():
            has_histograms = composite_key in self._delay_histograms
            
            self.stop_analysis_dict[composite_key]['has_histograms'] = has_histograms
            
            if has_histograms:
                # Add quick histogram info with time category breakdown
                hist_data = self._delay_histograms[composite_key]
                
                # Count available time categories and histogram types
                available_time_categories = list(hist_data['time_categories'].keys())
                histogram_summary = {}
                
                for time_cat, time_data in hist_data['time_categories'].items():
                    histogram_summary[time_cat] = {
                        'available_types': list(time_data['histograms'].keys()),
                        'total_delay_sample_size': time_data['metadata']['total_delay_sample_size'],
                        'incremental_delay_sample_size': time_data['metadata']['incremental_delay_sample_size']
                    }
                
                self.stop_analysis_dict[composite_key]['histogram_info'] = {
                    'available_time_categories': available_time_categories,
                    'total_time_categories': len(available_time_categories),
                    'time_category_details': histogram_summary
                }

    def generate_travel_times_data(self):
        """
        Generate travel times statistics for route segments by time category
        """
        print("\n=== GENERATING TRAVEL TIMES DATA ===")
        df_final = self.df_final.copy()
        route_name = self.route_long_name
        self._travel_times_data = {}
        
        # Sort data and create next stop mapping
        df_sorted = df_final.sort_values(['trip_id', 'stop_sequence'])
        df_sorted['next_stop'] = df_sorted.groupby('trip_id')['stop_name'].shift(-1)
        
        # Filter for valid travel times (no sequence gaps + has next stop)
        valid_segments = df_sorted[
            (df_sorted['travel_time_valid'] == True) & 
            (df_sorted['next_stop'].notna()) &
            (df_sorted['observed_travel_time'].notna()) &
            (df_sorted['scheduled_travel_time'].notna())
        ].copy()
        
        # Convert travel times to seconds for analysis
        valid_segments['observed_travel_time_seconds'] = valid_segments['observed_travel_time'].dt.total_seconds()
        valid_segments['scheduled_travel_time_seconds'] = valid_segments['scheduled_travel_time'].dt.total_seconds()
        
        # Define time categories
        time_categories = ['am_rush', 'day', 'pm_rush', 'night', 'weekend']
        
        # Generate statistics for each route segment and time category
        for (from_stop, to_stop), segment_group in valid_segments.groupby(['stop_name', 'next_stop']):
            
            segment_key = f"{from_stop} → {to_stop}"
            route_segment_key = f"{route_name}_{segment_key}"
            
            print(f"Processing segment: {segment_key}")
            
            # Initialize segment data structure
            segment_data = {
                'route_name': route_name,
                'from_stop': from_stop,
                'to_stop': to_stop,
                'time_categories': {}
            }
            
            # Process each time category
            for time_category in time_categories:
                # Filter for this time category
                if 'time_type' in segment_group.columns:
                    time_data = segment_group[segment_group['time_type'] == time_category]
                else:
                    # If no time_type column, skip this category
                    continue
                
                if len(time_data) >= 3:  # Minimum sample size per time category
                    
                    # Get travel times for this time category
                    observed_times = time_data['observed_travel_time_seconds'].dropna()
                    scheduled_times = time_data['scheduled_travel_time_seconds'].dropna()
                    
                    category_stats = {
                        'observed': self._calculate_travel_time_stats(observed_times, 'observed'),
                        'scheduled': self._calculate_travel_time_stats(scheduled_times, 'scheduled'),
                        'comparison': {
                            'mean_difference_seconds': float(observed_times.mean() - scheduled_times.mean()) if len(observed_times) > 0 and len(scheduled_times) > 0 else None,
                            'delay_rate': float((observed_times > scheduled_times).mean()) if len(observed_times) == len(scheduled_times) else None
                        },
                        'sample_size': len(time_data)
                    }
                    
                    segment_data['time_categories'][time_category] = category_stats
            
            # Only store if we have data for at least one time category
            if segment_data['time_categories']:
                self._travel_times_data[route_segment_key] = segment_data
                print(f"  ✓ Generated data for {len(segment_data['time_categories'])} time categories")
        
        print(f"Generated travel time data for {len(self._travel_times_data)} route segments")
        return self._travel_times_data

    def _calculate_travel_time_stats(self, travel_times, data_type):
        """Calculate statistics for a set of travel times"""
        
        if len(travel_times) == 0:
            return None
        
        return {
            'data_type': data_type,
            'mean_travel_time_seconds': float(travel_times.mean()),
            'std_travel_time_seconds': float(travel_times.std()),
            'percentile_25': float(np.percentile(travel_times, 25)),
            'percentile_75': float(np.percentile(travel_times, 75)),
            'percentile_5': float(np.percentile(travel_times, 5)),
            'percentile_95': float(np.percentile(travel_times, 95)),
            'median_travel_time_seconds': float(travel_times.median()),
            'min_travel_time_seconds': float(travel_times.min()),
            'max_travel_time_seconds': float(travel_times.max()),
            'sample_size': len(travel_times)
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
        elif hasattr(obj, 'item'):  # pandas/numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # pandas/numpy arrays
            return obj.tolist()
        else:
            return obj
        

#   ========================== Exporting to JSON =================================    
    def export_to_json_files(self, output_dir="./analysis_output"):
        """
        Export analysis to 5 JSON files, creating new files or appending to existing ones
        """
        print(f"\n=== EXPORTING TO JSON FILES ===")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        file_paths = {
            "route_stops": os.path.join(output_dir, "route_stops.json"),
            "stop_routes": os.path.join(output_dir, "stop_routes.json"),
            "stop_analysis": os.path.join(output_dir, "stop_analysis.json"),
            "logs_details": os.path.join(output_dir, "logs_details.json"),
            "route_tables": os.path.join(output_dir, "route_tables.json")
        }
        
        # Load existing data or initialize empty dictionaries
        def load_existing_json(file_path):
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    print(f"Warning: Could not load {file_path}, starting fresh")
                    return {}
            return {}
        
        # Load existing data
        route_stops_data = load_existing_json(file_paths["route_stops"])
        stop_routes_data = load_existing_json(file_paths["stop_routes"])
        stop_analysis_data = load_existing_json(file_paths["stop_analysis"])
        logs_details_data = load_existing_json(file_paths["logs_details"])
        route_tables_data = load_existing_json(file_paths["route_tables"])
        route_name = self.route_long_name
            
        print(f"Processing route: {route_name}")

        # 1. UPDATE ROUTE-TO-STOPS MAPPING
        if hasattr(self, 'stop_analysis_dict') and self.stop_analysis_dict:
            stops_on_route = list(set(data['stop_name'] for data in self.stop_analysis_dict.values()))
            
            # Order stops by direction 0 sequence
            if hasattr(self, 'df_final'):
                # Get direction 0 sequences for sorting
                dir0_sequences = self.df_final[self.df_final['direction_id'] == 0][['stop_name', 'stop_sequence']].drop_duplicates()
                seq_map = dict(zip(dir0_sequences['stop_name'], dir0_sequences['stop_sequence']))
                
                # Sort stops by sequence
                stops_on_route.sort(key=lambda stop: seq_map.get(stop, 999))
            else:
                stops_on_route = sorted(stops_on_route)
            
            # Calculate summary statistics with updated field names
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
                    "before_and_after_tables": hasattr(self, 'direction_tables_before') and hasattr(self, 'direction_tables_after'),
                    "directional_punctuality": False,  
                    "travel_times": False  
                }
            }
            print(f"  Updated route_stops for {route_name} ({len(stops_on_route)} stops)")
        
        # 2. UPDATE STOP-TO-ROUTES MAPPING
        if hasattr(self, 'stop_analysis_dict') and self.stop_analysis_dict:
            for data in self.stop_analysis_dict.values():
                stop_name = data['stop_name']
                
                # Check if this stop is identified as a regulation stop
                is_regulation_stop = any(composite_key.endswith(f"_{stop_name}") 
                                    for composite_key in getattr(self, '_sequence_root_causes_log', {}))
                
                # Initialize stop entry if it doesn't exist
                if stop_name not in stop_routes_data:
                    stop_routes_data[stop_name] = {
                        "stop_name": stop_name,
                        "routes": [],
                        "total_routes": 0,
                        "summary": {"normal_routes_count": 0, "minor_routes_count": 0, "severe_routes_count": 0},
                        "regulation_stop": is_regulation_stop
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
                    
                    # Update regulation_stop flag if this route identifies it as such
                    if is_regulation_stop:
                        stop_routes_data[stop_name]["regulation_stop"] = True
            
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
        
        # 5. UPDATE ROUTE TABLES 
        if hasattr(self, 'direction_tables_before') and hasattr(self, 'direction_tables_after'):
            # Simply convert your existing tables to JSON
            route_tables_data[route_name] = {
                "route_name": route_name,
                "before_table": self._convert_table_to_json(self.direction_tables_before),
                "after_table": self._convert_table_to_json(self.direction_tables_after)
            }
            print(f"  Updated route_tables for {route_name}")
        else:
            print(f"  No before/after tables found for {route_name}")

        # 6. SAVE ALL FILES
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
        save_json_file(route_tables_data, file_paths["route_tables"], "route_tables.json")
        
        print(f"\nExport complete! Files saved to: {output_dir}")
        
        return {
            "route_stops_count": len(route_stops_data),
            "stop_routes_count": len(stop_routes_data),
            "stop_analysis_count": len(stop_analysis_data),
            "logs_details_count": len(logs_details_data),
            "route_tables_count": len(route_tables_data),
            "output_directory": output_dir
        }

    def export_histograms_to_json(self, output_dir="./analysis_output"):
        """Export histograms to separate JSON file with time category structure"""
        
        if not hasattr(self, '_delay_histograms') or not self._delay_histograms:
            print("No histograms to export")
            return
        
        # Load existing histogram data
        histograms_path = os.path.join(output_dir, "delay_histograms_by_time.json")
        
        def load_existing_json(file_path):
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    return {}
            return {}
        
        existing_histograms = load_existing_json(histograms_path)
        
        # Add current route's histograms
        existing_histograms.update(self._delay_histograms)
        
        # Save updated histograms
        try:
            serializable_data = self._make_json_serializable(existing_histograms)
            with open(histograms_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            # Count total histograms for summary
            total_stops = len(existing_histograms)
            total_time_categories = sum(len(stop_data['time_categories']) for stop_data in existing_histograms.values())
            total_histograms = 0
            for stop_data in existing_histograms.values():
                for time_cat_data in stop_data['time_categories'].values():
                    total_histograms += len(time_cat_data['histograms'])
            
            print(f"  ✓ Saved delay_histograms_by_time.json:")
            print(f"    - {total_stops} stops")
            print(f"    - {total_time_categories} time categories")  
            print(f"    - {total_histograms} individual histograms")
            
        except Exception as e:
            print(f"  ✗ Error saving delay_histograms_by_time.json: {e}")

    def export_travel_times_to_json(self, output_dir="./analysis_output"):
        """Export travel times to separate JSON file"""
        
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
        
        # Group by route for cleaner structure
        route_name = self.df_final['route_long_name'].iloc[0] if 'route_long_name' in self.df_final.columns else 'Unknown Route'
        
        # Create route entry if it doesn't exist
        if route_name not in existing_travel_times:
            existing_travel_times[route_name] = {
                'route_name': route_name,
                'segments': {},
                'metadata': {
                    'last_updated': pd.Timestamp.now().isoformat(),
                    'total_segments': 0
                }
            }
        
        # Add current route's segments
        segments_added = 0
        for route_segment_key, segment_data in self._travel_times_data.items():
            # Extract segment key (remove route prefix)
            segment_key = route_segment_key.replace(f"{route_name}_", "", 1)
            
            existing_travel_times[route_name]['segments'][segment_key] = segment_data
            segments_added += 1
        
        # Update metadata
        existing_travel_times[route_name]['metadata']['total_segments'] = len(existing_travel_times[route_name]['segments'])
        existing_travel_times[route_name]['metadata']['last_updated'] = pd.Timestamp.now().isoformat()
        
        # Save updated travel times
        try:
            serializable_data = self._make_json_serializable(existing_travel_times)
            with open(travel_times_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved travel_times.json: {segments_added} segments added for {route_name}")
        except Exception as e:
            print(f"  ✗ Error saving travel_times.json: {e}")