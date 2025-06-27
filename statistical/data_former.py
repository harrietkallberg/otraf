import pandas as pd
import json as json 
import numpy as np
from collections import defaultdict, Counter
from statistical.lv_logger import LVLogger
import matplotlib.pyplot as plt
import datetime as datetime

class DataFormer:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.route_id = raw_data["route_id"].iloc[0]
        self.route_long_name = self.get_route_long_name()
        self.route_short_name = raw_data["route_short_name"].iloc[0]

        # Initialize logger with extracted route metadata
        self.log = LVLogger({
            "route_id": self.route_id,
            "route_long_name": self.route_long_name,
            "route_short_name": self.route_short_name
        })
        
        # STEP 0: Preprocessing
        self.df_before = self.prepare_columns(raw_data)

        # Run full validation pipeline
        self.create_and_validate_stop_topology(self.df_before)
        self.create_and_validate_direction_topology(self.df_before)
        self.create_and_validate_regulatory_behavior(self.df_before)
        self.generate_performance_logs(self.df_before)

        self.build_routewise_navigation_structure()
        print('NAVIGATIONAL STRUCTURE DONE')

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

        # In DataFormer.__init__, after self.df_before = df.copy():
        for col in ["direction_id", "stop_id", "time_type"]:
            if col in df.columns:
                df[col] = df[col].astype(str)


        return df
    
#   ================ CLEARLY SEPARATED VALIDATION METHODS =====================

    def create_and_validate_stop_topology(self, df):
        """Main topology validation method for 2-layer approach using LVLogger."""
        print("\n=== COMPLETE STOP VALIDATION WORKFLOW ===")

        # STEP 1: Validate and log parent station topology
        self._classify_all_parent_stations_diversity(df)

        # STEP 2: Validate and log stop ID direction behavior
        self._validate_stop_id_directions(df)

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
        stop_ids   = parent_data['stop_id'].unique().tolist()
        stop_names = parent_data['stop_name'].unique().tolist()

        # 1) Gather per-stop directional analysis
        stop_id_analysis = {}
        missing_data = []
        for sid in stop_ids:
            dirs, counts = self._get_stop_id_directions(df, sid, parent_station)
            is_multi = len(dirs) > 1
            stop_id_analysis[str(sid)] = {
                'directions':         dirs,
                'direction_counts':   counts,
                'is_multi_directional': is_multi
            }
            if not dirs:
                missing_data.append(sid)

        total = len(stop_ids)
        multi = sum(a['is_multi_directional'] for a in stop_id_analysis.values())
        single = total - multi

        parent_key = self.log.build_entity_key('stop_topology', parent_station=parent_station)

        def log_and_return(label_desc, parent_violation=None):
            # — single label on the parent station entity —
            label = self.log.create_label_entry(
                label_type='parent_station_stop_topology',
                description=label_desc,
                entity_key=parent_key,
                stop_ids=stop_ids,
                stop_names=stop_names,
                directional_analysis=stop_id_analysis
            )
            self.log.add_label('stop_topology', 'parent_station', parent_key, label)

            # — if invalid, one parent violation + propagate per-stop violations —
            if parent_violation:
                self.log.add_violation('stop_topology', 'parent_station', parent_key, parent_violation)
                print(f"🚩 Parent station {parent_station} [{label_desc}]: {parent_violation['description']}")
                for sid in stop_ids:
                    sk = self.log.build_entity_key(
                        'stop_topology', parent_station=parent_station, stop_id=sid
                    )
                    sv = self.log.create_violation_entry(
                        violation_type='stop_id_from_invalid_parent_topology',
                        severity='medium',
                        description='Inherited invalid-parent-station topology',
                        entity_key=sk,
                        parent_station=parent_station,
                        stop_id=str(sid),
                        details=stop_id_analysis[str(sid)]
                    )
                    self.log.add_violation('stop_topology', 'stop_id', sk, sv)
            else:
                print(f"✅ Parent station {parent_station} [{label_desc}]: Valid topology")

            return label_desc

        # 2) Missing-data
        if missing_data:
            v = self.log.create_violation_entry(
                violation_type='stop_id_missing_direction_data',
                severity='medium',
                description=f"Missing direction data for stops: {missing_data}",
                entity_key=parent_key,
                parent_station=parent_station,
                stop_ids=missing_data,
                details=stop_id_analysis
            )
            return log_and_return('Undefined', v)

        # 3) Single-stop
        if total == 1:
            return log_and_return('Shared' if multi else 'Unidirectional')

        # 4) Two-stop
        if total == 2:
            if single == 2 and multi == 0:
                return log_and_return('Bidirectional')
            v = self.log.create_violation_entry(
                violation_type='two_stop_misassigned_directions',
                severity='high',
                description=f"2-stop station misassigned: {single} single, {multi} multi",
                entity_key=parent_key,
                parent_station=parent_station,
                details={'analysis': stop_id_analysis}
            )
            return log_and_return('Undefined', v)

        # 5) Even >2
        if total % 2 == 0:
            if single == total and multi == 0:
                return log_and_return('Bidirectional')
            if single % 2 == 0 and multi % 2 == 0:
                return log_and_return('Hybrid')
            v = self.log.create_violation_entry(
                violation_type='unpaired_even_directional_stops',
                severity='medium',
                description=f"Even-stop unpaired: {single} single, {multi} multi",
                entity_key=parent_key,
                parent_station=parent_station,
                details={'analysis': stop_id_analysis}
            )
            return log_and_return('Undefined', v)

        # 6) Odd >2
        if single % 2 == 0 and multi % 2 == 1:
            return log_and_return('Hybrid')

        v = self.log.create_violation_entry(
            violation_type='unpaired_odd_directional_stops',
            severity='medium',
            description=f"Odd-stop unpaired: {single} single, {multi} multi",
            entity_key=parent_key,
            parent_station=parent_station,
            details={'analysis': stop_id_analysis}
        )
        return log_and_return('Undefined', v)

    def _validate_stop_id_directions(self, df):
        print("Validating individual stop ID directions...")
        stop_id_data = df[['stop_id', 'stop_name', 'parent_station']].drop_duplicates()
        for _, row in stop_id_data.iterrows():
            sid = row['stop_id']
            name = row['stop_name']
            ps  = row['parent_station']
            key = self.log.build_entity_key("stop_topology", parent_station=ps, stop_id=sid)
            dirs, counts = self._get_stop_id_directions(df, sid, ps, entity_key=key)
            desc = 'multi_directional' if len(dirs)>1 else 'single_directional' if dirs else 'no_data'
            le = self.log.create_label_entry(
                label_type='stop_id_stop_topology',
                description=desc,
                entity_key=key,
                stop_name=name,
                stop_id=sid,
                parent_station=ps,
                directions=dirs,
                direction_counts=counts
            )
            self.log.add_label('stop_topology', 'stop_id', key, le)
            if not dirs:
                ve = self.log.create_violation_entry(
                    violation_type='stop_id_without_direction_data',
                    severity='high',
                    description=f"Stop {sid} has no direction data",
                    entity_key=key,
                    stop_id=sid,
                    stop_name=name,
                    parent_station=ps,
                    details={'directions':[], 'direction_counts':{}}
                )
                self.log.add_violation('stop_topology', 'stop_id', key, ve)
                print(f"🚩 Stop ID {sid}: No direction data")
        print("✅ Stop ID validation complete.")

    def _get_stop_id_directions(self, df, stop_id, parent_station, entity_key=None):
        subset = df[
            (df['stop_id'].astype(str)==str(stop_id)) &
            (df['parent_station'].astype(str)==str(parent_station))
        ]
        if subset.empty:
            if entity_key is None:
                entity_key = self.log.build_entity_key('stop_topology',
                                                    parent_station=parent_station,
                                                    stop_id=stop_id)
            ve = self.log.create_violation_entry(
                violation_type='stop_id_missing_from_schedule',
                severity='medium',
                description=f"No data for stop {stop_id} in parent_station {parent_station}",
                entity_key=entity_key,
                stop_id=stop_id,
                parent_station=parent_station
            )
            self.log.add_violation('stop_topology','stop_id',entity_key,ve)
            print(f"⚠️ {ve['description']}")
            return [], {}
        dirs = subset['direction_id'].unique().tolist()
        counts = subset['direction_id'].value_counts().to_dict()
        return dirs, counts

#   ================ DIRECTION TOPOLOGY VALIDATION (following stop pattern) =====================

    def create_and_validate_direction_topology(self, df):
        print("\n=== COMPLETE DIRECTION VALIDATION WORKFLOW ===")
        self._analyze_all_directions(df)
        print(f"✅ Direction validation complete")

    def _analyze_all_directions(self, df):
        grouped = df.groupby("direction_id")
        for direction_id, group in grouped:
            direction_id = str(direction_id)
            self._analyze_single_direction_topology(group, direction_id)

    def _analyze_single_direction_topology(self, df, direction_id):
        # 1) Count each unique stop‐sequence pattern per trip
        trip_groups = df.groupby(['trip_id', 'start_date'])
        pattern_instance_counter = Counter()
        for _, trip_df in trip_groups:
            sorted_trip = trip_df.sort_values("stop_sequence")
            pattern = tuple(sorted_trip['stop_id'])
            pattern_instance_counter[pattern] += 1

        # 2) Prepare logging helper
        entity_key = self.log.build_entity_key('direction_topology', direction_id=direction_id)
        def log_and_return(description, direction_violation=None):
            label_entry = self.log.create_label_entry(
                label_type="direction_id_direction_topology",
                description=description,
                entity_key=entity_key,
                direction_id=direction_id
            )
            self.log.add_label("direction_topology", "direction", entity_key, label_entry)
            if direction_violation:
                self.log.add_violation("direction_topology", "direction", entity_key, direction_violation)
                print(f"🚩 Direction {direction_id} [{description}]: {direction_violation['description']}")
            else:
                print(f"✅ Direction {direction_id} [{description}]: Valid topology")
            return description

        # 3) No data?
        if not pattern_instance_counter:
            v = self.log.create_violation_entry(
                violation_type="missing_data_for_direction",
                description="no_data",
                severity="high",
                entity_key=entity_key,
                direction_id=direction_id
            )
            return log_and_return("Undefined", v)

        # 4) Valid candidates = strictly sequential patterns (1..N)
        valid_candidates = [
            (p, len(p))
            for p in pattern_instance_counter
            if p and p[0] and np.all(np.diff([1] + list(range(2, len(p)+1))) == 1)
        ]
        if not valid_candidates:
            v = self.log.create_violation_entry(
                violation_type="missing_valid_canonical_for_direction",
                description="no_valid_canonical",
                severity="high",
                entity_key=entity_key,
                direction_id=direction_id
            )
            return log_and_return("Undefined", v)

        # 5) Pick the canonical (longest) pattern
        canonical = max(valid_candidates, key=lambda x: x[1])[0]
        canonical_set = set(canonical)
        canonical_str = self.convert_pattern_to_position_string(list(canonical), list(canonical))
        canonical_count = pattern_instance_counter[canonical]
        self.log.direction_topology_logs \
            .setdefault('metadata', {}) \
            .setdefault('canonical_patterns', {})[direction_id] = list(canonical)

        label_entry = self.log.create_label_entry(
            label_type="direction_canonical_pattern",
            description=canonical_str,
            entity_key=entity_key,
            direction_id=direction_id,
            count=canonical_count
        )
        self.log.add_label("direction_topology", "direction_canonical_pattern", entity_key, label_entry)

        # 6) Prepare structures for alternatives & per-stop tallies
        alt_counter = Counter()
        alt_detailed = {}
        missing_counter = Counter()
        unexpected_counter = Counter()
        missing_by_pattern = defaultdict(Counter)
        unexpected_by_pattern = defaultdict(Counter)

        # 7) Analyze non-canonical patterns
        for pattern, cnt in pattern_instance_counter.items():
            if pattern == canonical:
                continue

            alt_str = self.convert_pattern_to_position_string(list(pattern), list(canonical))
            alt_counter[alt_str] += cnt

            p_set = set(pattern)
            missing   = sorted(canonical_set - p_set)
            unexpected= sorted(p_set - canonical_set)

            for sid in missing:
                missing_counter[sid] += cnt
                missing_by_pattern[sid][alt_str] += cnt
            for sid in unexpected:
                unexpected_counter[sid] += cnt
                unexpected_by_pattern[sid][alt_str] += cnt

            alt_detailed[alt_str] = {
                "count": cnt,
                "missing_stop_ids": [str(s) for s in missing],
                "unexpected_stop_ids": [str(s) for s in unexpected]
            }

        # 8) Multiple-patterns violation?
        if alt_counter:
            v = self.log.create_violation_entry(
                violation_type="multiple_patterns_in_direction",
                severity="low",
                description=f"Direction {direction_id} has {len(alt_counter)} alternative patterns.",
                entity_key=entity_key,
                direction_id=direction_id,
                details={
                    "canonical_pattern":    {canonical_str: canonical_count},
                    "alternative_patterns": alt_detailed
                }
            )
            self.log.add_violation("direction_topology", "direction", entity_key, v)

        # 9) Per-stop labels, violations & prints, in canonical order
        for sid in canonical:
            stop_key = self.log.build_entity_key(
                "direction_topology", direction_id=direction_id, stop_id=sid
            )

            # a) Missing-stop?
            miss_cnt = missing_counter.get(sid, 0)
            if miss_cnt:
                # create label "missing_in_patterns"
                label_entry = self.log.create_label_entry(
                    label_type="missing_in_patterns",
                    description=f"Stop ID {sid} missing in {miss_cnt} trips",
                    entity_key=stop_key,
                    direction_id=direction_id,
                    stop_id=str(sid),
                    details={"patterns": dict(missing_by_pattern[sid])}
                )
                self.log.add_label("direction_topology", "stop_id", stop_key, label_entry)

                # existing violation
                desc = f"Stop ID {sid} is missing from {miss_cnt} trips"
                v = self.log.create_violation_entry(
                    violation_type="missing_stop_id_in_pattern",
                    severity="medium",
                    description=desc,
                    entity_key=stop_key,
                    direction_id=direction_id,
                    stop_id=str(sid),
                    details={
                        "total_missing_count": miss_cnt,
                        "patterns": dict(missing_by_pattern[sid])
                    }
                )
                self.log.add_violation("direction_topology", "stop_id", stop_key, v)
                print(f"🚩 Stop {sid} [missing_stop_id_in_pattern]: {desc}")

            # b) Unexpected-stop?
            unexp_cnt = unexpected_counter.get(sid, 0)
            if unexp_cnt:
                desc = f"Stop ID {sid} unexpectedly appears in {unexp_cnt} trips"
                v = self.log.create_violation_entry(
                    violation_type="unexpected_stop_id_in_pattern",
                    severity="low",
                    description=desc,
                    entity_key=stop_key,
                    direction_id=direction_id,
                    stop_id=str(sid),
                    details={
                        "total_unexpected_count": unexp_cnt,
                        "patterns": dict(unexpected_by_pattern[sid])
                    }
                )
                self.log.add_violation("direction_topology", "stop_id", stop_key, v)
                print(f"🚩 Stop {sid} [unexpected_stop_id_in_pattern]: {desc}")

            # c) Present in all patterns?
            if miss_cnt == 0 and unexp_cnt == 0:
                # create label "present_in_all_patterns"
                label_entry = self.log.create_label_entry(
                    label_type="present_in_all_patterns",
                    description=f"Stop ID {sid} present in all {1 + len(alt_counter)} patterns",
                    entity_key=stop_key,
                    direction_id=direction_id,
                    stop_id=str(sid),
                    details={"total_patterns": 1 + len(alt_counter)}
                )
                self.log.add_label("direction_topology", "stop_id", stop_key, label_entry)

                print(f"✅ Stop {sid}: present in all patterns")

        # 10) Final label & return
        pattern_label_desc = "Multiple Patterns Detected" if alt_counter else "Full Route Only"
        return log_and_return(pattern_label_desc)

    def convert_pattern_to_position_string(self, pattern, canonical):
        canonical_index = {stop_id: idx + 1 for idx, stop_id in enumerate(canonical)}
        canonical_pos = {stop_id: idx for idx, stop_id in enumerate(canonical)}

        segments = []
        i = 0
        while i < len(pattern):
            stop_id = pattern[i]

            if stop_id not in canonical_index:
                # Explicitly mark anomalous stop IDs with asterisks
                segments.append(f"_*{stop_id}*_")
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

            # Add separator "_" if next stop exists and is canonical
            if i < len(pattern) and pattern[i] in canonical_index:
                segments.append("_")

        return "".join(segments)

#   ================ REGULATORY STOPS IDENTIFICATION (following stop pattern) =====================
   
    def create_and_validate_regulatory_behavior(self, df, threshold: float = 0.95):
        """
        Identify regulatory stops: if ≥ `threshold` proportion of scheduled departures have `.second == 0`.
        Works with datetime.datetime or pandas.Timestamp.
        """
        df = df.copy()
        print(f"🔎 Analyzing regulatory stops using threshold {threshold*100:.1f}% zero-second scheduled times...")

        regulatory_count = 0

        for (direction_id, stop_id), group in df.groupby(["direction_id", "stop_id"]):
            times = group["scheduled_departure_time"].dropna()
            if times.empty:
                continue

            zero_second_count = sum(
                1 for t in times
                if (isinstance(t, (pd.Timestamp, datetime.datetime)) and t.second == 0)
                or (not isinstance(t, (pd.Timestamp, datetime.datetime)) and pd.to_datetime(t, errors='coerce').second == 0)
            )

            proportion = zero_second_count / len(times)
            if proportion >= threshold:
                entity_key = self.log.build_entity_key("regulatory_stops", direction_id=direction_id, stop_id=stop_id)
                label = self.log.create_label_entry(
                    label_type="stop_id_regulatory",
                    description="regulatory",
                    entity_key=entity_key,
                    direction_id=direction_id,
                    stop_id=stop_id,
                    threshold=threshold,
                    proportion_zero_seconds=round(proportion, 3)
                )
                self.log.add_label("regulatory_stops", "stop_id_regulatory", entity_key, label)
                regulatory_count += 1

        print(f"✅ {regulatory_count} stops labeled as regulatory (≥ {threshold*100:.1f}% zero-second departures).")

#   ======================== HISTOGRAMS AND PUNCTUALITY DIAGRAMS ==================
    def generate_performance_logs(self, df, min_ratio: float = 0.80):
        print("\n=== GENERATING PERFORMANCE HISTOGRAMS, PUNCTUALITY METRICS, AND TRAVEL TIME STATISTICS ===")

        # 1) Compute travel times & delays
        df = self.calculate_travel_times_and_delays(df.copy())

        # 2) Prepare empty logs
        histograms_log    = {}
        punctuality_log   = {}
        travel_time_log   = {}

        # 3) Grab your precomputed regulatory‐stop labels
        reg_labels = self.log.regulatory_stops_logs.get("stop_id_regulatory_labels", {})

        first_stop_grp = df[df['stop_sequence'] == 1]
        for (direction_id, stop_id, stop_name, time_type), group in first_stop_grp.groupby(
                ['direction_id', 'stop_id', 'stop_name', 'time_type']
        ):
            delay_data = group['departure_delay'].dropna()
            if len(delay_data) < 5:
                continue

            key = self.log.build_entity_key(
                domain="performance",
                direction_id=direction_id,
                stop_id=stop_id,
                time_type=time_type
            )

            # Total-delay histogram only
            total_hist = self._create_normalized_histogram(delay_data)
            histograms_log[key] = {
                'route_id': self.route_id,
                'direction_id': direction_id,
                'stop_id': stop_id,
                'stop_name': stop_name,
                'time_type': time_type,
                'total_delay_histogram': total_hist,
                'incremental_delay_histogram': None
            }

            # Punctuality
            punctuality = self._calculate_punctuality_metrics(delay_data)
            if punctuality:
                is_reg = bool(reg_labels.get(
                    self.log.build_entity_key(
                        "regulatory_stops",
                        direction_id=direction_id,
                        stop_id=stop_id
                    )
                ))
                punctuality_log[key] = {
                    'route_id': self.route_id,
                    'direction_id': direction_id,
                    'stop_id': stop_id,
                    'stop_name': stop_name,
                    'time_type': time_type,
                    'punctuality': punctuality,
                    'is_regulatory_stop': is_reg
                }

        # 4) Build per‐stop histograms and punctuality
        for (direction_id, stop_id, stop_name, time_type), group in df[df['valid_segment']].groupby(
                ['direction_id', 'stop_id', 'stop_name', 'time_type']
        ):
            # Extract delays
            delay_data = group['departure_delay'].dropna()
            incr_data  = group['incremental_delay'].dropna()

            # Skip if too little data
            if len(delay_data) < 5:
                continue

            # Build the entity key
            entity_key = self.log.build_entity_key(
                domain="performance",
                direction_id=direction_id,
                stop_id=stop_id,
                time_type=time_type
            )

            # — Histogram construction —
            # Combine both delay streams to get consistent bins
            combined_data = pd.concat([delay_data, incr_data])
            bin_edges     = np.histogram_bin_edges(combined_data, bins='fd')
            total_hist    = self._create_normalized_histogram(delay_data, bin_edges)
            incr_hist     = self._create_normalized_histogram(incr_data,  bin_edges)

            # Store when at least one histogram exists
            if total_hist or incr_hist:
                histograms_log[entity_key] = {
                    'route_id':                  self.route_id,
                    'direction_id':              direction_id,
                    'stop_id':                   stop_id,
                    'stop_name':                 stop_name,
                    'time_type':                 time_type,
                    'total_delay_histogram':     total_hist,
                    'incremental_delay_histogram': incr_hist
                }

            # — Punctuality metrics —
            punctuality = self._calculate_punctuality_metrics(delay_data)
            if punctuality:
                # Detect regulatory stops
                reg_key = self.log.build_entity_key(
                    "regulatory_stops",
                    direction_id=direction_id,
                    stop_id=stop_id
                )
                is_reg = bool(reg_labels.get(reg_key))

                punctuality_log[entity_key] = {
                    'route_id':            self.route_id,
                    'direction_id':        direction_id,
                    'stop_id':             stop_id,
                    'stop_name':           stop_name,
                    'time_type':           time_type,
                    'punctuality':         punctuality,
                    'is_regulatory_stop':  is_reg
                }

        # 5) Travel-time segment analysis
        df_valid = (
            df[df['valid_segment'] & df['observed_travel_time'].notna()]
            .copy()
        )
        df_valid['from_stop_id']   = df_valid.groupby(
            ['trip_id', 'direction_id', 'start_date']
        )['stop_id'].shift(1)
        df_valid['from_stop_name'] = df_valid.groupby(
            ['trip_id', 'direction_id', 'start_date']
        )['stop_name'].shift(1)
        df_valid['to_stop_id']     = df_valid['stop_id']
        df_valid['to_stop_name']   = df_valid['stop_name']
        df_valid = df_valid.dropna(subset=['from_stop_id', 'observed_travel_time'])

        if 'route_id' not in df_valid.columns:
            df_valid['route_id'] = self.route_id

        segment_cols = ['route_id', 'direction_id', 'time_type', 'from_stop_id', 'to_stop_id']
        for (route_id, direction_id, time_type, from_stop_id, to_stop_id), group in df_valid.groupby(segment_cols):
            travel_times = group['observed_travel_time'].dt.total_seconds().dropna()
            if len(travel_times) < 5:
                continue

            seg_key = self.log.build_entity_key(
                domain="performance",
                direction_id=direction_id,
                from_stop_id=from_stop_id,
                to_stop_id=to_stop_id,
                time_type=time_type
            )

            travel_time_log[seg_key] = {
                'route_id':        route_id,
                'direction_id':    direction_id,
                'time_type':       time_type,
                'from_stop_id':    from_stop_id,
                'from_stop_name':  group['from_stop_name'].iloc[0],
                'to_stop_id':      to_stop_id,
                'to_stop_name':    group['to_stop_name'].iloc[0],
                'sample_size':     len(travel_times),
                'statistics': {
                    'mean':            round(travel_times.mean(), 2),
                    'median':          round(travel_times.median(), 2),
                    'std':             round(travel_times.std(), 2),
                    'min':             int(travel_times.min()),
                    'max':             int(travel_times.max()),
                    'percentile_25':   int(np.percentile(travel_times, 25)),
                    'percentile_75':   int(np.percentile(travel_times, 75)),
                    'percentile_95':   int(np.percentile(travel_times, 95))
                }
            }

        # 6) Write back to LVLogger
        self.log.performance_logs['histograms_stops']      = histograms_log
        self.log.performance_logs['punctuality_barcharts'] = punctuality_log
        self.log.performance_logs['travel_times']          = travel_time_log

        print(
            f"  ✅ {len(histograms_log)} histograms, "
            f"{len(punctuality_log)} punctuality entries, "
            f"{len(travel_time_log)} travel time segments logged."
        )

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
                'too_late': delays > 179
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

    def plot_performance_entry(self, key: str):
        """
        Plots a histogram or punctuality bar chart stored under the given key.
        """

        # Try histogram first
        hist_entry = self.log.performance_logs['histograms_stops'].get(key)
        if hist_entry:
            h = hist_entry['total_delay_histogram'] # or h = hist_entry['incremental_delay_histogram']
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

    def get_punctuality_flow(self, direction_id: str, time_type: str = "all_day"):
        """
        Return a list of punctuality metrics for each stop in canonical order for given direction.
        Now includes a True/False `is_regulatory_stop` flag.
        """
        flow = []

        pattern_dict = self.log.direction_topology_logs.get('metadata', {}) \
                                   .get('canonical_patterns', {})
        stop_sequence = pattern_dict.get(direction_id, [])

        punctuality_barcharts = self.log.performance_logs.get('punctuality_barcharts', {})

        for stop_id in stop_sequence:
            key = f"stop_id_{stop_id}_direction_{direction_id}_time_{time_type}"
            entry = punctuality_barcharts.get(key)
            if not entry:
                continue

            dist = entry['punctuality']['punctuality_distribution']['percentages']
            flow.append({
                'stop_id':              stop_id,
                'stop_name':            entry['stop_name'],
                'on_time_pct':          dist['on_time'],
                'too_early_pct':        dist['too_early'],
                'too_late_pct':         dist['too_late'],
                'sample_size':          entry['punctuality']['sample_size'],
                'is_regulatory_stop':   entry.get('is_regulatory_stop', False)
            })

        return flow

#   ============================== ROUTEWISE NAVIGATIONAL MAP ===================================

    def build_routewise_navigation_structure(self):
        """
        Build a lightweight, pointer-only navigation map:
        {
        "route_id": ...,
        "route_name": ...,
        "directions": [
            {
            "direction_id": ...,
            "direction_label_keys": [...],
            "direction_violation_keys": [...],
            "has_direction_violations": ...,
            "contains_violations": ...,
            "canonical_pattern": { pattern_str, pattern_ids, count, label_key },
            "alternative_patterns": [ … ],
            "travel_time_segments": [ … ],
            "stops": [ … ],
            },
            …
        ]
        }
        """
        dir_logs      = self.log.direction_topology_logs
        stop_logs     = self.log.stop_topology_logs
        reg_logs      = self.log.regulatory_stops_logs
        perf_logs     = self.log.performance_logs

        # Top-level pointers & metadata
        canonical_patterns                  = dir_logs.get("metadata", {}).get("canonical_patterns", {})
        classification_labels               = dir_logs.get("direction_labels", {})
        canonical_pattern_label_entries     = dir_logs.get("direction_canonical_pattern_labels", {})
        direction_violations                = dir_logs.get("direction_violations", {})
        histograms                          = perf_logs.get("histograms_stops", {})
        punctuality                         = perf_logs.get("punctuality_barcharts", {})
        travel_times                        = perf_logs.get("travel_times", {})

        # Pre-aggregate stop-level violations
        all_stop_violations = self._collect_all_stop_id_violations()

        nav = {
            "route_id":   self.route_id,
            "route_name": self.route_long_name,
            "directions": []
        }

        # Determine all time_types
        all_tts = {
            e.get("time_type")
            for e in list(histograms.values()) + list(punctuality.values()) + list(travel_times.values())
            if e.get("time_type") is not None
        }

        for direction_id, pattern_ids in canonical_patterns.items():
            dir_id_str = str(direction_id)
            dir_key = self.log.build_entity_key("direction_topology", direction_id=dir_id_str)

            # --- Gather direction-level keys ---
            # classification label keys for this direction
            direction_label_keys = [
                k for k, le in classification_labels.items()
                if (isinstance(le, dict) and le.get("entity_key") == dir_key) or k == dir_key
            ]
            # canonical-pattern label key, if any
            canonical_label_key = next(
                (k for k, le in canonical_pattern_label_entries.items()
                if isinstance(le, dict)
                    and le.get("entity_key") == dir_key
                    and le.get("label_type") == "direction_canonical_pattern"),
                None
            )

            # Flags
            has_dir_viol = any(v.get("entity_key") == dir_key
                            for v in direction_violations.values())
            contains_dir_viol = any(
                any(all_stop_violations.get(str(sid), {}).get(domain, []))
                for sid in pattern_ids
                for domain in ("stop_topology", "direction_topology", "regulatory", "parent_station")
            )

            # --- Canonical Pattern detail ---
            canonical_str = self.convert_pattern_to_position_string(list(pattern_ids), list(pattern_ids))
            canonical_count = None
            if canonical_label_key:
                lbl = canonical_pattern_label_entries[canonical_label_key]
                canonical_str = lbl.get("description", canonical_str)
                canonical_count = lbl.get("count")
            canonical = {
                "pattern_str": canonical_str,
                "pattern_ids": [str(sid) for sid in pattern_ids],
                "count":       canonical_count,
                "label_key":   canonical_label_key
            }

            # --- Alternative Patterns (unchanged) ---
            alts = []
            multiple_patterns_violation = next(
                (v for v in direction_violations.values()
                if v.get("entity_key") == dir_key
                and v.get("violation_type") == "multiple_patterns_in_direction"),
                None
            )
            if multiple_patterns_violation:
                for alt_str, alt_info in multiple_patterns_violation.get("details", {}).get("alternative_patterns", {}).items():
                    count = alt_info.get("count", 0)
                    missing_sids    = alt_info.get("missing_stop_ids", [])
                    unexpected_sids = alt_info.get("unexpected_stop_ids", [])
                    missing_keys = [
                        self.log.build_entity_key("direction_topology",
                                                direction_id=dir_id_str, stop_id=sid)
                        for sid in missing_sids
                        if self.log.build_entity_key("direction_topology",
                                                    direction_id=dir_id_str, stop_id=sid)
                        in dir_logs.get("stop_id_violations", {})
                    ]
                    unexpected_keys = [
                        self.log.build_entity_key("direction_topology",
                                                direction_id=dir_id_str, stop_id=sid)
                        for sid in unexpected_sids
                        if self.log.build_entity_key("direction_topology",
                                                    direction_id=dir_id_str, stop_id=sid)
                        in dir_logs.get("stop_id_violations", {})
                    ]
                    alts.append({
                        "pattern_str": alt_str,
                        "count": count,
                        "missing_stop_id_violation_keys": missing_keys,
                        "unexpected_stop_id_violation_keys": unexpected_keys
                    })

            # --- Travel-time Segments (unchanged) ---
            seg_map = {}
            for key, entry in travel_times.items():
                if entry.get("direction_id") != dir_id_str:
                    continue
                seg = (str(entry.get("from_stop_id")), str(entry.get("to_stop_id")))
                seg_map.setdefault(seg, []).append((entry.get("time_type"), key))
            travel_time_segments = []
            for (from_id, to_id), pairs in seg_map.items():
                availability = {tt: None for tt in sorted(all_tts)}
                for tt, k in pairs:
                    availability[tt] = k
                travel_time_segments.append({
                    "from_stop_id":  from_id,
                    "to_stop_id":    to_id,
                    "availability":  availability
                })

            # --- Stops List, now including direction-level keys per stop ---
            stops = []
            for pos, stop_id in enumerate(pattern_ids, start=1):
                sid = str(stop_id)
                meta = self._get_stop_metadata(self.df_before, sid)

                # collect labels & violations per domain
                labels = {d: [] for d in ("stop_topology","direction_topology","regulatory","parent_station")}
                viols  = {d: [] for d in ("stop_topology","direction_topology","regulatory","parent_station")}

                # stop_topology
                for k, e in stop_logs.get("stop_id_labels", {}).items():
                    if str(e.get("stop_id")) == sid:
                        labels["stop_topology"].append(k)
                for k, e in stop_logs.get("stop_id_violations", {}).items():
                    if str(e.get("stop_id")) == sid:
                        viols["stop_topology"].append(k)

                # direction_topology: include stop-id labels + direction-level keys
                for k, e in dir_logs.get("stop_id_labels", {}).items():
                    if str(e.get("stop_id")) == sid and e.get("direction_id") == dir_id_str:
                        labels["direction_topology"].append(k)
                for k, e in dir_logs.get("stop_id_violations", {}).items():
                    if str(e.get("stop_id")) == sid and e.get("direction_id") == dir_id_str:
                        viols["direction_topology"].append(k)
                # **new**: also inject classification & canonical direction labels here
                for k in direction_label_keys:
                    labels["direction_topology"].append(k)
                if canonical_label_key:
                    labels["direction_topology"].append(canonical_label_key)

                # regulatory
                for k, e in reg_logs.get("stop_id_regulatory_labels", {}).items():
                    if str(e.get("stop_id")) == sid and e.get("direction_id") == dir_id_str:
                        labels["regulatory"].append(k)
                # parent_station
                for key, entry in stop_logs.get("parent_station_labels", {}).items():
                    if sid in entry.get("stop_ids", []):
                        labels["parent_station"].append(key)
                for key, entry in stop_logs.get("parent_station_violations", {}).items():
                    if sid in entry.get("stop_ids", []):
                        viols["parent_station"].append(key)

                # Performance availability (unchanged) …
                tts_for_stop = {
                    e.get("time_type") for e in histograms.values()
                    if e.get("direction_id")==dir_id_str and str(e.get("stop_id"))==sid
                } | {
                    e.get("time_type") for e in punctuality.values()
                    if e.get("direction_id")==dir_id_str and str(e.get("stop_id"))==sid
                }
                perf_avail = {"histograms": {}, "punctuality": {}}
                for tt in sorted(tts_for_stop):
                    hkey = next((k for k, e in histograms.items()
                                if e.get("direction_id")==dir_id_str
                                and str(e.get("stop_id"))==sid
                                and e.get("time_type")==tt), None)
                    pkey = next((k for k, e in punctuality.items()
                                if e.get("direction_id")==dir_id_str
                                and str(e.get("stop_id"))==sid
                                and e.get("time_type")==tt), None)
                    perf_avail["histograms"][tt]  = hkey
                    perf_avail["punctuality"][tt] = pkey

                stops.append({
                    "position": pos,
                    "stop_id": sid,
                    "stop_name": meta.get("stop_name"),
                    "labels": labels,
                    "has_violations": any(viols.values()),
                    "violations": viols,
                    "performance_availability": perf_avail
                })

            # Direction violation keys (unchanged)
            direction_violation_keys = [
                k for k, v in direction_violations.items()
                if v.get("entity_key") == dir_key or k == dir_key
            ]

            # Append direction entry
            nav["directions"].append({
                "direction_id":             dir_id_str,
                "direction_label_keys":     direction_label_keys,
                "direction_violation_keys": direction_violation_keys,
                "has_direction_violations": has_dir_viol,
                "contains_violations":      contains_dir_viol,
                "canonical_pattern":        canonical,
                "alternative_patterns":     alts,
                "travel_time_segments":     travel_time_segments,
                "stops":                    stops
            })

        # Write back
        self.log.navigation_structures["routewise_navigation"] = nav



    def _get_stop_metadata(self, df, stop_id: str) -> dict:
        """
        Retrieve metadata for a given stop_id from the final DataFrame.
        Returns stop_name and parent_station, or defaults if not found.
        """
        df = df.copy()
        match = df[df['stop_id'].astype(str) == str(stop_id)]

        if not match.empty:
            first_row = match.iloc[0]
            return {
                "stop_name": first_row.get("stop_name", "UNKNOWN"),
                "parent_station": first_row.get("parent_station") or first_row.get("stop_name", "UNKNOWN")
            }
        return {
            "stop_name": "UNKNOWN",
            "parent_station": "UNKNOWN"
        }

    def _collect_all_stop_id_violations(self):
        """
        Collect all stop-level violations across domains and return:
        {
            stop_id: {
                "stop_topology": [...],
                "direction_topology": [...],
                "regulatory": [...]
            }
        }
        """
        stop_violations = defaultdict(lambda: defaultdict(list))

        domain_map = {
            "stop_topology":      self.log.stop_topology_logs.get("stop_id_violations", {}).values(),
            "direction_topology": self.log.direction_topology_logs.get("stop_id_violations", {}).values(),
            "regulatory":         self.log.regulatory_stops_logs.get("stop_id_regulatory_violations", {}).values(),
            "parent_station":     self.log.stop_topology_logs.get("parent_station_violations", {}).values()
        }

        for domain, source in domain_map.items():
            for v in source:
                stop_id = v.get("stop_id")
                if not stop_id:
                    entity_key = v.get("entity_key", "")
                    parts = entity_key.split("__")
                    if len(parts) == 3 and parts[1] == "stop_id":
                        stop_id = parts[2]
                if stop_id:
                    label = v.get("label") or v.get("violation_type") or "unknown"
                    stop_violations[str(stop_id)][domain].append(label)

        return dict(stop_violations)
