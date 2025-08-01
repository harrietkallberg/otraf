import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from statistical.lv_logger import LVLogger
import matplotlib.pyplot as plt
import datetime as datetime
import itertools

class DataFormer:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.route_id = raw_data["route_id"].iloc[0]
        self.route_long_name = raw_data["route_long_name"].iloc[0]
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

#   ================================ Preparation =====================================

    def prepare_columns(self, df):
        """Simplified version with better performance"""
        df = df.copy()
        
        # Convert timestamps
        df['observed_departure_time'] = pd.to_datetime(df['observed_departure_time'], unit='s')
        df['scheduled_departure_time'] = pd.to_datetime(df['scheduled_departure_time'])

        # Convert to string - use .astype() for Series
        df['route_id'] = df['route_id'].astype(str)
        df['route_long_name'] = df['route_long_name'].astype(str)
        df['route_short_name'] = df['route_short_name'].astype(str)

        # For direction_id, handle potential floats first, then convert to string
        df['direction_id'] = df['direction_id'].astype(int).astype(str)
        
        # Fix observed_departure_time if needed
        if 'departure_delay' in df.columns:
            df['observed_departure_time'] = df['scheduled_departure_time'] + pd.to_timedelta(df['departure_delay'], unit='s')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['trip_id', 'stop_id', 'stop_name','direction_id', 'start_date'])
        
        # Date processing
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['month'] = df['start_date'].dt.month
        df['month_type'] = (df['start_date'].dt.month.between(6, 8)).astype(int)
        
        # Day type
        weekday_num = df['start_date'].dt.weekday
        df['day_type'] = weekday_num.map({
            0: 'weekday', 1: 'weekday', 2: 'weekday', 3: 'weekday', 4: 'weekday',
            5: 'saturday', 6: 'sunday'
        })
        
        # Get trip start times
        trip_start_times = (
            df.sort_values('stop_sequence')
            .groupby(['trip_id', 'direction_id', 'start_date'])['scheduled_departure_time']
            .first()
            .reset_index(name='start_time')
        )
        
        df = df.merge(trip_start_times, on=['trip_id', 'direction_id', 'start_date'], how='left')
        
        # Time type - vectorized approach
        hours = df['start_time'].dt.hour
        
        # Initialize all as weekend
        df['time_type'] = 'weekend'
        
        # For weekdays only
        weekday_mask = df['day_type'] == 'weekday'
        
        # Apply time categories to weekdays
        df.loc[weekday_mask & (hours >= 6) & (hours < 9), 'time_type'] = 'am_rush'
        df.loc[weekday_mask & (hours >= 9) & (hours < 15), 'time_type'] = 'day'
        df.loc[weekday_mask & (hours >= 15) & (hours < 17), 'time_type'] = 'pm_rush'
        df.loc[weekday_mask & ~((hours >= 6) & (hours < 17)), 'time_type'] = 'night'
        
        # Handle missing start_time
        df.loc[df['start_time'].isna(), 'time_type'] = 'unknown'
        
        # Convert to string
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
        # collect all raw names …
        names = parent_data['stop_name'].astype(str)
        # … then pick the most common (mode) as a single name
        stop_name = names.mode().iloc[0] if not names.empty else str(parent_station)


        # 1) Gather per-stop directional analysis
        stop_id_analysis = {}
        missing_data = []
        for sid in stop_ids:
            dirs, counts = self._get_stop_id_directions(df, sid, parent_station, stop_name)
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
                stop_name=stop_name,
                parent_station=parent_station,
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
                        severity=3,
                        description='Inherited invalid-parent-station topology',
                        entity_key=sk,
                        parent_station=parent_station,
                        stop_id=str(sid),
                        stop_name=stop_name,
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
                severity=5,
                description=f"Missing direction data for stops: {missing_data}",
                entity_key=parent_key,
                parent_station=parent_station,
                stop_ids=missing_data,
                stop_name=stop_name,
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
                severity=4,
                description=f"2-stop station misassigned: {single} single, {multi} multi",
                entity_key=parent_key,
                parent_station=parent_station,
                stop_name=stop_name,
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
                severity=4,
                description=f"Even-stop unpaired: {single} single, {multi} multi",
                entity_key=parent_key,
                parent_station=parent_station,
                stop_name=stop_name,
                details={'analysis': stop_id_analysis}
            )
            return log_and_return('Undefined', v)

        # 6) Odd >2
        if single % 2 == 0 and multi % 2 == 1:
            return log_and_return('Hybrid')

        v = self.log.create_violation_entry(
            violation_type='unpaired_odd_directional_stops',
            severity=4,
            description=f"Odd-stop unpaired: {single} single, {multi} multi",
            entity_key=parent_key,
            parent_station=parent_station,
            stop_name=stop_name,
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
            dirs, counts = self._get_stop_id_directions(df, sid, ps, name, entity_key=key)
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
                    severity=5,
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

    def _get_stop_id_directions(self, df, stop_id, parent_station, stop_name, entity_key=None):
        """
        Returns the list of directions and their counts for a given stop_id under a parent_station.
        If no data is found, logs a violation with a single stop_name picked by mode.
        """
        subset = df[
            (df['stop_id'].astype(str) == str(stop_id)) &
            (df['parent_station'].astype(str) == str(parent_station))
        ]
        # If there's no data for this stop under the given parent_station, log a violation
        if subset.empty:
            if entity_key is None:
                entity_key = self.log.build_entity_key(
                    'stop_topology',
                    parent_station=parent_station,
                    stop_id=stop_id
                )
            ve = self.log.create_violation_entry(
                violation_type='stop_id_missing_from_schedule',
                severity=3,
                description=f"No data for stop {stop_id} in parent_station {parent_station}",
                entity_key=entity_key,
                stop_id=stop_id,
                stop_name=stop_name,
                parent_station=parent_station,
            )
            self.log.add_violation('stop_topology', 'stop_id', entity_key, ve)
            print(f"⚠️ {ve['description']}")
            return [], {}

        # Otherwise, compute unique directions and their occurrence counts
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
                severity=5,
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
                severity=5,
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

        total_trips = sum(pattern_instance_counter.values())

        md = self.log.direction_topology_logs['metadata']
        rs = md['route_summary']
        rs['total_trip_instances'] = total_trips
        rs['canonical_share']       = canonical_count / total_trips

        # 8) Multiple-patterns violation?
        if alt_counter:
            v = self.log.create_violation_entry(
                violation_type="multiple_patterns_in_direction",
                severity=1,
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
            # one-off lookup of stop names for this direction
            stop_names = df[df['stop_id'].astype(str) == str(sid)]['stop_name'].astype(str)
            common_name = stop_names.mode().iloc[0] if not stop_names.empty else str(sid)

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
                    stop_name = common_name,
                    details={"patterns": dict(missing_by_pattern[sid])}
                )
                self.log.add_label("direction_topology", "stop_id", stop_key, label_entry)
                
                # existing violation
                miss_perc = miss_cnt/total_trips *100
                desc = f"Stop ID {sid} is missing from {miss_cnt} trips, {miss_perc:.2f} %."
                severity = 1 + sum(miss_perc > t for t in (5, 10, 15, 25))
                v = self.log.create_violation_entry(
                    violation_type="missing_stop_id_in_pattern",
                    severity = severity,
                    description=desc,
                    entity_key=stop_key,
                    direction_id=direction_id,
                    stop_id=str(sid),
                    stop_name = common_name,
                    details={
                        "total_missing_count": miss_cnt,
                        "total_missing_perc": miss_perc,
                        "patterns": dict(missing_by_pattern[sid])
                    }
                )
                self.log.add_violation("direction_topology", "stop_id", stop_key, v)
                print(f"🚩 Stop {sid} [missing_stop_id_in_pattern]: {desc}")

            # b) Unexpected-stop?
            unexp_cnt = unexpected_counter.get(sid, 0)
            if unexp_cnt:
                # create label "missing_in_patterns"
                label_entry = self.log.create_label_entry(
                    label_type="unexpected_in_patterns",
                    description=f"Stop ID {sid} unexpected in {unexp_cnt} trips",
                    entity_key=stop_key,
                    direction_id=direction_id,
                    stop_id=str(sid),
                    stop_name = common_name,
                    details={"patterns": dict(unexpected_by_pattern[sid])}
                )
                self.log.add_label("direction_topology", "stop_id", stop_key, label_entry)

                unexp_perc = unexp_cnt/total_trips *100
                desc = f"Stop ID {sid} unexpectedly appears in {unexp_cnt} trips, {unexp_perc} %."
                v = self.log.create_violation_entry(
                    violation_type="unexpected_stop_id_in_pattern",
                    severity=5,
                    description=desc,
                    entity_key=stop_key,
                    direction_id=direction_id,
                    stop_id=str(sid),
                    stop_name = common_name,
                    details={
                        "total_unexpected_count": unexp_cnt,
                        "total_unexpected_perc": unexp_perc,
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
                    stop_name = common_name,
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
            # one-off lookup of stop names for this direction
            stop_names = df[df['stop_id'].astype(str) == str(stop_id)]['stop_name'].astype(str)
            common_name = stop_names.mode().iloc[0] if not stop_names.empty else str(stop_id)

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
                    stop_name = common_name,
                    threshold=threshold,
                    proportion_zero_seconds=round(proportion, 3)
                )
                self.log.add_label("regulatory_stops", "stop_id_regulatory", entity_key, label)
                regulatory_count += 1

        print(f"✅ {regulatory_count} stops labeled as regulatory (≥ {threshold*100:.1f}% zero-second departures).")

#   ======================== HISTOGRAMS AND PUNCTUALITY DIAGRAMS ==================
    
    def generate_performance_logs(self, df, min_ratio: float = 0.80):
        print("\n=== GENERATING PERFORMANCE HISTOGRAMS, PUNCTUALITY METRICS, AND TRAVEL TIME STATISTICS ===")

        # Compute travel times & delays
        df = self.calculate_travel_times_and_delays(df.copy())

        # Prepare empty logs
        analytics_log, travel_time_log = {}, {}
        reg_labels = self.log.regulatory_stops_logs.get("stop_id_regulatory_labels", {})

        # Define fixed bin edges for all histograms
        fixed_bin_edges = np.linspace(-180, 300, num=17)  # 16 bins of 30s

        # Travel-time segment analysis (Unchanged)
        df_valid = df[df['valid_segment'] & df['observed_travel_time'].notna()].copy()
        df_valid['from_stop_id'] = df_valid.groupby(['trip_id', 'direction_id', 'start_date'])['stop_id'].shift(1)
        df_valid['from_stop_name'] = df_valid.groupby(['trip_id', 'direction_id', 'start_date'])['stop_name'].shift(1)
        df_valid['to_stop_id'] = df_valid['stop_id']
        df_valid['to_stop_name'] = df_valid['stop_name']
        df_valid = df_valid.dropna(subset=['from_stop_id', 'observed_travel_time'])

        for (direction_id, time_type, from_stop_id, to_stop_id), group in df_valid.groupby(
                ['direction_id', 'time_type', 'from_stop_id', 'to_stop_id']):

            travel_times = group['observed_travel_time'].dt.total_seconds().dropna()
            if len(travel_times) < 5:
                continue

            seg_key = self.log.build_entity_key(
                domain="performance", direction_id=direction_id,
                from_stop_id=from_stop_id, to_stop_id=to_stop_id, time_type=time_type
            )

            travel_time_log[seg_key] = {
                'route_id': self.route_id, 'direction_id': direction_id, 'time_type': time_type,
                'from_stop_id': from_stop_id, 'from_stop_name': group['from_stop_name'].iloc[0],
                'to_stop_id': to_stop_id, 'to_stop_name': group['to_stop_name'].iloc[0],
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

        # Process all stop data for histograms and punctuality
        for (direction_id, stop_id, stop_name, time_type), group in df_valid.groupby(
                ['direction_id', 'to_stop_id', 'to_stop_name', 'time_type']):

            delay_data = group['departure_delay'].dropna()
            incr_data = group['incremental_delay'].dropna()

            if (len(delay_data) < 5) or (len(incr_data) < 5):
                continue

            key = self.log.build_entity_key(
                domain="performance", direction_id=direction_id, stop_id=stop_id, time_type=time_type
            )

            if key not in analytics_log:
                analytics_log[key] = {
                    'route_id': self.route_id, 'direction_id': direction_id, 'stop_id': stop_id,
                    'stop_name': stop_name, 'time_type': time_type, 'analytics': {}
                }

            total_hist = self._create_normalized_histogram(delay_data, bin_edges=fixed_bin_edges)
            incr_hist = self._create_normalized_histogram(incr_data, bin_edges=fixed_bin_edges)

            if total_hist and incr_hist:
                analytics_log[key]['analytics']['total_delay_histogram'] = {
                    'step_start': total_hist['step_start'],
                    'step_size': total_hist['step_size'],
                    'num_bins': total_hist['num_bins'],
                    'proportions': total_hist['proportions']
                }
                analytics_log[key]['analytics']['incremental_delay_histogram'] = {
                    'step_start': incr_hist['step_start'],
                    'step_size': incr_hist['step_size'],
                    'num_bins': incr_hist['num_bins'],
                    'proportions': incr_hist['proportions']
                }

                punctuality = self._calculate_punctuality_metrics(delay_data)
                if punctuality:
                    is_reg = bool(reg_labels.get(self.log.build_entity_key(
                        "regulatory_stops", direction_id=direction_id, stop_id=stop_id)))
                    analytics_log[key]['analytics']['punctuality'] = punctuality
                    analytics_log[key]['analytics']['is_regulatory_stop'] = is_reg

        for (direction_id, stop_id, stop_name, time_type), group in df.groupby(
                ['direction_id', 'stop_id', 'stop_name', 'time_type']):

            delay_data = group['departure_delay'].dropna()
            if len(delay_data) < 5:
                continue

            key = self.log.build_entity_key(
                domain="performance", direction_id=direction_id, stop_id=stop_id, time_type=time_type
            )

            if key not in analytics_log:
                analytics_log[key] = {
                    'route_id': self.route_id, 'direction_id': direction_id, 'stop_id': stop_id,
                    'stop_name': stop_name, 'time_type': time_type, 'analytics': {'total_delay_histogram': {}}
                }

            total_hist = self._create_normalized_histogram(delay_data, bin_edges=fixed_bin_edges)
            if total_hist:
                analytics_log[key]['analytics']['total_delay_histogram'] = {
                    'step_start': total_hist['step_start'],
                    'step_size': total_hist['step_size'],
                    'num_bins': total_hist['num_bins'],
                    'proportions': total_hist['proportions']
                }

                punctuality = self._calculate_punctuality_metrics(delay_data)
                if punctuality:
                    is_reg = bool(reg_labels.get(self.log.build_entity_key(
                        "regulatory_stops", direction_id=direction_id, stop_id=stop_id)))
                    analytics_log[key]['analytics']['punctuality'] = punctuality
                    analytics_log[key]['analytics']['is_regulatory_stop'] = is_reg

        # Compute route-level performance summary
        all_delays = df['departure_delay'].dropna()
        if len(all_delays) >= 1:
            route_metrics = self._calculate_punctuality_metrics(all_delays)
            md = self.log.performance_logs['metadata']['performance_summary']
            pdist = route_metrics['punctuality_distribution']['percentages']
            md.update({
                'overall_too_early_rate': pdist.get('too_early', 0.0),
                'overall_on_time_rate': pdist.get('on_time', 0.0),
                'overall_too_late_rate': pdist.get('too_late', 0.0),
                'average_departure_delay': route_metrics['basic_statistics']['mean_delay']
            })

        # Write to log
        self.log.performance_logs['analytics_logs'] = analytics_log
        self.log.performance_logs['travel_times'] = travel_time_log

        print(f"  ✅ {len(analytics_log)} analytics entries, {len(travel_time_log)} travel time segments logged.")

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

    def _create_normalized_histogram(self, data, bin_edges=None, bins=10):
        if len(data) < 5:
            return None

        # Calculate bin counts
        counts, _ = np.histogram(data, bins=bin_edges)
        total_count = counts.sum()
        if total_count == 0:
            return None

        proportions = [round(c / total_count, 4) for c in counts]

        return {
            "step_start": round(bin_edges[0], 2),            # -400
            "step_size":  round(bin_edges[1] - bin_edges[0], 2),  # 100
            "num_bins":   10,
            "proportions": proportions,
            "statistics": {
                'mean': round(float(data.mean()), 2),
                'median': round(float(data.median()), 2),
                'std': round(float(data.std()), 2),
                'min': round(float(data.min()), 2),
                'max': round(float(data.max()), 2),
                'percentile_25': round(float(np.percentile(data, 25)), 2),
                'percentile_75': round(float(np.percentile(data, 75)), 2),
                'percentile_95': round(float(np.percentile(data, 95)), 2),
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
