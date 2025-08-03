import json
from pathlib import Path
from collections import defaultdict
import csv as csv
import itertools
from statistical.lv_logger import LVLogger

class Exporter:
    """
    Exporter for serializing LVLogger data to JSON files.
    Supports per-route exports into separate folders and
    global aggregation (detailed stop index, violations, route index, and time types).
    """
    def __init__(self):
        # Map of route_id (str) to LVLogger instance
        self.logs: dict[str, any] = {}
        self.stop_index = {}

    def add_log(self, logger: any):
        """
        Register an LVLogger instance for a specific route.

        Raises ValueError if the route_id is already registered.
        """
        rid = str(logger.route_id)
        if rid in self.logs:
            raise ValueError(f"Route {rid} already registered for export")
        self.logs[rid] = logger

    def export_all(self, export_root: Path):
        """
        Export all
        """
        export_root = Path(export_root)
        export_root.mkdir(parents=True, exist_ok=True)

        # 2) Global indexes
        self._export_global_route_index(export_root)
        self._export_global_stop_index(export_root)

        # 1) Per-route exports
        for rid, log in self.logs.items():
            route_dir = export_root / 'routes'  # ✅ Create Path object
            route_dir.mkdir(parents=True, exist_ok=True)  # ✅ Create directory
            
            try:
                nav = self.build_routewise_nav(rid)
                self._dump_safe(nav, route_dir / f'{rid}_route_nav.json')  # ✅ Path object
                print(f"✅ Exported route {rid} to {route_dir}")
            except Exception as e:
                print(f"❌ Error building routewise nav for {rid}: {e}")
        
        # 2) Per-stop exports  
        for pid, info in self.stop_index.items():
            stop_dir = export_root / 'stops'   # ✅ Create Path object
            stop_dir.mkdir(parents=True, exist_ok=True)  # ✅ Create directory
            
            try:
                nav = self.build_stopwise_nav(pid)
                self._dump_safe(nav, stop_dir / f'{pid}_stop_nav.json')  # ✅ Path object
                print(f"✅ Exported stop {pid} to {stop_dir}")
            except Exception as e:
                print(f"❌ Error building stopwise nav for {pid}: {e}")
        
        self._export_global_time_types(export_root)
        self._export_global_violations(export_root)
        self._export_global_labels(export_root)

        self._export_global_travel_times(export_root)
        self._export_global_performance_analytics(export_root)

        # 2b) Prepare CSV folder
        csv_dir = export_root / "csv"
        csv_dir.mkdir(exist_ok=True)

        # 3) CSV exports
        self._export_csv_travel_times(csv_dir)
        self._export_csv_underperforming_regulatory_stops(csv_dir)
        self._export_csv_mis_tracked_stops(csv_dir)

    # Specific navigation files
    def build_routewise_nav(self, rid):
        """
        Build the routewise navigation structure for a given route ID.
        This structure will include metadata, violation counts, severity, and stop-level data.
        """
        # Fetch the log for the given route ID
        log = self.logs[rid]

        # Initialize the navigation structure for the route
        nav = {
            "route_id": rid,
            "route_long_name": str(log.route_long_name),
            "route_short_name": str(log.route_short_name),
            "route_summary": {},
            "performance_summary": log.performance_logs['metadata']['performance_summary'],
            "directions": {}  # Holds direction-level data for direction 0 and 1
        }
        # Get canonical patterns for this route
        canonical_patterns = log.direction_topology_logs['metadata'].get('canonical_patterns', {})
            
        direction_summary_list = []
        for direction_id, pattern_stops in canonical_patterns.items():
            direction_entry = {
                "direction_id": str(direction_id),
                "direction_summary": {},
                "stops_in_direction": {}  # canonical stop_id pattern for this direction
            }
            # 8) Collect stop-level data for each direction
            pos = 1
            stop_id_summary_list = []
            for stop_id in pattern_stops:
                stop_data = self._get_stop_id_data(rid, direction_id, stop_id)
                direction_entry["stops_in_direction"][pos] = stop_data
                pos += 1
                stop_id_summary_list.append(stop_data['stop_id_summary'])

            direction_summary_dict = self._merge_and_aggregate_dicts(stop_id_summary_list)
            direction_entry['direction_summary'] = direction_summary_dict

            # 9) Update the direction entry in the navigation structure
            nav["directions"][str(direction_id)] = direction_entry
            direction_summary_list.append(direction_summary_dict)
        
        route_summary_dict = self._merge_and_aggregate_dicts(direction_summary_list)
        nav['route_summary'] = route_summary_dict

        return nav

    def build_stopwise_nav(self, pid):
        """Fixed version of your stopwise navigation builder."""
        
        stop_info = self.stop_index[pid]

        nav = {
            "parent_station": pid,
            "stop_name": stop_info['stop_name'],
            "stop_ids": stop_info['stop_ids'],
            "on_routes": stop_info['routes'],
            "stop_summary": {},
            "performance_summary":{},
            "routes": {}
        }
        stop_summary_list = []
        performance_summary_list = []
        for route_id in stop_info['routes']:
            route_log = self.logs[route_id]  # Get the correct log for this route

            nav['routes'][route_id] = {
                'route_id': str(route_log.route_id),
                'route_long_name': str(route_log.route_long_name),
                'route_short_name': str(route_log.route_short_name),
                'route_summary': {},
                'performace_summary': route_log.performance_logs['metadata']['performance_summary'],
                'directions': {}
                }
           
            # Get canonical patterns for this route
            canonical_patterns = route_log.direction_topology_logs['metadata'].get('canonical_patterns', {})
            
            direction_summary_list = []
            for direction_id, pattern_stops in canonical_patterns.items():
                # Check if any of our stop_ids appear in this direction
                our_stops_in_direction = [sid for sid in stop_info['stop_ids'] if sid in pattern_stops]
                
                if our_stops_in_direction:  # Only include direction if it serves our stop
                    direction_entry = {
                        "direction_id": direction_id,
                        "direction_summary": {},
                        "stops_in_direction": {}
                    }
                    
                    stop_id_summary_list = []
                    for stop_id in our_stops_in_direction:
                        # Find canonical position
                        canonical_position = pattern_stops.index(stop_id) + 1 if stop_id in pattern_stops else None
                        stop_data = self._get_stop_id_data(route_id, direction_id, stop_id)
                        if "canonical_position" not in direction_entry["stops_in_direction"]:
                            direction_entry["stops_in_direction"]["canonical_position"] = {}
                        direction_entry["stops_in_direction"]["canonical_position"][str(canonical_position)] = stop_data
                        stop_id_summary_list.append(stop_data['stop_id_summary'])
                    
                    direction_summary_dict = self._merge_and_aggregate_dicts(stop_id_summary_list)
                    direction_entry['direction_summary'] = direction_summary_dict
                    # Add direction to route
                    nav["routes"][route_id]["directions"][direction_id] = direction_entry
                    direction_summary_list.append(direction_summary_dict)
            
            route_summary_dict = self._merge_and_aggregate_dicts(direction_summary_list)
            nav['routes'][route_id]['route_summary'] = route_summary_dict
            stop_summary_list.append(route_summary_dict)
            performance_summary_list.append(route_log.performance_logs['metadata']['performance_summary'])

        nav['stop_summary'] = self._merge_and_aggregate_dicts(stop_summary_list)
        nav['performance_summary'] = self._merge_and_average_dicts(performance_summary_list)

        return nav

    # Global index files
    def _export_global_route_index(self, export_root: Path):
        route_index = {}
        for rid, log in self.logs.items():
            stops = set()
            stop_ids = set()
            for label_entry in log.stop_topology_logs.get("parent_station_labels", {}).values():
                parent_station = str(label_entry.get("parent_station", ""))
                if parent_station:
                    stops.add(parent_station)
                if "stop_ids" in label_entry:
                    stop_ids.update(str(sid) for sid in label_entry["stop_ids"])

            route_index[rid] = {
                "route_id": str(rid),
                "route_long_name": getattr(log, "route_long_name", None),
                "route_short_name": getattr(log, "route_short_name", None),
                "stops": sorted(stops),
                "route_nav": f"routes/{rid}_route_nav.json",
                "summary": {
                    "total_stops": len(stops),      # ← Fixed: removed sorted()
                    "total_stop_ids": len(stop_ids) # ← Fixed: removed sorted()
                }
            }

        self._dump_safe(route_index, export_root / "global_route_index.json")
        print(f"🔄 Exported global route index to {export_root / 'global_route_index.json'}")

    def _export_global_stop_index(self, export_root: Path):
        """
        Build and export a global stop index JSON mapping each parent_station
        to its basic info and which routes serve it.
        
        Since all parent stations are classified in stop_topology_logs with labels,
        we can use that as the primary (and sufficient) source.
        """
        stop_index = {}
        
        for rid, log in self.logs.items():
            # Primary source: parent_station_labels from stop topology
            for label_entry in log.stop_topology_logs.get("parent_station_labels", {}).values():
                print(label_entry)
                parent_station = str(label_entry.get("parent_station", ""))
                
                if parent_station and parent_station != "UNKNOWN":
                    if parent_station not in stop_index:
                        stop_index[parent_station] = {
                            "parent_station": parent_station,
                            "stop_name": label_entry.get("stop_name", "UNKNOWN"),
                            "stop_ids": set(),
                            "routes": set(),
                            "stop_nav": f"stops/{self._sanitize_filename(parent_station)}_stop_nav.json"
                        }
                    
                    # Add route and stop_ids from this label
                    stop_index[parent_station]["routes"].add(rid)
                    if "stop_ids" in label_entry:
                        stop_index[parent_station]["stop_ids"].update(
                            str(sid) for sid in label_entry["stop_ids"] 
                        )
        # To use in the stopwise nav structures
        self.stop_index = stop_index

        # Convert sets to sorted lists for JSON serialization
        final_stop_index = {}
        for parent_station, data in stop_index.items():
            final_stop_index[parent_station] = {
                "parent_station": data["parent_station"],
                "stop_name": data["stop_name"],
                "stop_ids": sorted(data["stop_ids"]) if data["stop_ids"] else [],
                "routes": sorted(data["routes"]) if data["routes"] else [],
                "stop_nav": data["stop_nav"],  # ← Added this line
                "summary": {
                    "total_routes": len(data["routes"]),
                    "total_stop_ids": len(data["stop_ids"])
                }
            }

        self._dump_safe(final_stop_index, export_root / "global_stop_index.json")
        print(f"🔄 Exported global stop index with {len(final_stop_index)} parent stations to {export_root / 'global_stop_index.json'}")

    # Global log files
    def _export_global_time_types(self, export_root: Path):
        """
        Build and export a list of all time_types found in performance logs across all routes.
        """
        time_types = set()
        time_types.add("all")
        for log in self.logs.values():
            
            for entry in log.performance_logs.get("analytics_logs", {}).values():
                tt = entry.get("time_type")
                if tt:
                    time_types.add(tt)

            # Travel times:        
            for entry in log.performance_logs.get("travel_times",{}).values():
                if tt := entry.get("time_type"):
                    time_types.add(tt)

        tt_list = sorted(time_types)
        self._dump_safe(tt_list, export_root / "global_time_types.json")
        print(f"🔄 Exported global time types to {export_root / 'global_time_types.json'}")

    def _export_global_violations(self, export_root: Path):
        """Export violations as dict with keys for direct lookup."""
        violations_dict = {}
        
        for log in self.logs.values():
            # Preserve original keys from logs
            for domain_violations in [
                log.stop_topology_logs.get("parent_station_violations", {}),
                log.stop_topology_logs.get("stop_id_violations", {}),
                log.direction_topology_logs.get("direction_violations", {}),
                log.direction_topology_logs.get("stop_id_violations", {})
            ]:
                violations_dict.update(domain_violations)  # Merge with keys preserved
        
        self._dump_safe(violations_dict, export_root / "global_violations.json")

    def _export_global_labels(self, export_root: Path):
        """Export labels as dict with keys for direct lookup."""
        labels_dict = {}
        
        for log in self.logs.values():
            # Preserve original keys from logs
            for domain_labels in [
                log.stop_topology_logs.get("parent_station_labels", {}),
                log.stop_topology_logs.get("stop_id_labels", {}),
                log.direction_topology_logs.get("direction_labels", {}),
                log.direction_topology_logs.get("stop_id_labels", {}),
                log.regulatory_stops_logs.get("stop_id_regulatory_labels", {})
            ]:
                labels_dict.update(domain_labels)  # Merge with keys preserved
        
        self._dump_safe(labels_dict, export_root / "global_labels.json")

    def _export_global_performance_analytics(self, export_root: Path):
        """
        Export global performance analytics as a dictionary keyed by entity_key.
        Each entry mirrors the routewise structure:
        {
        "entity_key": {
            "route_id": ...,
            "direction_id": ...,
            "stop_id": ...,
            "stop_name": ...,
            "time_type": ...,
            "analytics": {
            "total_delay_histogram": {...},
            "incremental_delay_histogram": {...},
            "punctuality": {...},
            "is_regulatory_stop": ...
            }
        }
        }
        """
        analytics_dict = {}

        for rid, log in self.logs.items():
            entries = log.performance_logs.get("analytics_logs", {})
            for entity_key, entry in entries.items():
                analytics_dict[entity_key] = {
                    "route_id": rid,
                    "direction_id": entry.get("direction_id"),
                    "stop_id": entry.get("stop_id"),
                    "stop_name": entry.get("stop_name"),
                    "time_type": entry.get("time_type"),
                    "analytics": entry.get("analytics", {})
                }

        self._dump_safe(analytics_dict, export_root / "global_performance_analytics.json")
        print(f"🔄 Exported global performance analytics to {export_root / 'global_performance_analytics.json'}")

    def _export_global_travel_times(self, export_root: Path):
        """
        Export global aggregated travel times as an array.
        Each entry includes:
        - from_stop_id, to_stop_id, time_type
        - aggregated mean and sample size across all routes
        - per-route breakdowns
        """
        from collections import defaultdict

        grouped = defaultdict(list)

        # 1) Group entries by segment + time_type (regardless of route)
        for log in self.logs.values():
            for seg_key, entry in log.performance_logs.get("travel_times", {}).items():
                key = (
                    entry["from_stop_id"],
                    entry["from_stop_name"],
                    entry["to_stop_id"],
                    entry["to_stop_name"],
                    entry["time_type"]
                )
                grouped[key].append(entry)

        # 2) Build global entries - one per unique segment
        global_array = []

        for (from_id, from_name, to_id, to_name, tt), entries in grouped.items():
            # Calculate aggregated statistics across all routes for this segment
            total_samples = sum(e.get("sample_size", 0) for e in entries)
            weighted_sum = sum(e["statistics"]["mean"] * e.get("sample_size", 0) for e in entries)
            overall_mean = round(weighted_sum / total_samples, 2) if total_samples > 0 else None

            global_array.append({
                "from_stop_id": from_id,
                "from_stop_name": from_name,
                "to_stop_id": to_id,
                "to_stop_name": to_name,
                "time_type": tt,
                "aggregated": {
                    "mean": overall_mean,
                    "sample_size": total_samples
                },
                "by_route": [
                    {
                        "route_id": e["route_id"],
                        "direction_id": e["direction_id"],
                        "mean": e["statistics"]["mean"],
                        "sample_size": e.get("sample_size", 0)
                    }
                    for e in entries
                ]
            })

        # 3) Create 'all' time_type entries by aggregating across all time types for each segment
        segment_all_data = defaultdict(list)
        
        # Group by segment (ignoring time_type) to create 'all' entries
        for (from_id, from_name, to_id, to_name, tt), entries in grouped.items():
            segment_key = (from_id, from_name, to_id, to_name)
            segment_all_data[segment_key].extend(entries)
        
        # Add 'all' entries to global_array
        for (from_id, from_name, to_id, to_name), all_entries in segment_all_data.items():
            # First, aggregate by route+direction across time types
            route_aggregates = defaultdict(lambda: {"samples": 0, "weighted_sum": 0, "direction_id": None})
            
            for entry in all_entries:
                route_key = (entry["route_id"], entry["direction_id"])
                samples = entry.get("sample_size", 0)
                mean = entry["statistics"]["mean"]
                
                route_aggregates[route_key]["samples"] += samples
                route_aggregates[route_key]["weighted_sum"] += mean * samples
                route_aggregates[route_key]["direction_id"] = entry["direction_id"]
            
            # Create aggregated by_route entries
            aggregated_by_route = []
            for (route_id, direction_id), agg_data in route_aggregates.items():
                if agg_data["samples"] > 0:
                    route_mean = round(agg_data["weighted_sum"] / agg_data["samples"], 2)
                    aggregated_by_route.append({
                        "route_id": route_id,
                        "direction_id": direction_id,
                        "mean": route_mean,
                        "sample_size": agg_data["samples"]
                    })
            
            # Calculate overall aggregated statistics across all routes
            total_samples = sum(r["sample_size"] for r in aggregated_by_route)
            weighted_sum = sum(r["mean"] * r["sample_size"] for r in aggregated_by_route)
            overall_mean = round(weighted_sum / total_samples, 2) if total_samples > 0 else None

            global_array.append({
                "from_stop_id": from_id,
                "from_stop_name": from_name,
                "to_stop_id": to_id,
                "to_stop_name": to_name,
                "time_type": "all",
                "aggregated": {
                    "mean": overall_mean,
                    "sample_size": total_samples
                },
                "by_route": aggregated_by_route
            })

        # 4) Write out as an array
        self._dump_safe(global_array, export_root / "global_travel_times.json")
        print(f"🔄 Exported aggregated travel times to {export_root / 'global_travel_times.json'}")

    # CSV Exporter
    def _export_csv_travel_times(self, csv_dir: Path):
        """
        Export travel times in the legacy flat format:
        from_stop_id, to_stop_id, time_type, aggregated_mean, aggregated_sample_size,
        route_id, direction_id, route_mean, route_sample_size
        """
        json_path = csv_dir.parent / "global_travel_times.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        csv_path = csv_dir / "travel_times.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "from_stop_id", "from_stop_name", "to_stop_id", "to_stop_name", "time_type",
                "aggregated_mean", "aggregated_sample_size",
                "route_id", "direction_id", "route_mean", "route_sample_size"
            ])

            # Handle both old dict format and new array format
            entries = data if isinstance(data, list) else data.values()
            
            for entry in entries:
                agg = entry["aggregated"]
                from_stop_id = entry["from_stop_id"]
                from_stop_name = entry['from_stop_name']
                to_stop_id = entry["to_stop_id"]
                to_stop_name = entry['to_stop_name']
                time_type = entry["time_type"]

                for r in entry["by_route"]:
                    writer.writerow([
                        from_stop_id,
                        from_stop_name,
                        to_stop_id,
                        to_stop_name,
                        time_type,
                        agg.get("mean", ""),
                        agg.get("sample_size", ""),
                        r.get("route_id", ""),
                        r.get("direction_id", ""),
                        r.get("mean", ""),
                        r.get("sample_size", "")
                    ])

        print(f"🔄 Exported travel_times.csv to {csv_path}")
   
    def _export_csv_underperforming_regulatory_stops(self, csv_dir: Path, threshold: float = 80.0):
        """
        Export CSV of regulatory stops with on-time % below threshold,
        using global_performance_analytics.json.
        """
        jp = csv_dir.parent / "global_performance_analytics.json"
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        out_path = csv_dir / "underperforming_regulatory_stops.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "route_id", "direction_id", "stop_id",
                "stop_name", "time_type", "on_time_pct", "sample_size"
            ])

            for entity_key, e in data.items():
                analytics = e.get("analytics", {})
                punctuality = analytics.get("punctuality", {})
                is_reg = analytics.get("is_regulatory_stop", False)
                pct = punctuality.get("punctuality_distribution", {}).get("percentages", {})
                on_time = pct.get("on_time", 0.0)
                sample_size = punctuality.get("sample_size", 0)

                if is_reg and on_time < threshold:
                    writer.writerow([
                        e.get("route_id", ""),
                        e.get("direction_id", ""),
                        e.get("stop_id", ""),
                        e.get("stop_name", ""),
                        e.get("time_type", ""),
                        on_time,
                        sample_size
                    ])

        print(f"🔄 Exported underperforming regulatory stops to {out_path}")

    def _export_csv_mis_tracked_stops(self, csv_dir: Path):
        path = csv_dir / "mis_tracked_stops.csv"
        # aggregate per stop_id/stop_name → list of (violation_type, severity)
        agg: dict[tuple[str,str], list[tuple[str,int]]] = {}
        for log in self.logs.values():
            for dom in (log.stop_topology_logs, log.direction_topology_logs):
                for e in dom.get("stop_id_violations", {}).values():
                    sid = e["stop_id"]
                    sname = e["stop_name"]
                    vtype = e["violation_type"]
                    sev = int(e["severity"])
                    agg.setdefault((sid, sname), []).append((vtype, sev))

        with path.open("w", newline="", encoding="utf-8") as f:  # Writing rows to CSV with UTF-8 encoding
            writer = csv.writer(f)
            writer.writerow([
                "stop_id",
                "stop_name",
                "violation_count",
                "max_severity",
                "top_violation",
                "has_other_issues"
            ])

            for (sid, sname), violations in agg.items():
                # total number of flags
                violation_count = len(violations)
                # find the entry driving max severity
                top_vtype, max_sev = max(violations, key=lambda vs: vs[1])
                # check if there are other distinct violation_types
                distinct_types = {vtype for vtype, _ in violations}
                has_others = len(distinct_types) > 1

                writer.writerow([
                    sid,
                    sname,
                    violation_count,
                    max_sev,
                    top_vtype,
                    has_others,
                ])

        print(f"🔄 Exported mis_tracked_stops.csv to {path}")

    # Helpers
    def _dump_safe(self, obj: any, path: Path):
        """
        Serialize an object to JSON safely with NumPy support, creating parent dirs as needed.
        Ensures lists, dicts are sanitized; catches file errors.
        """
        import numpy as np
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif hasattr(obj, 'item'):  # Handle numpy scalars
                    return obj.item()
                return super().default(obj)
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        except Exception as e:
            print(f"Error exporting {path}: {e}")

    def _get_stop_id_data(self, rid, direction_id, stop_id):
        """
        Collect stop-specific data for a given stop in the route.
        This includes stop name, parent station, labels, violations, and performance metrics.
        """
        
        log = self.logs[rid]

        st_dict = {'labels_by_type':
                    { 'parent_station': len(log.get_all_keys_regex('stop_topology', f'_{rid}_', 'parent_station', f'_{self._get_parent_station(rid, stop_id)}', match_all = True, log_type = 'parent_station_labels', exclude=f'{stop_id}')),
                      'stop_id': len(log.get_all_keys_regex('stop_topology', f'_{rid}_', 'stop_id', f'_{stop_id}','parent_station', f'_{self._get_parent_station(rid, stop_id)}', match_all=True, log_type='stop_id_labels'))},
                   'violations_by_type':
                    { 'parent_station': len(log.get_all_keys_regex('stop_topology', f'_{rid}_', 'parent_station', f'_{self._get_parent_station(rid, stop_id)}', match_all = True, log_type = 'parent_station_labels', exclude=f'{stop_id}' )),
                      'stop_id': len(log.get_all_keys_regex('stop_topology', f'_{rid}_', 'stop_id', f'_{stop_id}','parent_station', f'_{self._get_parent_station(rid, stop_id)}', match_all=True, log_type='stop_id_violations'))}}
        
        dt_dict = {'labels_by_type': 
                   {'direction_id': len(log.get_all_keys_regex('direction_topology', f'_{rid}_', 'direction_id', f'_{direction_id}', match_all=True, log_type='direction_labels', exclude=f'{stop_id}')),
                    'stop_id': len(log.get_all_keys_regex('direction_topology', f'_{rid}_', 'stop_id', f'_{stop_id}', 'direction_id', f'_{direction_id}', match_all=True, log_type='stop_id_labels'))},
                   'violations_by_type':
                   {'direction_id' : len(log.get_all_keys_regex('direction_topology', f'_{rid}_', 'direction_id', f'_{direction_id}', match_all=True, log_type='direction_labels', exclude=f'{stop_id}')),
                    'stop_id': len(log.get_all_keys_regex('direction_topology', f'_{rid}_', 'stop_id', f'_{stop_id}', 'direction_id', f'_{direction_id}', match_all=True, log_type='stop_id_violations'))}}
        
        rs_dict = {"regulatory_stop_ids": len(log.get_all_keys_regex('regulatory_stops', f'_{rid}_', 'stop_id', f'_{stop_id}', 'direction_id', f'_{direction_id}', match_all=True, log_type='stop_id_regulatory_labels'))}
        p_dict = {"available_performace_analytics": len(log.get_all_keys_regex('performance', f'_{rid}_', 'stop_id', f'_{stop_id}','direction_id', f'_{direction_id}_', match_all = True, log_type = 'analytics_logs'))}
        
        stop_summary_dict = {'stop_topology' : st_dict,
                        'direction_topology': dt_dict,
                        'regulatory_stops' : rs_dict,
                        'performance': p_dict}

        stop_id_data = {
            "stop_id": stop_id,
            "stop_name": self._get_stop_name(rid, stop_id),
            "parent_station":self._get_parent_station(rid, stop_id),

            "direction_id_label_key": log.get_all_keys_regex('direction_topology', f'_{rid}_', 'direction_id', f'_{direction_id}', match_all=True, log_type='direction_labels', exclude=f'{stop_id}'),
            "direction_id_violation_key": log.get_all_keys_regex('direction_topology', f'_{rid}_', 'direction_id', f'_{direction_id}', match_all=True, log_type='direction_violations', exclude=f'{stop_id}'),

            "parent_station_label_key": log.get_all_keys_regex('stop_topology', f'_{rid}_', 'parent_station', f'_{self._get_parent_station(rid, stop_id)}', match_all = True, log_type = 'parent_station_labels', exclude=f'{stop_id}'),
            "parent_station_violation_key": log.get_all_keys_regex('stop_topology', f'_{rid}_', 'parent_station', f'_{self._get_parent_station(rid, stop_id)}', match_all = True, log_type = 'parent_station_labels', exclude=f'{stop_id}' ),
            
            "stop_id_label_keys": 
                log.get_all_keys_regex('stop_topology', f'_{rid}_', 'stop_id', f'_{stop_id}','parent_station', f'_{self._get_parent_station(rid, stop_id)}', match_all=True, log_type='stop_id_labels') +
                log.get_all_keys_regex('direction_topology', f'_{rid}_', 'stop_id', f'_{stop_id}', 'direction_id', f'_{direction_id}', match_all=True, log_type='stop_id_labels') +
                log.get_all_keys_regex('regulatory_stops', f'_{rid}_', 'stop_id', f'_{stop_id}', 'direction_id', f'_{direction_id}', match_all=True, log_type='stop_id_regulatory_labels'),
            
            "stop_id_violation_keys": 
                log.get_all_keys_regex('stop_topology', f'_{rid}_', 'stop_id', f'_{stop_id}','parent_station', f'_{self._get_parent_station(rid, stop_id)}', match_all=True, log_type='stop_id_violations') +
                log.get_all_keys_regex('direction_topology', f'_{rid}_', 'stop_id', f'_{stop_id}', 'direction_id', f'_{direction_id}', match_all=True, log_type='stop_id_violations'),
            
            "stop_id_performance_keys": log.get_all_keys_regex('performance', f'_{rid}_', 'stop_id', f'_{stop_id}','direction_id', f'_{direction_id}_', match_all = True, log_type = 'analytics_logs'),
            "stop_id_summary": stop_summary_dict
        }
        return stop_id_data
    
    def _get_stop_name(self, rid, stop_id):
        """
        Retrieve the name of the stop by finding the stop_id in the keys.
        """
        log = self.logs[rid]
        
        # Use get_all_keys_regex to find keys containing this stop_id in stop_id_labels
        matching_keys = log.get_all_keys_regex('stop_topology', 'stop_id', f'{stop_id}', log_type='stop_id_labels')
        
        # If we found matching keys, get the stop_name from the first one
        if matching_keys:
            first_key = matching_keys[0]
            entry = log.stop_topology_logs.get("stop_id_labels", {}).get(first_key, {})
            return entry.get("stop_name", "UNKNOWN")
        
        return "UNKNOWN"

    def _get_parent_station(self, rid, stop_id):
        """
        Retrieve the parent station by finding the stop_id in the keys.
        """
        log = self.logs[rid]
        
        # Use get_all_keys_regex to find keys containing this stop_id in stop_id_labels
        matching_keys = log.get_all_keys_regex('stop_topology', 'stop_id_', f'{stop_id}', log_type='stop_id_labels')
        
        # If we found matching keys, get the parent_station from the first one
        if matching_keys:
            first_key = matching_keys[0]
            entry = log.stop_topology_logs.get("stop_id_labels", {}).get(first_key, {})
            return entry.get("parent_station", "UNKNOWN")
        
        return "UNKNOWN"
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', str(name))
        sanitized = sanitized.replace(' ', '_')
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized or "unknown"

    def _merge_and_aggregate_dicts(self, dict_list):
        """
        Merge multiple dictionaries with the same structure and aggregate numerical values at the finest level.
        
        Args:
            dict_list: List of dictionaries to merge and aggregate
            
        Returns:
            Dictionary with aggregated values
        """
        if not dict_list:
            return {}
        
        def recursive_merge(target, source):
            """Recursively merge source into target, aggregating numbers"""
            for key, value in source.items():
                if key not in target:
                    # If key doesn't exist in target, copy the entire value
                    if isinstance(value, dict):
                        target[key] = {}
                        recursive_merge(target[key], value)
                    else:
                        target[key] = value
                else:
                    # Key exists in target
                    if isinstance(value, dict) and isinstance(target[key], dict):
                        # Both are dicts, merge recursively
                        recursive_merge(target[key], value)
                    elif isinstance(value, (int, float)) and isinstance(target[key], (int, float)):
                        # Both are numbers, aggregate
                        target[key] += value
                    else:
                        # Different types or non-numeric, keep the new value (or handle as needed)
                        target[key] = value
        
        # Start with empty result
        result = {}
        
        # Merge each dictionary
        for dict_item in dict_list:
            recursive_merge(result, dict_item)
        
        return result

    def _merge_and_average_dicts(self, dict_list):
        """
        Merge multiple dictionaries with the same structure and average numerical values.
        Optimized for flat dictionaries like performance_summary.
        
        Args:
            dict_list: List of dictionaries to merge and average
            
        Returns:
            Dictionary with averaged values
        """
        if not dict_list:
            return {}
        
        # Track sums and counts for each key
        sums = {}
        counts = {}
        
        # Accumulate values from all dictionaries
        for dict_item in dict_list:
            for key, value in dict_item.items():
                if isinstance(value, (int, float)):
                    if key not in sums:
                        sums[key] = 0
                        counts[key] = 0
                    sums[key] += value
                    counts[key] += 1
                else:
                    # For non-numeric values, keep the last value
                    sums[key] = value
                    counts[key] = None
        
        # Calculate averages
        result = {}
        for key in sums:
            if counts[key] is not None and counts[key] > 0:
                result[key] = sums[key] / counts[key]
            else:
                result[key] = sums[key]
        
        return result