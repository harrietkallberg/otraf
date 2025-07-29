import json
from pathlib import Path
from collections import defaultdict
import csv as csv

class Exporter:
    """
    Exporter for serializing LVLogger data to JSON files.
    Supports per-route exports into separate folders and
    global aggregation (detailed stop index, violations, route index, and time types).
    """
    def __init__(self):
        # Map of route_id (str) to LVLogger instance
        self.logs: dict[str, any] = {}

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
        Export all registered route logs under export_root/, each in its own route_<id>/ folder,
        then build and export global indexes.
        """
        export_root = Path(export_root)
        export_root.mkdir(parents=True, exist_ok=True)

        # 1) Per-route exports
        for rid, log in self.logs.items():
            route_dir = export_root / f"route_{rid}"
            route_dir.mkdir(parents=True, exist_ok=True)

            # Topology & regulatory
            self._dump_safe(log.stop_topology_logs, route_dir / "stop_topology.json")
            self._dump_safe(log.direction_topology_logs, route_dir / "direction_topology.json")
            self._dump_safe(log.regulatory_stops_logs, route_dir / "regulatory_stops.json")

            # Performance logs - Now we include analytics_logs which contains histograms and punctuality
            perf_bundle = {
                "performance_summary": log.performance_logs['metadata']['performance_summary'],
                "analytics_logs": log.performance_logs.get("analytics_logs", {}),  # New combined logs
                "travel_times": log.performance_logs.get("travel_times", {})
            }

            self._dump_safe(perf_bundle, route_dir / "performance_logs.json")

            # Navigation map
            nav = log.navigation_structures.get("routewise_navigation", {})
            self._dump_safe(nav, route_dir / "routewise_navigation.json")

            print(f"✅ Exported route {rid} to {route_dir}")

        # 2) Global indexes (no changes needed here)
        self._export_global_route_index(export_root)
        self._export_global_stop_index(export_root)

        self._export_global_time_types(export_root)
        self._export_global_violations(export_root)
        self._export_global_labels(export_root)

        self._export_global_travel_times(export_root)
        self._export_global_performance_analytics(export_root)

        # 2b) Prepare CSV folder
        csv_dir = export_root / "csv"
        csv_dir.mkdir(exist_ok=True)

        # 3) CSV exports
        self._export_csv_global_travel_times(csv_dir)
        self._export_csv_underperforming_regulatory_stops(csv_dir)
        self._export_csv_mis_tracked_stops(csv_dir)

    def _export_global_route_index(self, export_root: Path):
        """
        Build and export a global route index JSON mapping each route_id
        to its human-readable long and short names.
        """
        route_index = {}
        for rid, log in self.logs.items():
            route_index[rid] = {
                "route_long_name":  getattr(log, "route_long_name", None),
                "route_short_name": getattr(log, "route_short_name", None)
            }
        self._dump_safe(route_index, export_root / "global_route_index.json")
        print(f"🔄 Exported global route index to {export_root / 'global_route_index.json'}")

    def _export_global_stop_index(self, export_root: Path):
        """
        Build and export a detailed global stop index JSON by scanning each
        route's navigation structures. Each stop entry includes:
        - stop_name
        - routes list
        - directions per route
        - label_stats per domain (routes, occurrences)
        - label_keys per domain
        - violation_stats per domain (routes, occurrences)
        - violation_keys per domain
        - performance_keys availability (lists of all seen keys)
        - severity_counts_by_severity: aggregated 1–5 buckets
        """
        domains = ["stop_topology", "direction_topology", "regulatory", "parent_station"]
        stop_index: dict[str, dict] = {}

        # 1️⃣ Gather per-route data
        for rid, log in self.logs.items():
            nav = log.navigation_structures.get("routewise_navigation", {}) or {}
            for direction in nav.get("directions", []):
                did = direction.get("direction_id")
                for stop in direction.get("stops", []):
                    sid = stop.get("stop_id")
                    if sid is None:
                        continue

                    # Initialize entry if first time seeing this stop
                    entry = stop_index.setdefault(sid, {
                        "stop_name":      stop.get("stop_name", "UNKNOWN"),
                        "routes":         set(),
                        "directions":     defaultdict(set),
                        "label_stats":    {d: {"routes": set(), "occurrences": 0} for d in domains},
                        "violation_stats":{d: {"routes": set(), "occurrences": 0} for d in domains},
                        "label_keys":     {d: [] for d in domains},
                        "violation_keys": {d: [] for d in domains},
                        "performance_keys": defaultdict(lambda: {"histograms": [], "punctualities": []}),
                        "severity_counts_by_severity": {}    # ← NEW histogram
                    })

                    # Track which routes & directions this stop appears on
                    entry["routes"].add(rid)
                    entry["directions"][rid].add(did)

                    # Label stats & keys
                    for dom, keys in (stop.get("labels") or {}).items():
                        if dom not in domains:
                            continue
                        for k in (keys or []):
                            entry["label_stats"][dom]["routes"].add(rid)
                            entry["label_stats"][dom]["occurrences"] += 1
                            entry["label_keys"][dom].append(k)

                    # Violation stats & keys
                    for dom, keys in (stop.get("violations") or {}).items():
                        if dom not in domains:
                            continue
                        for k in (keys or []):
                            entry["violation_stats"][dom]["routes"].add(rid)
                            entry["violation_stats"][dom]["occurrences"] += 1
                            entry["violation_keys"][dom].append(k)

                    # Performance availability
                    p_avail = stop.get("performance_availability") or {}
                    for tt, hkey in (p_avail.get("histograms") or {}).items():
                        if hkey:
                            entry["performance_keys"][tt]["histograms"].append(hkey)
                    for tt, pkey in (p_avail.get("punctuality") or {}).items():
                        if pkey:
                            entry["performance_keys"][tt]["punctualities"].append(pkey)

                    # ──────────────────────────────────────────────────────
                    # 2️⃣ Aggregate per-stop severity histograms
                    for sev, cnt in (stop.get("violation_severity_counts") or {}).items():
                        sev_str = str(sev)
                        sc = entry["severity_counts_by_severity"]
                        sc[sev_str] = sc.get(sev_str, 0) + cnt
                    # ──────────────────────────────────────────────────────

        # 3️⃣ Sanitize and export for JSON
        final_index: dict[str, dict] = {}
        for sid, e in stop_index.items():
            final_index[sid] = {
                "stop_name":      e["stop_name"],
                "routes":         sorted(e["routes"]),
                "directions":     {rid: sorted(ds) for rid, ds in e["directions"].items()},
                "label_stats": {
                    dom: {
                        "routes":     sorted(data["routes"]),
                        "occurrences": data["occurrences"]
                    }
                    for dom, data in e["label_stats"].items()
                },
                "label_keys":     {dom: list(keys) for dom, keys in e["label_keys"].items()},
                "violation_stats": {
                    dom: {
                        "routes":     sorted(data["routes"]),
                        "occurrences": data["occurrences"]
                    }
                    for dom, data in e["violation_stats"].items()
                },
                "violation_keys": {dom: list(keys) for dom, keys in e["violation_keys"].items()},
                "performance_keys": {
                    rid: {
                        "histograms":   ks["histograms"],
                        "punctualities": ks["punctualities"]
                    }
                    for rid, ks in e["performance_keys"].items()
                },
                "severity_counts_by_severity": e["severity_counts_by_severity"]  # ← serialized
            }

        # Write out the JSON
        self._dump_safe(final_index, export_root / "global_stop_index.json")
        print(f"🔄 Exported global stop index to {export_root / 'global_stop_index.json'}")

    def _export_global_time_types(self, export_root: Path):
        """
        Build and export a list of all time_types found in performance logs across all routes.
        """
        time_types = set()
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
        """
        Flatten all violation entries across routes into a single list JSON.
        """
        flat = []

        for log in self.logs.values():
            # STOP TOPOLOGY violations
            st_logs = log.stop_topology_logs
            for key in ("parent_station_violations", "stop_id_violations"):
                for entry in st_logs.get(key, {}).values():
                    flat.append(entry)

            # DIRECTION TOPOLOGY violations
            dt_logs = log.direction_topology_logs
            for key in ("direction_violations", "stop_id_violations"):
                for entry in dt_logs.get(key, {}).values():
                    flat.append(entry)

        # Write out the combined list
        self._dump_safe(flat, export_root / "global_violations.json")
        print(f"🔄 Exported global violations to {export_root / 'global_violations.json'}")

    def _export_global_labels(self, export_root: Path):
        """
        Flatten all label entries across routes into a single list JSON.
        """
        flat = []

        for log in self.logs.values():
            # STOP TOPOLOGY labels
            st_logs = log.stop_topology_logs
            for key in ("parent_station_labels", "stop_id_labels"):
                flat.extend(st_logs.get(key, {}).values())

            # DIRECTION TOPOLOGY labels
            dt_logs = log.direction_topology_logs
            for key in ("direction_labels",):
                flat.extend(dt_logs.get(key, {}).values())

            flat.extend(log.regulatory_stops_logs.get("stop_id_regulatory_labels", {}).values())

        # Write out the combined list
        self._dump_safe(flat, export_root / "global_labels.json")
        print(f"🔄 Exported global labels to {export_root / 'global_labels.json'}")

    def _export_global_travel_times(self, export_root: Path):
        """
        Export global aggregated travel times keyed by seg_key.
        Each entry includes:
        - seg_key (entity_key for the segment)
        - from_stop_id, to_stop_id, time_type
        - aggregated mean and sample size
        - per-route breakdowns
        """
        from collections import defaultdict

        grouped = defaultdict(list)

        # 1) Group entries by segment + time_type (and keep their seg_key)
        for log in self.logs.values():
            for seg_key, entry in log.performance_logs.get("travel_times", {}).items():
                key = (
                    entry["from_stop_id"],
                    entry["to_stop_id"],
                    entry["time_type"]
                )
                grouped[key].append((seg_key, entry))

        # 2) Build global entries
        global_dict = {}

        for (from_id, to_id, tt), seg_entries in grouped.items():
            total_samples = sum(e.get("sample_size", 0) for _, e in seg_entries)
            weighted_sum = sum(e["statistics"]["mean"] * e.get("sample_size", 0) for _, e in seg_entries)
            overall_mean = round(weighted_sum / total_samples, 2) if total_samples > 0 else None

            for seg_key, entry in seg_entries:
                global_dict[seg_key] = {
                    "seg_key":       seg_key,
                    "from_stop_id":  from_id,
                    "to_stop_id":    to_id,
                    "time_type":     tt,
                    "aggregated": {
                        "mean":        overall_mean,
                        "sample_size": total_samples
                    },
                    "by_route": [
                        {
                            "route_id":     e["route_id"],
                            "direction_id": e["direction_id"],
                            "mean":         e["statistics"]["mean"],
                            "sample_size":  e.get("sample_size", 0)
                        }
                        for _, e in seg_entries
                    ]
                }

        # 3) Write out as a dictionary keyed by seg_key
        self._dump_safe(global_dict, export_root / "global_travel_times.json")
        print(f"🔄 Exported aggregated travel times to {export_root / 'global_travel_times.json'}")

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

    def _export_csv_global_travel_times(self, csv_dir: Path):
        """
        Export travel times in the legacy flat format:
        from_stop_id, to_stop_id, time_type, aggregated_mean, aggregated_sample_size,
        route_id, direction_id, route_mean, route_sample_size
        """
        import json, csv

        json_path = csv_dir.parent / "global_travel_times.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        csv_path = csv_dir / "global_travel_times.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "from_stop_id", "to_stop_id", "time_type",
                "aggregated_mean", "aggregated_sample_size",
                "route_id", "direction_id", "route_mean", "route_sample_size"
            ])

            for entry in data.values() if isinstance(data, dict) else data:
                agg = entry["aggregated"]
                from_stop_id = entry["from_stop_id"]
                to_stop_id = entry["to_stop_id"]
                time_type = entry["time_type"]

                for r in entry["by_route"]:
                    writer.writerow([
                        from_stop_id,
                        to_stop_id,
                        time_type,
                        agg.get("mean", ""),
                        agg.get("sample_size", ""),
                        r.get("route_id", ""),
                        r.get("direction_id", ""),
                        r.get("mean", ""),
                        r.get("sample_size", "")
                    ])

        print(f"🔄 Exported global_travel_times.csv to {csv_path}")

    def _export_csv_underperforming_regulatory_stops(self, csv_dir: Path, threshold: float = 80.0):
        """
        Export CSV of regulatory stops with on-time % below threshold,
        using global_performance_analytics.json.
        """
        import json, csv

        jp = csv_dir.parent / "global_performance_analytics.json"
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        out_path = csv_dir / "underperforming_regulatory_stops.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "entity_key", "route_id", "direction_id", "stop_id",
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
                        entity_key,
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
                    sid   = e["stop_id"]
                    sname = e["stop_name"]
                    vtype = e["violation_type"]
                    sev   = int(e["severity"])
                    agg.setdefault((sid, sname), []).append((vtype, sev))

        with path.open("w", newline="", encoding="utf-8") as f:
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

    def _dump_safe(self, obj: any, path: Path):
        """
        Serialize an object to JSON safely, creating parent dirs as needed.
        Ensures lists, dicts are sanitized; catches file errors.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error exporting {path}: {e}")
