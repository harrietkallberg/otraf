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
            self._dump_safe(log.stop_topology_logs,      route_dir / "stop_topology.json")
            self._dump_safe(log.direction_topology_logs, route_dir / "direction_topology.json")
            self._dump_safe(log.regulatory_stops_logs,   route_dir / "regulatory_stops.json")

            # Performance logs
            perf_bundle = {
            "performance_summary": log.performance_logs['metadata']['performance_summary'],
            "histograms_stops":      log.performance_logs.get("histograms_stops", {}),
            "punctuality_barcharts": log.performance_logs.get("punctuality_barcharts", {}),
            "travel_times":          log.performance_logs.get("travel_times", {})
            }

            self._dump_safe(perf_bundle, route_dir / "performance_logs.json")

            # Navigation map
            nav = log.navigation_structures.get("routewise_navigation", {})
            self._dump_safe(nav, route_dir / "routewise_navigation.json")

            print(f"✅ Exported route {rid} to {route_dir}")

        # 2) Global indexes
        self._export_global_route_index(export_root)
        self._export_global_stop_index(export_root)
        self._export_global_time_types(export_root)
        self._export_global_violations(export_root)
        self._export_global_labels(export_root)
        self._export_global_travel_times(export_root)
        self._export_global_histograms(export_root)
        self._export_global_punctuality(export_root)
        self._export_csv_underperforming_regulatory_stops(export_root)
        self._export_csv_mis_tracked_stops(export_root)

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
            # Histograms:
            for entry in (log.performance_logs.get("histograms_stops", {}).values()):
                tt = entry.get("time_type")
                if tt:
                    time_types.add(tt)
            # Punctuality:
            for entry in (log.performance_logs.get("punctuality_barcharts", {}).values()):
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
        Flatten all per-route, per-direction segment travel-time summaries into
        a single JSON where each entry corresponds to either a specific time_type
        or the aggregated 'all' time_type for that segment.
        """
        import json
        from collections import defaultdict

        # 1) Build bucketed entries as before
        agg: dict[tuple, dict] = {}
        for rid, log in self.logs.items():
            for entry in log.performance_logs.get("travel_times", {}).values():
                frm_id   = entry["from_stop_id"]
                frm_name = entry["from_stop_name"]
                to_id    = entry["to_stop_id"]
                to_name  = entry["to_stop_name"]
                tt       = entry["time_type"]
                mean     = entry["statistics"]["mean"]
                n        = entry["statistics"].get("sample_size", 0)
                key      = (frm_id, to_id, tt)

                slot = agg.setdefault(key, {
                    "from_stop_id":   frm_id,
                    "from_stop_name": frm_name,
                    "to_stop_id":     to_id,
                    "to_stop_name":   to_name,
                    "time_type":      tt,
                    "aggregated": {
                        "weighted_sum": 0.0,
                        "sample_size":  0
                    },
                    "by_route": []
                })

                slot["aggregated"]["weighted_sum"] += mean * n
                slot["aggregated"]["sample_size"]  += n
                slot["by_route"].append({
                    "route_id":     rid,
                    "direction_id": entry["direction_id"],
                    "mean":         mean,
                    "sample_size":  n
                })

        # 2) Finalize the bucketed means
        output = []
        for (frm_id, to_id, tt), data in agg.items():
            total_n = data["aggregated"]["sample_size"]
            wsum    = data["aggregated"]["weighted_sum"]
            data["aggregated"] = {
                "mean":        (wsum / total_n) if total_n else None,
                "sample_size": total_n
            }
            output.append(data)

        # 3) Compute the 'all' aggregation per segment (frm_id, to_id)
        all_buckets = defaultdict(lambda: {
            "weighted_sum": 0.0,
            "sample_size":  0,
            "by_route":     defaultdict(lambda: {"weighted_sum": 0.0, "sample_size": 0})
        })

        # accumulate across every time_type entry we just finalized
        for rec in output:
            mean = rec["aggregated"]["mean"]
            n    = rec["aggregated"]["sample_size"]
            # skip empty buckets
            if mean is None or n == 0:
                continue

            key     = (rec["from_stop_id"], rec["to_stop_id"])
            agg_all = all_buckets[key]
            agg_all["weighted_sum"] += mean * n
            agg_all["sample_size"]  += n

            # also accumulate per-route into this 'all' bucket
            for br in rec["by_route"]:
                rn = br["sample_size"]
                rm = br["mean"]
                if rn and rm is not None:
                    pr = agg_all["by_route"][(br["route_id"], br["direction_id"])]
                    pr["weighted_sum"] += rm * rn
                    pr["sample_size"]  += rn

        # append one 'all' entry per segment
        for (frm_id, to_id), agg_all in all_buckets.items():
            total_n = agg_all["sample_size"]
            wsum    = agg_all["weighted_sum"]
            # find names from any existing bucket
            first = next(o for o in output if (o["from_stop_id"], o["to_stop_id"]) == (frm_id, to_id))
            slot = {
                "from_stop_id":   frm_id,
                "from_stop_name": first["from_stop_name"],
                "to_stop_id":     to_id,
                "to_stop_name":   first["to_stop_name"],
                "time_type":      "all",
                "aggregated": {
                    "mean":        (wsum / total_n) if total_n else None,
                    "sample_size": total_n
                },
                "by_route": []
            }
            # build per-route entries
            for (rid, did), pr in agg_all["by_route"].items():
                rn = pr["sample_size"]
                rm = (pr["weighted_sum"] / rn) if rn else None
                slot["by_route"].append({
                    "route_id":     rid,
                    "direction_id": did,
                    "mean":         rm,
                    "sample_size":  rn
                })
            output.append(slot)

        # 4) Write out the JSON
        self._dump_safe(output, export_root / "global_travel_times.json")
        print(f"🔄 Exported global travel‐times (with 'all' aggregated entries) to {export_root / 'global_travel_times.json'}")

    def _export_global_histograms(self, export_root: Path):
        """
        Flatten all per-route stop histograms into one file with:
          - stop_id, stop_name
          - route_id, direction_id
          - total_delay_histogram, incremental_delay_histogram
        """
        flat: list[dict] = []
        for rid, log in self.logs.items():
            for entry in log.performance_logs.get("histograms_stops", {}).values():
                flat.append({
                    "route_id":                    rid,
                    "direction_id":                entry["direction_id"],
                    "stop_id":                     entry["stop_id"],
                    "stop_name":                   entry["stop_name"],  # :contentReference[oaicite:2]{index=2}
                    "time_type":                   entry["time_type"],
                    "total_delay_histogram":       entry.get("total_delay_histogram"),
                    "incremental_delay_histogram": entry.get("incremental_delay_histogram"),
                })

        self._dump_safe(flat, export_root / "global_histograms_stops.json")
        print(f"🔄 Exported global histograms to {export_root / 'global_histograms_stops.json'}")

    def _export_global_punctuality(self, export_root: Path):
        """
        Flatten per-time_type entries and then add an "all" aggregation,
        carrying through an is_regulatory_stop flag if *any* underlying entry is regulatory.
        """
        from collections import defaultdict

        flat = []
        # 1) Flatten time_type–specific entries
        for rid, log in self.logs.items():
            for entry in log.performance_logs.get("punctuality_barcharts", {}).values():
                pct = entry["punctuality"]["punctuality_distribution"]["percentages"]
                n   = entry["punctuality"].get("sample_size", 0)
                is_reg = entry.get("is_regulatory_stop", False)
                flat.append({
                    "route_id":           rid,
                    "direction_id":       entry["direction_id"],
                    "stop_id":            entry["stop_id"],
                    "stop_name":          entry["stop_name"],
                    "time_type":          entry["time_type"],
                    "sample_size":        n,
                    "too_early_pct":      pct.get("too_early", 0.0),
                    "on_time_pct":        pct.get("on_time",   0.0),
                    "too_late_pct":       pct.get("too_late",  0.0),
                    "is_regulatory_stop": is_reg
                })

        # 2) Group for "all" aggregation
        buckets = defaultdict(lambda: {
            "sum_on":    0.0,
            "sum_early": 0.0,
            "sum_late":  0.0,
            "n":         0,
            "reg_flag":  False
        })
        for e in flat:
            key = (e["route_id"], e["direction_id"], e["stop_id"], e["stop_name"])
            b = buckets[key]
            b["n"]         += e["sample_size"]
            b["sum_on"]    += e["on_time_pct"]   * e["sample_size"]
            b["sum_early"] += e["too_early_pct"] * e["sample_size"]
            b["sum_late"]  += e["too_late_pct"]  * e["sample_size"]
            b["reg_flag"]   = b["reg_flag"] or e["is_regulatory_stop"]

        # 3) Append the aggregated entries
        for (rid, did, sid, sname), agg in buckets.items():
            total_n = agg["n"]
            if total_n == 0:
                continue
            flat.append({
                "route_id":           rid,
                "direction_id":       did,
                "stop_id":            sid,
                "stop_name":          sname,
                "time_type":          "all",
                "sample_size":        total_n,
                "too_early_pct":      agg["sum_early"] / total_n,
                "on_time_pct":        agg["sum_on"]   / total_n,
                "too_late_pct":       agg["sum_late"]  / total_n,
                # carry through regulatory status:
                "is_regulatory_stop": agg["reg_flag"]
            })

        # 4) Write out the file
        self._dump_safe(flat, export_root / "global_punctuality_stops.json")
        print(f"🔄 Exported global punctuality to {export_root / 'global_punctuality_stops.json'}")

    def _export_csv_underperforming_regulatory_stops(self, export_root: Path, threshold: float = 80.0):
        """
        Export CSV of regulatory stops with aggregated on-time % below threshold.
        Reads `global_punctuality_stops.json` and pulls only the `time_type=="all"` entries
        where `is_regulatory_stop` is True and `on_time_pct` < threshold.

        Columns: route_id, direction_id, stop_id, stop_name, on_time_pct, sample_size
        """
        # load the global punctuality file we just wrote
        jp = export_root / "global_punctuality_stops.json"
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        out_path = export_root / "underperforming_regulatory_stops.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "route_id", "direction_id", "stop_id",
                "stop_name", "on_time_pct", "sample_size"
            ])

            for e in data:
                if (
                    e.get("time_type") == "all"
                    and e.get("is_regulatory_stop", False)
                    and e.get("on_time_pct", 0.0) < threshold
                ):
                    writer.writerow([
                        e.get("route_id", ""),
                        e.get("direction_id", ""),
                        e.get("stop_id", ""),
                        e.get("stop_name", ""),
                        e.get("on_time_pct", ""),
                        e.get("sample_size", "")
                    ])

        print(f"🔄 Exported underperforming regulatory stops to {out_path}")

    def _export_csv_mis_tracked_stops(self, export_root: Path):
        path = export_root / "mis_tracked_stops.csv"
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
