import json
from pathlib import Path
from collections import defaultdict

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
          - violation_stats per domain
          - label_keys per domain
          - violation_keys per domain
          - performance_keys availability
        """
        domains = ["stop_topology", "direction_topology", "regulatory", "parent_station"]
        stop_index: dict[str, dict] = {}

        for rid, log in self.logs.items():
            nav = log.navigation_structures.get("routewise_navigation", {}) or {}
            for direction in nav.get("directions", []):
                did = direction.get("direction_id")
                for stop in direction.get("stops", []):
                    sid = stop.get("stop_id")
                    if sid is None:
                        continue

                    entry = stop_index.setdefault(sid, {
                        "stop_name":        stop.get("stop_name", "UNKNOWN"),
                        "routes":           set(),
                        "directions":       defaultdict(set),
                        "label_stats":      {d: {"routes": set(), "occurrences": 0} for d in domains},
                        "violation_stats":  {d: {"routes": set(), "occurrences": 0} for d in domains},
                        "label_keys":       {d: [] for d in domains},
                        "violation_keys":   {d: [] for d in domains},
                        "performance_keys": defaultdict(lambda: {"histogram": None, "punctuality": None})
                    })

                    entry["routes"].add(rid)
                    entry["directions"][rid].add(did)

                    # Label stats & keys
                    for domain, keys in (stop.get("labels") or {}).items():
                        if domain not in domains:
                            continue
                        for k in (keys or []):
                            entry["label_stats"][domain]["routes"].add(rid)
                            entry["label_stats"][domain]["occurrences"] += 1
                            entry["label_keys"][domain].append(k)

                    # Violation stats & keys
                    for domain, keys in (stop.get("violations") or {}).items():
                        if domain not in domains:
                            continue
                        for k in (keys or []):
                            entry["violation_stats"][domain]["routes"].add(rid)
                            entry["violation_stats"][domain]["occurrences"] += 1
                            entry["violation_keys"][domain].append(k)

                    # Performance availability
                    p_avail = stop.get("performance_availability") or {}
                    for tt, hkey in (p_avail.get("histograms") or {}).items():
                        if hkey:
                            entry["performance_keys"][tt]["histogram"] = hkey
                    for tt, pkey in (p_avail.get("punctuality") or {}).items():
                        if pkey:
                            entry["performance_keys"][tt]["punctuality"] = pkey

        # Sanitize for JSON: convert sets to lists
        final_index = {}
        for sid, e in stop_index.items():
            final_index[sid] = {
                "stop_name":      e["stop_name"],
                "routes":         sorted(e["routes"]),
                "directions":     {rid: sorted(list(ds)) for rid, ds in e["directions"].items()},
                "label_stats": {
                    dom: {"routes": sorted(list(data["routes"])), "occurrences": data["occurrences"]}
                    for dom, data in e["label_stats"].items()
                },
                "label_keys":    {dom: list(keys) for dom, keys in e["label_keys"].items()},
                "violation_stats": {
                    dom: {"routes": sorted(list(data["routes"])), "occurrences": data["occurrences"]}
                    for dom, data in e["violation_stats"].items()
                },
                "violation_keys": {dom: list(keys) for dom, keys in e["violation_keys"].items()},
                "performance_keys": {
                    tt: {"histogram": data["histogram"], "punctuality": data["punctuality"]}
                    for tt, data in e["performance_keys"].items()
                }
            }

        self._dump_safe(final_index, export_root / "global_stop_index.json")
        print(f"🔄 Exported detailed global stop index to {export_root / 'global_stop_index.json'}")

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

        tt_list = sorted(time_types)
        self._dump_safe(tt_list, export_root / "global_time_types.json")
        print(f"🔄 Exported global time types to {export_root / 'global_time_types.json'}")

    def _export_global_violations(self, export_root: Path):
        """
        Flatten all violation entries across routes into a single list JSON.
        """
        flat = []
        for log in self.logs.values():
            flat.extend(list(log.stop_topology_logs.get("stop_id_violations", {}).values()))
            flat.extend(list(log.direction_topology_logs.get("direction_violations", {}).values()))
            flat.extend(list(log.regulatory_stops_logs.get("stop_id_regulatory_violations", {}).values()))

        self._dump_safe(flat, export_root / "global_violations.json")
        print(f"🔄 Exported global violations to {export_root / 'global_violations.json'}")

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
