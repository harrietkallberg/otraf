# backend/search_backend.py
from flask import Blueprint, request, jsonify
from pathlib import Path
import json

search_bp = Blueprint('search', __name__)

data_dir = Path(__file__).resolve().parent / 'data'
route_dirs = [p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()]
global_dir = data_dir

TIME_TYPES = ["am_rush", "day", "pm_rush", "night", "weekend"]

def load_json(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else {}

stop_cache = load_json(global_dir / "global_stop_classifications.json")
direction_cache = load_json(global_dir / "global_direction_classifications.json")
direction_stop_cache = load_json(global_dir / "global_direction_stop_classifications.json")
stop_violations = load_json(global_dir / "global_stop_violations.json")
direction_violations = load_json(global_dir / "global_direction_violations.json")
nav_struct = load_json(global_dir / "navigation_structures.json")
performance_logs = load_json(global_dir / "performance_logs.json")

@search_bp.route("/api/search")
def advanced_search():
    q = request.args.get("q", "").lower().strip()
    if not q:
        return jsonify({"tokens": [], "matches": {}})

    tokens = q.split()
    matches = {
        "stops": [],
        "routes": [],
        "directions": [],
        "time_types": [],
        "labels": [],
        "regulatory_flags": []
    }

    # Time types
    for token in tokens:
        for t in TIME_TYPES:
            if token in t:
                matches["time_types"].append(t)

    # Stops
    for stop_id, meta in stop_cache.get("metadata", {}).items():
        for token in tokens:
            for key in ["stop_id", "stop_name", "parent_station"]:
                if token in str(meta.get(key, "")).lower():
                    matches["stops"].append({
                        "stop_id": stop_id,
                        "matched_on": key,
                        "meta": meta,
                        "violations": stop_violations.get(stop_id, [])
                    })
                    break

    # Routes
    seen_routes = set()
    for dir_key, meta in direction_cache.get("metadata", {}).items():
        route_id = meta.get("route_id", "")
        route_long = meta.get("route_long_name", "")
        route_short = meta.get("route_short_name", "")
        for token in tokens:
            matched_on = None
            if token in route_id.lower():
                matched_on = "route_id"
            elif token in route_long.lower():
                matched_on = "route_long_name"
            elif token in route_short.lower():
                matched_on = "route_short_name"

            if matched_on and route_id not in seen_routes:
                matches["routes"].append({
                    "route_id": route_id,
                    "matched_on": matched_on,
                    "meta": meta
                })
                seen_routes.add(route_id)

    # Directions
    for dir_key, meta in direction_cache.get("metadata", {}).items():
        route_id = meta.get("route_id")
        direction_id = dir_key.split("|")[-1]
        for token in tokens:
            if token in direction_id.lower():
                matches["directions"].append({
                    "direction_id": direction_id,
                    "route_id": route_id,
                    "meta": meta,
                    "navigation": nav_struct.get(dir_key, {}),
                    "violations": direction_violations.get(dir_key, [])
                })
                break

    # Labels
    label_sets = [stop_cache, direction_cache, direction_stop_cache]
    for label_set in label_sets:
        for label_key in label_set:
            if label_key == "metadata":
                continue
            for token in tokens:
                if token in label_key.lower():
                    matches["labels"].append(label_key)
                    break

    # Regulatory
    reg_violations = performance_logs.get("regulatory_stop_id_violations", {})
    for key, value in reg_violations.items():
        for token in tokens:
            if token in key.lower():
                matches["regulatory_flags"].append({"key": key, "value": value})

    return jsonify({"tokens": tokens, "matches": matches})
