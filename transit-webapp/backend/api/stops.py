import os
import json
from flask import Blueprint, jsonify, abort, current_app

stops_bp = Blueprint('stops', __name__)

def get_data_dir():
    return current_app.config['DATA_DIR']

def list_route_ids():
    d = get_data_dir()
    return [
        name.split('route_')[1]
        for name in os.listdir(d)
        if name.startswith('route_') and os.path.isdir(os.path.join(d, name))
    ]

@stops_bp.route('', methods=['GET'])
def list_stops():
    fn = os.path.join(get_data_dir(), 'global_stop_index.json')
    return jsonify(json.load(open(fn, encoding='utf-8')))

@stops_bp.route('/<stop_id>', methods=['GET'])
def stop_detail(stop_id):
    all_stops = list_stops().json
    if stop_id not in all_stops:
        abort(404, 'Stop not found')
    meta = all_stops[stop_id]

    # collect violations per route+direction
    violations_by_route = {}
    for rid in list_route_ids():
        fn = os.path.join(get_data_dir(), f'route_{rid}', 'regulatory_stops.json')
        if not os.path.exists(fn):
            continue
        reg = json.load(open(fn, encoding='utf-8'))
        for dir_id, stops in reg.items():
            if stop_id in stops:
                violations_by_route.setdefault(rid, {})[dir_id] = stops[stop_id]

    return jsonify({
        'meta': meta,
        'violations_by_route': violations_by_route
    })
