import json
from flask import Blueprint, jsonify, abort, request
from data_loader import load_data_file, list_user_files

stops_bp = Blueprint('stops', __name__)

def get_user_id():
    return request.headers.get('X-User-Id') or 'dev-user'

def list_route_ids():
    files = list_user_files()
    return [name.split('/')[0].split('route_')[1] for name in files if name.startswith('route_')]

@stops_bp.route('', methods=['GET'])
def list_stops():
    return jsonify(load_data_file('global_stop_index.json'))

@stops_bp.route('/<stop_id>', methods=['GET'])
def stop_detail(stop_id):
    all_stops = list_stops().json
    if stop_id not in all_stops:
        abort(404, 'Stop not found')
    meta = all_stops[stop_id]

    violations_by_route = {}
    for rid in list_route_ids():
        try:
            reg = load_data_file(f'route_{rid}/regulatory_stops.json')
        except:
            continue
        for dir_id, stops in reg.items():
            if stop_id in stops:
                violations_by_route.setdefault(rid, {})[dir_id] = stops[stop_id]

    return jsonify({
        'meta': meta,
        'violations_by_route': violations_by_route
    })
