import json
from flask import Blueprint, jsonify, abort, request
from .data_loader import load_global_file, load_route_file, list_user_files

stops_bp = Blueprint('stops', __name__)

def list_route_ids():
    """Get list of route IDs from file structure"""
    files = list_user_files()
    route_ids = []
    for file_path in files:
        if file_path.startswith('route_') and '/' in file_path:
            route_id = file_path.split('/')[0].replace('route_', '')
            if route_id not in route_ids:
                route_ids.append(route_id)
    return route_ids

@stops_bp.route('', methods=['GET'])
def list_stops():
    return jsonify(load_global_file('stop_index'))

@stops_bp.route('/<stop_id>', methods=['GET'])
def stop_detail(stop_id):
    all_stops = load_global_file('stop_index')
    
    if stop_id not in all_stops:
        abort(404, 'Stop not found')
    
    meta = all_stops[stop_id]

    violations_by_route = {}
    for rid in list_route_ids():
        try:
            reg = load_route_file(rid, 'regulatory_stops.json')
        except:
            continue
        
        for dir_id, stops in reg.items():
            if stop_id in stops:
                violations_by_route.setdefault(rid, {})[dir_id] = stops[stop_id]

    return jsonify({
        'meta': meta,
        'violations_by_route': violations_by_route
    })
