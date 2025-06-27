import os
import json
from flask import Blueprint, jsonify, abort, current_app

routes_bp = Blueprint('routes', __name__)

def get_data_dir():
    return current_app.config['DATA_DIR']

def list_route_ids():
    d = get_data_dir()
    return [
        name.split('route_')[1]
        for name in os.listdir(d)
        if name.startswith('route_') and os.path.isdir(os.path.join(d, name))
    ]

@routes_bp.route('', methods=['GET'])
def list_routes():
    d = get_data_dir()
    idx = json.load(open(os.path.join(d, 'global_route_index.json'), encoding='utf-8'))
    existing = set(list_route_ids())
    return jsonify({rid: info for rid, info in idx.items() if rid in existing})

@routes_bp.route('/<route_id>', methods=['GET'])
def route_detail(route_id):
    routes = list_routes().json
    if route_id not in routes:
        abort(404, 'Route not found')
    return jsonify(routes[route_id])

@routes_bp.route('/<route_id>/navigation', methods=['GET'])
def route_navigation(route_id):
    fn = os.path.join(get_data_dir(), f'route_{route_id}', 'routewise_navigation.json')
    if not os.path.exists(fn):
        abort(404, 'Navigation not found')
    return jsonify(json.load(open(fn, encoding='utf-8')))

@routes_bp.route('/<route_id>/directions/<int:dir_id>/topology', methods=['GET'])
def direction_topology(route_id, dir_id):
    fn = os.path.join(get_data_dir(), f'route_{route_id}', 'direction_topology.json')
    if not os.path.exists(fn):
        abort(404, 'Topology file not found')
    data = json.load(open(fn, encoding='utf-8'))
    key = str(dir_id)
    if key not in data:
        abort(404, 'Direction not found')
    return jsonify(data[key])

@routes_bp.route('/<route_id>/directions/<int:dir_id>/performance', methods=['GET'])
def direction_performance(route_id, dir_id):
    fn = os.path.join(get_data_dir(), f'route_{route_id}', 'performance_logs.json')
    if not os.path.exists(fn):
        abort(404, 'Performance logs not found')
    data = json.load(open(fn, encoding='utf-8'))
    key = str(dir_id)
    if key not in data:
        abort(404, 'No performance data for this direction')
    return jsonify(data[key])

@routes_bp.route('/<route_id>/directions/<int:dir_id>/violations', methods=['GET'])
def direction_violations(route_id, dir_id):
    fn = os.path.join(get_data_dir(), f'route_{route_id}', 'regulatory_stops.json')
    if not os.path.exists(fn):
        abort(404, 'Violations file not found')
    data = json.load(open(fn, encoding='utf-8'))
    key = str(dir_id)
    if key not in data:
        abort(404, 'No violations for this direction')
    return jsonify(data[key])

@routes_bp.route('/<route_id>/directions/<int:dir_id>/stop_topology', methods=['GET'])
def direction_stop_topology(route_id, dir_id):
    fn = os.path.join(get_data_dir(), f'route_{route_id}', 'stop_topology.json')
    if not os.path.exists(fn):
        abort(404, 'Stop topology file not found')
    data = json.load(open(fn, encoding='utf-8'))
    key = str(dir_id)
    if key not in data:
        abort(404, 'No stop topology for this direction')
    return jsonify(data[key])
