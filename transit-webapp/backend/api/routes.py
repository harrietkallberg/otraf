import json
from flask import Blueprint, jsonify, abort, request
from data_loader import load_data_file, list_user_files

routes_bp = Blueprint('routes', __name__)

def get_user_id():
    return request.headers.get('X-User-Id') or 'dev-user'

@routes_bp.route('', methods=['GET'])
def list_routes():
    idx = load_data_file('global_route_index.json')
    files = list_user_files()
    existing = set(name.split('/')[0].split('route_')[1] for name in files if name.startswith('route_'))
    return jsonify({rid: info for rid, info in idx.items() if rid in existing})

@routes_bp.route('/<route_id>', methods=['GET'])
def route_detail(route_id):
    routes = list_routes().json
    if route_id not in routes:
        abort(404, 'Route not found')
    return jsonify(routes[route_id])

@routes_bp.route('/<route_id>/navigation', methods=['GET'])
def route_navigation(route_id):
    try:
        return jsonify(load_data_file(f'route_{route_id}/routewise_navigation.json'))
    except:
        abort(404, 'Navigation not found')

@routes_bp.route('/<route_id>/directions/<int:dir_id>/topology', methods=['GET'])
def direction_topology(route_id, dir_id):
    try:
        data = load_data_file(f'route_{route_id}/direction_topology.json')
        return jsonify(data[str(dir_id)])
    except:
        abort(404, 'Topology file or direction not found')

@routes_bp.route('/<route_id>/directions/<int:dir_id>/performance', methods=['GET'])
def direction_performance(route_id, dir_id):
    try:
        data = load_data_file(f'route_{route_id}/performance_logs.json')
        return jsonify(data[str(dir_id)])
    except:
        abort(404, 'Performance logs not found')

@routes_bp.route('/<route_id>/directions/<int:dir_id>/violations', methods=['GET'])
def direction_violations(route_id, dir_id):
    try:
        data = load_data_file(f'route_{route_id}/regulatory_stops.json')
        return jsonify(data[str(dir_id)])
    except:
        abort(404, 'Violations file not found')

@routes_bp.route('/<route_id>/directions/<int:dir_id>/stop_topology', methods=['GET'])
def direction_stop_topology(route_id, dir_id):
    try:
        data = load_data_file(f'route_{route_id}/stop_topology.json')
        return jsonify(data[str(dir_id)])
    except:
        abort(404, 'Stop topology file not found')
