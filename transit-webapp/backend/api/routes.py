import json
from flask import Blueprint, jsonify, abort, request
from .data_loader import load_global_file, load_route_file, list_user_files

routes_bp = Blueprint('routes', __name__)

@routes_bp.route('', methods=['GET'])
def list_routes():
    """Fetch and list available routes from the route index"""
    try:
        # Load the route index
        idx = load_global_file('route_index')
        files = list_user_files()  # List all files for the authenticated user
        
        # Find existing route folders
        existing = set()
        for file_path in files:
            if file_path.startswith('route_') and '/' in file_path:
                route_id = file_path.split('/')[0].replace('route_', '')
                existing.add(route_id)
        
        # Filter and return the routes that exist
        return jsonify({rid: info for rid, info in idx.items() if rid in existing})
    except Exception as e:
        print(f"Error loading route index: {str(e)}")
        abort(404, f"Route index not found")

@routes_bp.route('/<route_id>', methods=['GET'])
def route_detail(route_id):
    """Fetch details of a specific route by route_id"""
    try:
        # Load the route index and find the requested route
        routes_data = load_global_file('route_index')  # Load the global route index
        if route_id not in routes_data:
            abort(404, 'Route not found')
        
        # Return the route details
        return jsonify(routes_data[route_id])
    except Exception as e:
        print(f"Error fetching details for route {route_id}: {str(e)}")
        abort(404, 'Route not found')

@routes_bp.route('/<route_id>/navigation', methods=['GET'])
def route_navigation(route_id):
    """Fetch navigation data for a specific route"""
    try:
        return jsonify(load_route_file(route_id, 'routewise_navigation.json'))
    except Exception as e:
        print(f"Error loading navigation for route {route_id}: {str(e)}")
        abort(404, 'Navigation not found')

@routes_bp.route('/<route_id>/directions/<int:dir_id>/topology', methods=['GET'])
def direction_topology(route_id, dir_id):
    """Fetch topology data for a specific route and direction"""
    try:
        data = load_route_file(route_id, 'direction_topology.json')
        return jsonify(data[str(dir_id)])
    except Exception as e:
        print(f"Error loading topology for route {route_id}, direction {dir_id}: {str(e)}")
        abort(404, 'Topology file or direction not found')

@routes_bp.route('/<route_id>/directions/<int:dir_id>/performance', methods=['GET'])
def direction_performance(route_id, dir_id):
    """Fetch performance data for a specific route and direction"""
    try:
        data = load_route_file(route_id, 'performance_logs.json')
        return jsonify(data[str(dir_id)])
    except Exception as e:
        print(f"Error loading performance for route {route_id}, direction {dir_id}: {str(e)}")
        abort(404, 'Performance logs not found')

@routes_bp.route('/<route_id>/directions/<int:dir_id>/violations', methods=['GET'])
def direction_violations(route_id, dir_id):
    """Fetch violations data for a specific route and direction"""
    try:
        data = load_route_file(route_id, 'regulatory_stops.json')
        return jsonify(data[str(dir_id)])
    except Exception as e:
        print(f"Error loading violations for route {route_id}, direction {dir_id}: {str(e)}")
        abort(404, 'Violations file not found')

@routes_bp.route('/<route_id>/directions/<int:dir_id>/stop_topology', methods=['GET'])
def direction_stop_topology(route_id, dir_id):
    """Fetch stop topology data for a specific route and direction"""
    try:
        data = load_route_file(route_id, 'stop_topology.json')
        return jsonify(data[str(dir_id)])
    except Exception as e:
        print(f"Error loading stop topology for route {route_id}, direction {dir_id}: {str(e)}")
        abort(404, 'Stop topology file not found')
