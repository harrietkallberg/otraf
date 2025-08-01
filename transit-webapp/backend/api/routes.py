import json
from flask import Blueprint, jsonify, abort, request
from .data_loader import load_global_file, load_route_file, list_user_items, list_route_items

routes_bp = Blueprint('routes', __name__)

@routes_bp.route('', methods=['GET'])
def list_routes():
    """Fetch and list available routes from the route index"""
    try:
        # Load the route index
        idx = load_global_file('route_index')
        files = list_user_items()  # List all files for the authenticated user
        
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

@routes_bp.route('/<route_id>/navigation_structure', methods=['GET'])
def route_navigation_structure(route_id):
    """Fetch the first level of the navigation structure for a specific route by route_id"""
    try:
        # Load the navigational structure for the route
        navigation_structure = load_route_file(route_id, 'routewise_navigation')  # Load the file

        # Return only the first level of the navigation structure
        # Assuming the structure is a dictionary or a list of dictionaries
        first_level = {key: navigation_structure[key] for key in list(navigation_structure.keys())}  # First key-value

        return jsonify(first_level)
    except Exception as e:
        print(f"Error loading navigation structure for route {route_id}: {str(e)}")
        abort(404, 'Navigation structure not found')
