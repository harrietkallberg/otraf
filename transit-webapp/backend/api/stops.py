import json
from flask import Blueprint, jsonify, abort, request
from .data_loader import load_global_file, load_route_file, list_user_items

stops_bp = Blueprint('stops', __name__)

@stops_bp.route('', methods=['GET'])
def list_stops():
    """Fetch and list all available stops."""
    try:
        # Load the stop index from the global stop file
        stop_data = load_global_file('stop_index')
        files = list_user_items()  # List all files for the authenticated user
        
        # Extract and filter the stops based on their availability in user files (if needed)
        available_stops = {stop_id: info for stop_id, info in stop_data.items() if stop_id in files}
        
        return jsonify(available_stops)
    except Exception as e:
        print(f"Error loading stop index: {str(e)}")
        abort(404, "Stop index not found")

@stops_bp.route('/<stop_name>', methods=['GET'])  # Corrected route for stop_name
def get_stop_by_name(stop_name):
    if not stop_name:
        abort(400, 'Stop name is required')  # Ensure stop_name is provided in the request

    stop_data = load_global_file('stop_index')  # Load stop index data (assuming global_stop_index.json)
    
    # Look for the stop_name within the stop_index and return the corresponding stop information
    for stop_id, stop_info in stop_data.items():
        if stop_info['stop_name'].lower() == stop_name.lower():  # Case insensitive match
            # Return the stop information as a JSON response
            return jsonify({
                "stop_id": stop_id,
                "stop_name": stop_info['stop_name'],
                "routes": stop_info['routes'],
                "directions": stop_info['directions'],
                "label_stats": stop_info['label_stats'],
                "violation_stats": stop_info['violation_stats'],
                "performance_keys": stop_info.get('performance_keys', {}),
                "severity_counts_by_severity": stop_info.get('severity_counts_by_severity', {}),
            })
    
    # If no matching stop_name is found, return a 404 error
    abort(404, 'Stop not found')
