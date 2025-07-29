import json
from flask import Blueprint, request, jsonify
from data_loader import load_data_file

search_bp = Blueprint('search', __name__)

@search_bp.route('', methods=['GET'])
def search():
    q = request.args.get('q', '').strip().lower()
    results = {'routes': {}, 'stops': {}}
    if not q:
        return jsonify(results)

    routes = load_data_file('global_route_index.json')
    for rid, info in routes.items():
        if q in info.get('short_name', '').lower() or q in info.get('long_name', '').lower():
            results['routes'][rid] = info

    stops = load_data_file('global_stop_index.json')
    for sid, info in stops.items():
        if q in info.get('stop_name', '').lower():
            results['stops'][sid] = info

    return jsonify(results)
