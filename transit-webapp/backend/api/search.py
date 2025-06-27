import os
import json
from flask import Blueprint, request, jsonify, current_app

search_bp = Blueprint('search', __name__)

def get_data_dir():
    return current_app.config['DATA_DIR']

@search_bp.route('', methods=['GET'])
def search():
    q = request.args.get('q', '').strip().lower()
    results = {'routes': {}, 'stops': {}}
    if not q:
        return jsonify(results)

    d = get_data_dir()
    # search routes
    routes = json.load(open(os.path.join(d, 'global_route_index.json'), encoding='utf-8'))
    for rid, info in routes.items():
        if q in info.get('short_name', '').lower() or q in info.get('long_name', '').lower():
            results['routes'][rid] = info

    # search stops
    stops = json.load(open(os.path.join(d, 'global_stop_index.json'), encoding='utf-8'))
    for sid, info in stops.items():
        if q in info.get('stop_name', '').lower():
            results['stops'][sid] = info

    return jsonify(results)
