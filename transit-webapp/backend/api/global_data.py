# backend/api/global_data.py
import os, json
from flask import Blueprint, jsonify, current_app, abort

global_bp = Blueprint('global', __name__)

def get_data_dir():
    return current_app.config['DATA_DIR']

def _load(name):
    fn = f"global_{name}.json"
    path = os.path.join(get_data_dir(), fn)
    if not os.path.exists(path):
        abort(404, f"{fn} not found")
    return json.load(open(path, encoding="utf-8"))

@global_bp.route('/routes', methods=['GET'])
def routes_index():
    """Return global_route_index.json"""
    return jsonify(_load('route_index'))

@global_bp.route('/labels', methods=['GET'])
def labels():
    return jsonify(_load('labels'))

@global_bp.route('/violations', methods=['GET'])
def violations():
    return jsonify(_load('violations'))

@global_bp.route('/time_types', methods=['GET'])
def time_types():
    return jsonify(_load('time_types'))

@global_bp.route('/stops', methods=['GET'])
def stops():
    return jsonify(_load('stop_index'))

@global_bp.route('/travel_times', methods=['GET'])
def travel_times():
    """Return global_travel_times.json"""
    return jsonify(_load('travel_times'))

