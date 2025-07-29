import json
from flask import Blueprint, jsonify, abort
from data_loader import load_data_file

global_bp = Blueprint('global', __name__)

def _load(name):
    try:
        return load_data_file(f'global_{name}.json')
    except:
        abort(404, f"global_{name}.json not found")

@global_bp.route('/routes', methods=['GET'])
def routes_index():
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
    return jsonify(_load('travel_times'))
