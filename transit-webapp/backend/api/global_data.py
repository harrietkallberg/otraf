import json
from flask import Blueprint, jsonify
from .data_loader import load_global_file

global_bp = Blueprint('global', __name__)

@global_bp.route('/routes', methods=['GET'])
def routes_index():
    return jsonify(load_global_file('route_index'))

@global_bp.route('/labels', methods=['GET'])
def labels():
    return jsonify(load_global_file('labels'))

@global_bp.route('/violations', methods=['GET'])
def violations():
    return jsonify(load_global_file('violations'))

@global_bp.route('/time_types', methods=['GET'])
def time_types():
    return jsonify(load_global_file('time_types'))

@global_bp.route('/stops', methods=['GET'])
def stops():
    return jsonify(load_global_file('stop_index'))

@global_bp.route('/travel_times', methods=['GET'])
def travel_times():
    return jsonify(load_global_file('travel_times'))
