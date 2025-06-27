# backend/api/global.py
import os
import json
from flask import Blueprint, jsonify, current_app, abort

global_bp = Blueprint('global', __name__)

def get_data_dir():
    return current_app.config['DATA_DIR']

def _load(name):
    path = os.path.join(get_data_dir(), f'global_{name}.json')
    if not os.path.exists(path):
        abort(404, f"{name} not found")
    return json.load(open(path, encoding='utf-8'))

@global_bp.route('/labels', methods=['GET'])
def labels():
    """Return the full global_labels.json"""
    return jsonify(_load('labels'))

@global_bp.route('/violations', methods=['GET'])
def violations():
    """Return the full global_violations.json"""
    return jsonify(_load('violations'))

@global_bp.route('/time_types', methods=['GET'])
def time_types():
    """Return global_time_types.json"""
    return jsonify(_load('time_types'))

@global_bp.route('/stops', methods=['GET'])
def stop_index():
    """Return global_stop_index.json"""
    return jsonify(_load('stop_index'))
