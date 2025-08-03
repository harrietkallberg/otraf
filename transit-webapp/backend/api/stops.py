from flask import Blueprint
from .data_loader import load_stop_file

stops_bp = Blueprint('stops', __name__)

@stops_bp.route('/<parent_id>', methods=['GET'])  # Corrected route for stop_name
def stop_nav(parent_id):
    return load_stop_file(parent_id, 'stop_nav')
