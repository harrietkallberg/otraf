from flask import Blueprint
from .data_loader import  load_route_file
routes_bp = Blueprint('routes', __name__)

@routes_bp.route('/<route_id>', methods=['GET'])  # Corrected route for stop_name
def route_nav(route_id):
    return load_route_file(route_id, 'route_nav')