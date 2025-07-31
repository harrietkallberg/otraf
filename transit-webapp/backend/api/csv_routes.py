from flask import Blueprint
from .data_loader import load_csv_file

csv_routes_bp = Blueprint('csv', __name__)

@csv_routes_bp.route('/travel_times', methods=['GET'])
def travel_times():
    return load_csv_file('global_travel_times')

@csv_routes_bp.route('/mis_tracked_stops', methods=['GET'])
def mistracked_stops():
    return load_csv_file('mis_tracked_stops')

@csv_routes_bp.route('/underperforming_regulatory_stops', methods=['GET'])
def underperforming_regulatory_stops():
    return load_csv_file('underperforming_regulatory_stops')
