# api/csv_routes.py
import os
from flask import Blueprint, current_app, send_from_directory, abort

csv_routes_bp = Blueprint('csv_routes', __name__)

@csv_routes_bp.route('/<path:filename>')
def download_csv(filename):
    data_dir = current_app.config['DATA_DIR']
    csv_dir  = os.path.join(data_dir, 'csv')
    full_path = os.path.join(csv_dir, filename)

    # debug
    print("→ [csv_routes] DATA_DIR:", data_dir)
    print("→ [csv_routes] CSV contents:", os.listdir(csv_dir))

    if not os.path.isfile(full_path):
        abort(404)
    return send_from_directory(csv_dir, filename, as_attachment=True)
