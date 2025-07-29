from flask import Blueprint, request, abort, send_file
from data_loader import load_data_file
from io import BytesIO
import mimetypes
import json

csv_routes_bp = Blueprint('csv_routes', __name__)

@csv_routes_bp.route('/<path:filename>')
def download_csv(filename):
    try:
        raw = load_data_file(f'csv/{filename}')
        buf = BytesIO()
        buf.write(json.dumps(raw).encode('utf-8'))
        buf.seek(0)
        mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        return send_file(buf, mimetype=mime_type, as_attachment=True, download_name=filename)
    except:
        abort(404)
