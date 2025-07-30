from flask import Blueprint, request, abort, send_file
from .data_loader import load_data_file
from io import BytesIO
import mimetypes
import json

csv_routes_bp = Blueprint('csv', __name__)

@csv_routes_bp.route('/<path:filename>', methods=['GET'])
def download_csv(filename):
    try:
        # Load the file from Supabase storage as raw binary data (not as JSON)
        raw = load_data_file(f'csv/{filename}')  # Assuming the filename includes 'csv/'
        
        # Create a buffer and write the raw content to it
        buf = BytesIO()
        buf.write(raw)  # raw is already binary data, so no need to encode to UTF-8
        buf.seek(0)
        
        # Guess the mime type or manually set it for CSV files
        mime_type = mimetypes.guess_type(filename)[0] or 'text/csv'  # Default to 'text/csv' for CSV files
        
        # Send the file as an attachment with the appropriate mime type
        return send_file(buf, mimetype=mime_type, as_attachment=True, download_name=filename)
    
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error in downloading file {filename}: {e}")
        abort(404)

