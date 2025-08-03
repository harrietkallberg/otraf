from supabase import create_client
import json
from flask import abort, send_file
from io import BytesIO

SUPABASE_URL = 'https://uaqfokbeqneynggspqwk.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVhcWZva2JlcW5leW5nZ3NwcXdrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM0NTQwMjIsImV4cCI6MjA2OTAzMDAyMn0.m04qZFLCqgHZGCpFjWVGrX6mxskdlsARsul7MStgV_8'

def get_supabase(auth_token: str, refresh_token: str):
    """Initialize Supabase client with JWT token and refresh token for session management."""
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Set the authentication session using the JWT token and refresh token
    supabase.auth.set_session(auth_token, refresh_token)  # Pass both tokens here
    
    return supabase

def fetch_json_from_supabase(user_id: str, filename: str, auth_token: str, refresh_token:str):
    """Fetch JSON file from Supabase storage bucket."""
    supabase = get_supabase(auth_token, refresh_token)  # Pass JWT token here for authentication
    bucket_name = 'your-bucket'
    path = f"{user_id}/{filename}" if not filename.startswith(user_id + '/') else filename
    print(f"Attempting to fetch: {path} from bucket: {bucket_name}")
    
    try:
        # Try downloading with Supabase client
        res = supabase.storage.from_(bucket_name).download(path)
        return json.loads(res.decode('utf-8'))
    except Exception as e:
        print(f"Failed with Supabase client: {e}")
        raise Exception(f"Error fetching file: {path}")

def fetch_csv_from_supabase(user_id: str, filename: str, auth_token: str, refresh_token: str):
    """Fetch CSV file from Supabase storage bucket and return as downloadable file."""
    supabase = get_supabase(auth_token, refresh_token)
    bucket_name = 'your-bucket'  # Update this if using a different bucket
    path = f"{user_id}/{filename}" if not filename.startswith(user_id + '/') else filename
    
    try:
        # Download the file as raw binary data
        raw = supabase.storage.from_(bucket_name).download(path)
        # Create a buffer and write the raw content to it
        buf = BytesIO()
        buf.write(raw)
        buf.seek(0)
        mime_type = 'text/csv, charset=utf-8'
        # Send the file as an attachment with appropriate mime type
        return send_file(buf, mimetype=mime_type, as_attachment=True, download_name=filename)
    
    except Exception as e:
        print(f"Error in downloading file {filename}: {e}")
        abort(404)
