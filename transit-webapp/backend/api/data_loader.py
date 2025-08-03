from .supa_helper import fetch_json_from_supabase, fetch_csv_from_supabase
from flask import request, abort

def get_user_id_and_tokens():
    """Extract user ID and JWT token from the request headers."""
    # Get user_id and JWT token from headers
    user_id = request.headers.get('X-User-Id')
    auth_header = request.headers.get('Authorization')
    
    # Ensure user_id and Authorization header exist
    if not user_id:
        abort(401, "Unauthorized: No user ID provided.")
    
    if not auth_header or not auth_header.startswith('Bearer '):  # Ensure token is present and valid
        abort(401, "Unauthorized: No valid session token provided.")
    
    # Extract the JWT token from Authorization header
    auth_token = auth_header.split(' ')[1]
    
    # Optionally extract the refresh token, if needed for session management
    refresh_token = request.headers.get('X-Refresh-Token')
    if not refresh_token:
        print("No refresh token provided.")
        # Optionally handle refresh token expiration or requests requiring it.

    return user_id, auth_token, refresh_token  # Return user_id, auth_token, and refresh_token if needed

def load_static_json_file(filename):
    """Load a data file from Supabase storage."""
    user_id, auth_token, refresh_token = get_user_id_and_tokens()  # Get user ID, auth_token, and refresh_token
    print(f"Loading file: {filename} for user: {user_id}")
    return fetch_json_from_supabase(user_id, filename, auth_token, refresh_token)

def load_global_file(name):
    """Helper function specifically for global files."""
    full_path = f'global_{name}.json'
    return load_static_json_file(full_path)

def load_route_file(route_id, filename):
    """Helper function specifically for route files."""
    full_path = f'routes/{route_id}_{filename}.json'
    return load_static_json_file(full_path)

def load_stop_file(parent_id, filename):
    """Helper function specifically for route files."""
    full_path = f'stops/{parent_id}_{filename}.json'
    return load_static_json_file(full_path)

def load_csv_file(filename):
    """Helper function specifically for csv files."""
    full_path = f'csv/{filename}.csv'
    user_id, auth_token, refresh_token = get_user_id_and_tokens()  # Get user ID, auth_token, and refresh_token
    print(f"Loading file: {filename} for user: {user_id}")
    return fetch_csv_from_supabase(user_id, full_path, auth_token, refresh_token)