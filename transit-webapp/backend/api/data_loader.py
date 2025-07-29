# data_loader.py
import json
from supa_helper import fetch_json_from_supabase, list_files_in_user_bucket
from flask import current_app, request

def get_user_id():
    return request.headers.get('X-User-Id')  # Adapt as needed

def load_data_file(filename):
    user_id = get_user_id()
    return fetch_json_from_supabase(user_id, filename)

def list_user_files():
    user_id = get_user_id()
    return list_files_in_user_bucket(user_id)
