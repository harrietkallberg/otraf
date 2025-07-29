from supabase import create_client
import json

def get_supabase():
    # Only call this once ideally
    SUPABASE_URL = 'https://uaqfokbeqneynggspqwk.supabase.co'
    SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVhcWZva2JlcW5leW5nZ3NwcXdrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM0NTQwMjIsImV4cCI6MjA2OTAzMDAyMn0.m04qZFLCqgHZGCpFjWVGrX6mxskdlsARsul7MStgV_8'
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_json_from_supabase(user_id: str, filename: str):
    supabase = get_supabase()
    path = f"{user_id}/{filename}"
    res = supabase.storage.from_('your-bucket').download(path)
    return json.loads(res.decode('utf-8'))

def list_files_in_user_bucket(user_id: str):
    supabase = get_supabase()
    res = supabase.storage.from_('your-bucket').list(path=user_id)
    return [file['name'] for file in res]
