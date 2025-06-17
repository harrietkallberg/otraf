from flask import Flask, send_from_directory, jsonify
from pathlib import Path

app = Flask(__name__, static_folder='data', static_url_path='/data')

# Serve routewise and global data
@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

# Serve specific global JSON files as API endpoints
@app.route('/api/direction-navigation')
def api_direction_navigation():
    return send_from_directory('data/global', 'global_direction_navigation.json')

@app.route('/api/stop-violations')
def api_stop_violations():
    return send_from_directory('data/global', 'global_stop_violations.json')

@app.route('/api/direction-violations')
def api_direction_violations():
    return send_from_directory('data/global', 'global_direction_violations.json')

@app.route('/api/stop-classifications')
def api_stop_classifications():
    return send_from_directory('data/global', 'global_stop_classifications.json')

@app.route('/api/direction-classifications')
def api_direction_classifications():
    return send_from_directory('data/global', 'global_direction_classifications.json')

@app.route('/api/direction-stop-classifications')
def api_direction_stop_classifications():
    return send_from_directory('data/global', 'global_direction_stop_classifications.json')
