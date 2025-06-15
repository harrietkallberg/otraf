from flask import Flask, jsonify
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)

DATA_FOLDER = os.path.join(app.root_path, 'data')

def load_json(filename):
    path = os.path.join(DATA_FOLDER, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.route('/api/health')
def api_health():
    expected_files = [
        'analysis.json',
        'direction_navigation.json',
        'direction_violations.json',
        'hierarchies.json',
        'stop_violations.json'
    ]
    result = {
        fname: os.path.exists(os.path.join(DATA_FOLDER, fname))
        for fname in expected_files
    }
    return jsonify({
        'status': 'healthy',
        'files': result
    })

@app.route('/api/analysis')
def api_analysis():
    return jsonify(load_json('analysis.json'))

@app.route('/api/direction-navigation')
def api_direction_navigation():
    return jsonify(load_json('direction_navigation.json'))

@app.route('/api/direction-violations')
def api_direction_violations():
    return jsonify(load_json('direction_violations.json'))

@app.route('/api/hierarchies')
def api_hierarchies():
    return jsonify(load_json('hierarchies.json'))

@app.route('/api/stop-violations')
def api_stop_violations():
    return jsonify(load_json('stop_violations.json'))

@app.route('/api/stats')
def api_stats():
    """Optional combined object if frontend prefers one request"""
    return jsonify({
        'analysis': load_json('analysis.json'),
        'directionNavigation': load_json('direction_navigation.json'),
        'directionViolations': load_json('direction_violations.json'),
        'hierarchies': load_json('hierarchies.json'),
        'stopViolations': load_json('stop_violations.json')
    })

if __name__ == '__main__':
    print(f"Expecting files in: {DATA_FOLDER}")
    app.run(debug=True, host='0.0.0.0', port=5000)
