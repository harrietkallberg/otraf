from flask import Flask, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

def load_json_file(filename):
    """Load JSON file safely"""
    try:
        with open(f'./data/{filename}', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filename}")
        return {}

# Health check endpoint
@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Transit Analysis API",
        "available_files": os.listdir('./data') if os.path.exists('./data') else []
    })

# Main endpoints that your React app expects
@app.route('/api/routes')
def get_routes():
    """Get all routes - React expects this endpoint"""
    return jsonify(load_json_file('route_stops.json'))

@app.route('/api/analysis')
def get_analysis():
    """Get stop analysis - React expects this endpoint"""
    return jsonify(load_json_file('stop_analysis.json'))
@app.route('/api/stats')
def get_stats():
    """Get statistics directly from the JSON files - no calculations"""
    route_stops = load_json_file('route_stops.json')
    stop_analysis = load_json_file('stop_analysis.json')
    
    if not route_stops or not stop_analysis:
        return jsonify({"error": "Required data files not found"}), 404
    
    # Just return the data as-is from the JSON files
    return jsonify({
        "route_stops": route_stops,
        "stop_analysis": stop_analysis
    })

# Your original endpoints (keep them working)
@app.route('/api/route-stops')
def get_route_stops():
    return jsonify(load_json_file('route_stops.json'))

@app.route('/api/stop-analysis')
def get_stop_analysis():
    return jsonify(load_json_file('stop_analysis.json'))

@app.route('/api/histograms')
def get_histograms():
    return jsonify(load_json_file('delay_histograms_by_time.json'))

if __name__ == '__main__':
    app.run(debug=True)