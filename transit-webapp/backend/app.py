from flask import Flask, send_from_directory
from pathlib import Path

# ✅ Add this
from search_backend import search_bp

app = Flask(__name__, static_folder='data', static_url_path='/data')

# ✅ Register the blueprint
app.register_blueprint(search_bp)

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

@app.route('/api/routes-summary')
def api_routes_summary():
    return send_from_directory('data', 'routes_summary.json')

@app.route('/api/navigation')
def api_navigation():
    return send_from_directory('data', 'navigation_structures.json')

@app.route('/api/stop-violations')
def api_stop_violations():
    return send_from_directory('data', 'global_stop_violations.json')

@app.route('/api/direction-violations')
def api_direction_violations():
    return send_from_directory('data', 'global_direction_violations.json')

@app.route('/api/performance-violations')
def api_performance_violations():
    return send_from_directory('data', 'global_performance_violations.json')

if __name__ == "__main__":
    app.run(debug=True)
