# app.py (or create_app())
import os
from flask import Flask, jsonify

from api.routes      import routes_bp
from api.stops       import stops_bp
from api.global_data import global_bp
from api.csv         import csv_bp

def create_app():
    app = Flask(__name__)

    # Example flag
    app.config['USE_SUPABASE'] = os.getenv("USE_SUPABASE", "false").lower() == "true"

    app.register_blueprint(routes_bp,     url_prefix='/api/routes')
    app.register_blueprint(stops_bp,      url_prefix='/api/stops')
    app.register_blueprint(global_bp,     url_prefix='/api/global')
    app.register_blueprint(csv_bp,        url_prefix='/api/csv')


    
    @app.get("/")
    def index():
        return jsonify(status="ok", service="backend", docs="/api/*")

    @app.get("/api/health")
    def health():
        return jsonify(status="healthy")


    return app

if __name__ == '__main__':
    create_app().run(debug=True)
