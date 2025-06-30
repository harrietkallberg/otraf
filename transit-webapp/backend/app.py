# app.py (or create_app())
import os
from flask import Flask

from api.routes      import routes_bp
from api.stops       import stops_bp
from api.search      import search_bp
from api.global_data import global_bp
from api.csv_routes  import csv_routes_bp

def create_app():
    app = Flask(__name__)
    app.config['DATA_DIR'] = os.path.join(os.path.dirname(__file__), 'data')

    app.register_blueprint(routes_bp,     url_prefix='/api/routes')
    app.register_blueprint(stops_bp,      url_prefix='/api/stops')
    app.register_blueprint(search_bp,     url_prefix='/api/search')
    app.register_blueprint(global_bp,     url_prefix='/api/global')
    app.register_blueprint(csv_routes_bp, url_prefix='/api/csv')

    return app

if __name__ == '__main__':
    create_app().run(debug=True)
