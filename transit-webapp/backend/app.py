import os
from flask import Flask
from api.routes import routes_bp
from api.stops  import stops_bp
from api.search import search_bp

def create_app():
    app = Flask(__name__)
    # point all blueprints at the same data folder
    app.config['DATA_DIR'] = os.path.join(os.path.dirname(__file__), 'data')

    app.register_blueprint(routes_bp, url_prefix='/api/routes')
    app.register_blueprint(stops_bp,  url_prefix='/api/stops')
    app.register_blueprint(search_bp, url_prefix='/api/search')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
