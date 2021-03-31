"""Initialize Flask app."""
from flask import Flask
from flask_assets import Environment
import os

UPLOAD_FOLDER = os.path.join(os.getcwd(),"/app/uploads/")

def init_app():
    """Construct core Flask application"""
    app = Flask(__name__, instance_relative_config=False)
    #app.config.from_object("config.Config")
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config["SECRET_KEY"] = "457895"
    assets = Environment()
    assets.init_app(app)

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes
        return app