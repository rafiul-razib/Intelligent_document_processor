from flask import Flask

from .config import Config


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    from .routes.main import main_bp

    app.register_blueprint(main_bp)
    return app
