"""src/api/app.py — Flask application factory."""

from __future__ import annotations

from pathlib import Path
from flask import Flask
from flask_cors import CORS


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    - Sets template folder to dashboard/templates/
    - Sets static folder to dashboard/static/
    - Registers the api blueprint (all routes)
    - Enables CORS for dashboard XHR requests
    """
    dashboard_dir = Path(__file__).parents[2] / "dashboard"

    app = Flask(
        __name__,
        template_folder=str(dashboard_dir / "templates"),
        static_folder=str(dashboard_dir / "static"),
    )
    CORS(app)

    from src.api.routes import api_bp
    app.register_blueprint(api_bp)

    return app
