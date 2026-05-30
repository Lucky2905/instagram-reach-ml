"""
app.py — Flask server entry point (project root).

Usage:
    python app.py
    python app.py --port 5001 --no-debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from config import API_HOST, API_PORT, API_DEBUG
from src.api.app import create_app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instagram Reach ML — Flask API Server")
    p.add_argument("--host",     default=API_HOST)
    p.add_argument("--port",     type=int, default=API_PORT)
    p.add_argument("--no-debug", action="store_true", dest="no_debug")
    return p.parse_args()


def print_startup_banner(host: str, port: int) -> None:
    url = f"http://localhost:{port}"
    print(f"""
+--------------------------------------------------------------+
|       Instagram Reach ML -- Flask API & Dashboard            |
+--------------------------------------------------------------+
|  Dashboard  ->  {url:<44}  |
|  Health     ->  {url + "/health":<44}  |
|  Predict    ->  POST {url + "/predict":<40}  |
|  Metrics    ->  {url + "/metrics":<44}  |
+--------------------------------------------------------------+
""")


if __name__ == "__main__":
    args = parse_args()
    app = create_app()
    print_startup_banner(args.host, args.port)
    app.run(host=args.host, port=args.port, debug=not args.no_debug)
