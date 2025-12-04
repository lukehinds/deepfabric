#!/usr/bin/env python3
"""Simple HTTP server that logs request bodies for testing DeepFabric metrics.

Usage:
    python examples/mock_metrics_server.py [port]

Default port is 8888.
"""

from __future__ import annotations

import json
import sys

from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler that logs POST request bodies."""

    def log_message(self, format, *args):  # noqa: A002
        """Override to add timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {args[0]}")

    def do_POST(self):
        """Handle POST requests and log the body."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Parse and pretty-print JSON
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n{'='*60}")
        print(f"[{timestamp}] POST {self.path}")
        print(f"{'='*60}")

        # Log headers
        auth = self.headers.get("Authorization", "")
        if auth and auth.startswith("Bearer "):
            # Mask the API key
            masked = auth[:12] + "..." + auth[-4:] if len(auth) > 20 else auth
            print(f"Authorization: {masked}")

        # Parse and pretty-print body
        try:
            data = json.loads(body)
            print(f"\nPayload ({self.path}):")
            print(json.dumps(data, indent=2))
        except json.JSONDecodeError:
            print(f"\nRaw body: {body.decode('utf-8', errors='replace')}")

        print(f"{'='*60}\n")

        # Send success response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {"status": "ok", "message": "Metrics received"}
        self.wfile.write(json.dumps(response).encode())

    def do_GET(self):
        """Handle GET requests (health check)."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {
            "status": "ok",
            "service": "DeepFabric Mock Metrics Server",
            "endpoints": [
                "POST /v1/training/metrics - Receive training metrics",
                "POST /v1/training/runs - Receive run events",
            ],
        }
        self.wfile.write(json.dumps(response, indent=2).encode())


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888

    server = HTTPServer(("", port), MetricsHandler)

    print(f"""
{'='*60}
  DeepFabric Mock Metrics Server
{'='*60}

  Listening on: http://localhost:{port}

  Expected endpoints:
    POST /v1/training/metrics  - Training metrics batches
    POST /v1/training/runs     - Run start/end events

  To use with DeepFabric:
    export DEEPFABRIC_API_URL=http://localhost:{port}
    export DEEPFABRIC_API_KEY=test-key

  Press Ctrl+C to stop.
{'='*60}
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()
