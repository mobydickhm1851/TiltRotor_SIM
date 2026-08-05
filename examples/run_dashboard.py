"""Launch the real-time Bokeh dashboard."""
from __future__ import annotations

import os
import webbrowser

from bokeh.server.server import Server
from tornado.ioloop import IOLoop

from rotorpy_tiltrotor.dashboard import build_dashboard


def app(doc):
    build_dashboard(doc)


def main() -> None:
    port = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", "8050")))
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    default_origins = f"127.0.0.1:{port},localhost:{port}"
    origins = [item.strip() for item in os.getenv("BOKEH_ALLOW_WS_ORIGIN", default_origins).split(",") if item.strip()]
    loop = IOLoop.current()
    server = Server({"/": app}, io_loop=loop, port=port, address=host, allow_websocket_origin=origins)
    server.start()
    local_url = f"http://127.0.0.1:{port}/"
    print(f"Dashboard server listening on {host}:{port}")
    print(f"Local URL: {local_url}")
    if origins == ["*"]:
        print("WARNING: wildcard WebSocket origin is intended only for development/testing.")
    if os.getenv("DASHBOARD_NO_BROWSER", "0") not in {"1", "true", "TRUE"} and host in {"127.0.0.1", "localhost"}:
        loop.add_callback(lambda: webbrowser.open(local_url))
    loop.start()


if __name__ == "__main__":
    main()
