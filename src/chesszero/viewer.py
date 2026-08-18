"""A tiny, dependency-free browser viewer for saved self-play games.

Serves the packaged ``viewer/index.html`` plus a ``manifest.json`` describing the
games in a directory. The HTML is vanilla JS (no CDN, works offline) and renders
the board from FEN, steps through moves, and shows the per-move softmax
evaluation captured during self-play.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import webbrowser
from pathlib import Path

VIEWER_HTML = Path(__file__).parent / "viewer" / "index.html"


def _build_manifest(games_dir: Path) -> list[dict]:
    games: list[dict] = []
    for path in sorted(games_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        games.append({
            "file": path.name,
            "game_index": data.get("game_index"),
            "iteration": data.get("iteration"),
            "result": data.get("result_str", data.get("result")),
            "termination": data.get("termination"),
            "num_plies": data.get("num_plies", len(data.get("moves", []))),
        })
    games.sort(key=lambda g: (g["game_index"] if g["game_index"] is not None else 0))
    return games


def run_viewer(games_dir: str | Path = "games", port: int = 8000,
               open_browser: bool = True) -> None:
    games_dir = Path(games_dir).resolve()
    if not games_dir.exists():
        raise SystemExit(
            f"Games directory not found: {games_dir}\n"
            "Run training with --sample-games first, e.g.\n"
            "  poetry run chesszero loop --iterations 4 --games 5 --sample-games 10"
        )
    html_bytes = VIEWER_HTML.read_bytes()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, directory=str(games_dir), **kwargs)

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            if self.path in ("/", "/index.html"):
                self._send(html_bytes, "text/html; charset=utf-8")
            elif self.path == "/manifest.json":
                # Rebuilt each request so new games appear on refresh.
                body = json.dumps(_build_manifest(games_dir)).encode()
                self._send(body, "application/json")
            else:
                super().do_GET()

        def log_message(self, *args) -> None:  # silence per-request logging
            pass

    manifest = _build_manifest(games_dir)
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"Serving {len(manifest)} game(s) from {games_dir}")
        print(f"Viewer running at {url}  (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
