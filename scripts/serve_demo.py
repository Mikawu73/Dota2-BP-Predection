from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

import torch

from src.inference import DEFAULT_CHECKPOINT, DraftPredictor


WEB_DIR = Path(__file__).resolve().parents[1] / "src" / "web"
DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "reports" / "results.json"
STATIC_FILES = {
    "/": ("overview.html", "text/html; charset=utf-8"),
    "/sandbox": ("index.html", "text/html; charset=utf-8"),
    "/sandbox/": ("index.html", "text/html; charset=utf-8"),
    "/overview.css": ("overview.css", "text/css; charset=utf-8"),
    "/overview.js": ("overview.js", "text/javascript; charset=utf-8"),
    "/report-data.js": ("report-data.js", "text/javascript; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/team-picker.js": ("team-picker.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/lucide.min.js": ("lucide.min.js", "text/javascript; charset=utf-8"),
}


def create_server(
    predictor: DraftPredictor, port: int = 8765, report_path: Path = DEFAULT_REPORT,
) -> ThreadingHTTPServer:
    catalog_path = WEB_DIR / "heroes.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else []
    names = {hero["id"]: hero for hero in catalog}
    metadata = predictor.metadata()
    teams_path = WEB_DIR / "teams.json"
    teams = json.loads(teams_path.read_text(encoding="utf-8")) if teams_path.exists() else []
    known_teams = set(metadata["team_ids"])
    metadata["teams"] = [{**team, "known_to_model": team["id"] in known_teams} for team in teams]
    metadata["heroes"] = [names.get(hero, {"id": hero, "name": f"Hero {hero}", "image": None})
                          for hero in metadata["hero_ids"]]
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' https://cdn.cloudflare.steamstatic.com; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, status: int, value: dict) -> None:
            self.respond(status, json.dumps(value, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def local_request(self) -> bool:
            port = self.server.server_port
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            host = self.headers.get("Host")
            origin = self.headers.get("Origin")
            if host not in allowed_hosts or (origin is not None and origin != f"http://{host}"):
                self.json_response(403, {"error": "Only same-origin local requests are allowed"})
                return False
            return True

        def do_GET(self) -> None:
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            if path == "/api/metadata":
                self.json_response(200, metadata)
            elif path == "/api/report":
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if not isinstance(report, dict) or report.get("schema_version") != 1:
                        raise ValueError("Unsupported report schema")
                    if report.get("evaluation_split") != "validation":
                        raise ValueError("Expected a validation report")
                    self.json_response(200, report)
                except (OSError, ValueError):
                    self.json_response(503, {"error": "Public report is missing or invalid. Regenerate it with scripts.export_portfolio_report."})
            elif path in STATIC_FILES:
                filename, content_type = STATIC_FILES[path]
                self.respond(200, (WEB_DIR / filename).read_bytes(), content_type)
            else:
                self.json_response(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if not self.local_request():
                return
            if self.path != "/api/predict":
                self.json_response(404, {"error": "Not found"})
                return
            if self.headers.get_content_type() != "application/json":
                self.json_response(415, {"error": "Content-Type must be application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16_384:
                    self.json_response(413, {"error": "Request must be between 1 and 16384 bytes"})
                    return
                state = json.loads(self.rfile.read(length))
                with lock:
                    result = predictor.predict(state)
                self.json_response(200, result)
            except (ValueError, UnicodeError) as exc:
                self.json_response(400, {"error": str(exc)})
            except Exception:
                self.log_error("Inference failed")
                self.json_response(500, {"error": "Inference failed; check the server"})

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(10)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a loopback-only BP research demo (not a production server)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Public aggregate results JSON for the homepage")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        predictor = DraftPredictor(args.checkpoint)
        server = create_server(predictor, args.port, args.report)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Cannot start demo: {exc}\n")
    print(f"Project overview: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"BP sandbox: http://127.0.0.1:{server.server_port}/sandbox", flush=True)
    print(f"Checkpoint: {predictor.checkpoint_hash[:12]} | epoch {predictor.epoch}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
