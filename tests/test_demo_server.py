from io import BytesIO
import json

from scripts.serve_demo import create_server, STATIC_FILES, WEB_DIR
from src.inference import DraftPredictor
from tests.test_inference import checkpoint, state


class Connection:
    """Exercise the real HTTP handler without requiring a listening socket in CI."""

    def __init__(self, request):
        self.input = BytesIO(request)
        self.output = bytearray()

    def makefile(self, *args):
        return self.input

    def sendall(self, value):
        self.output.extend(value)

    def settimeout(self, value):
        pass


def test_http_routes_and_local_boundary(checkpoint, state, monkeypatch, tmp_path):
    # Avoid binding a socket; the handler still parses actual HTTP requests.
    from http.server import ThreadingHTTPServer
    original = ThreadingHTTPServer.__init__

    def unbound(self, address, handler):
        original(self, address, handler, bind_and_activate=False)
        self.server_port = address[1]

    monkeypatch.setattr(ThreadingHTTPServer, "__init__", unbound)
    report_path = tmp_path / "report.json"
    public_report = {"schema_version": 1, "evaluation_split": "validation", "dataset": {}, "runs": []}
    report_path.write_text(json.dumps(public_report), encoding="utf-8")
    server = create_server(DraftPredictor(checkpoint), 8765, report_path)

    def request(method, path, body=None, extra="", host="127.0.0.1:8765", content_type="application/json"):
        payload = json.dumps(body).encode() if body is not None else b""
        data = (f"{method} {path} HTTP/1.0\r\nHost: {host}\r\nContent-Type: {content_type}\r\n"
                f"Content-Length: {len(payload)}\r\n{extra}\r\n").encode() + payload
        connection = Connection(data)
        server.RequestHandlerClass(connection, ("127.0.0.1", 9999), server)
        headers, response = bytes(connection.output).split(b"\r\n\r\n", 1)
        return int(headers.split()[1]), headers, response

    try:
        for path in STATIC_FILES:
            code, headers, content = request("GET", path)
            assert code == 200 and content
            assert b"Content-Security-Policy" in headers
        assert b"Dota 2 Draft Prediction" in request("GET", "/")[2]
        assert b'href="/sandbox"' in request("GET", "/")[2]
        assert b"Draft sandbox" in request("GET", "/sandbox")[2]
        assert b'href="/"' in request("GET", "/sandbox")[2]
        code, _, body = request("GET", "/api/report")
        assert code == 200 and json.loads(body) == public_report
        code, _, body = request("GET", "/api/metadata")
        assert code == 200
        assert len(json.loads(body)["heroes"]) == 30
        teams = json.loads(body)["teams"]
        assert isinstance(teams, list)
        assert all(team["known_to_model"] == (team["id"] in {100, 200}) for team in teams)
        code, _, body = request("POST", "/api/predict", state)
        assert code == 200 and len(json.loads(body)["predictions"]) == 5
        assert request("GET", "/../../.env")[0] == 404
        assert request("POST", "/api/predict", {})[0] == 400
        assert request("POST", "/api/predict", state, content_type="text/plain")[0] == 415
        assert request("POST", "/api/predict", {"padding": "x" * 17000})[0] == 413
        assert request("POST", "/api/predict", state, extra="Origin: https://example.com\r\n")[0] == 403
        assert request("GET", "/api/metadata", host="evil.example:8765")[0] == 403
        assert request("GET", "/api/report", host="evil.example:8765")[0] == 403
        report_path.write_text("not JSON", encoding="utf-8")
        assert request("GET", "/api/report")[0] == 503
        report_path.write_text('{"schema_version": 1, "evaluation_split": "test"}', encoding="utf-8")
        assert request("GET", "/api/report")[0] == 503
        report_path.unlink()
        assert request("GET", "/api/report")[0] == 503
        # A missing report must not break the existing sandbox's prediction route.
        assert request("POST", "/api/predict", state)[0] == 200
    finally:
        server.server_close()


def test_assets_are_local_and_catalog_is_display_only():
    catalog = json.loads((WEB_DIR / "heroes.json").read_text(encoding="utf-8"))
    assert len({hero["id"] for hero in catalog}) == len(catalog)
    assert all(set(hero) == {"id", "name", "image"} for hero in catalog)
    assert all(hero["image"].startswith("https://cdn.cloudflare.steamstatic.com/") for hero in catalog)
    assert (WEB_DIR / "LUCIDE-LICENSE").is_file()
