from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import pytest

from scripts.build_pages import OUTPUT_FILES, ROOT, build_pages, static_html


class Links(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.links = []
        self.report_url = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for key in ("href", "src"):
            if values.get(key):
                self.links.append(values[key])
        if tag == "body":
            self.report_url = values.get("data-report-url")


def test_pages_output_links_work_at_repository_subpath(tmp_path):
    out = build_pages(tmp_path / "pages")
    assert {p.name for p in out.iterdir()} == OUTPUT_FILES
    assert json.loads((out / "results.json").read_text()) == json.loads((ROOT / "reports/results.json").read_text())
    for filename in ("index.html", "sandbox.html"):
        html = (out / filename).read_text(encoding="utf-8")
        parsed = Links(html)
        base = f"https://example.github.io/Dota2-BP-Predection/{filename}"
        for link in parsed.links + [parsed.report_url]:
            if urlsplit(link).scheme:
                continue
            assert not link.startswith("/"), link
            resolved = urlsplit(urljoin(base, link)).path
            assert resolved.startswith("/Dota2-BP-Predection/")
            target = resolved.removeprefix("/Dota2-BP-Predection/") or "index.html"
            assert (out / target).is_file(), target
    homepage = (out / "index.html").read_text(encoding="utf-8")
    assert "Dota 2 Draft Prediction" in homepage
    assert Links(homepage).report_url == "./results.json"
    assert "./sandbox.html" in homepage
    assert "weights are not included" in (out / "sandbox.html").read_text()
    assert not any(p.name in OUTPUT_FILES for p in [Path("app.js"), Path("teams.json"), Path("model_b_best.pt"), Path(".env")])
    # Building again is deterministic and does not mutate local-server source paths.
    original = (out / "index.html").read_bytes()
    build_pages(out)
    assert (out / "index.html").read_bytes() == original
    assert 'href="/sandbox"' in (ROOT / "src/web/overview.html").read_text()


def test_html_conversion_preserves_entities_and_external_links():
    source = '<!doctype html><body><a href="/sandbox" title="A &amp; B">A &amp; B &#42;</a><a href="https://example.com">External</a><br/></body>'
    result = static_html(source)
    assert 'href="./sandbox.html"' in result
    assert 'title="A &amp; B"' in result
    assert "A &amp; B &#42;" in result
    assert 'href="https://example.com"' in result


def test_refuses_to_publish_unexpected_files(tmp_path):
    out = tmp_path / "pages"
    out.mkdir()
    secret = out / ".env"
    secret.write_text("must not be published")
    with pytest.raises(ValueError, match="unexpected files"):
        build_pages(out)
    assert secret.read_text() == "must not be published"
    assert not (out / "index.html").exists()


def test_invalid_report_stops_build_before_writing(tmp_path):
    root = tmp_path / "source"
    (root / "reports").mkdir(parents=True)
    (root / "reports/results.json").write_text('{"schema_version": 1, "evaluation_split": "test"}')
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="validation report"):
        build_pages(output, root)
    assert not output.exists()
