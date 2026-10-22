from __future__ import annotations

import argparse
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    "style.css", "overview.css", "overview.js", "report-data.js",
    "lucide.min.js", "LUCIDE-LICENSE",
)
OUTPUT_FILES = set(ASSETS) | {"index.html", "sandbox.html", "results.json", ".nojekyll"}


class PagesHTML(HTMLParser):
    """Preserve the shared page while making links work under any Pages base path."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []

    def start(self, tag: str, attrs: list, closed: bool = False) -> None:
        attributes = dict(attrs)
        for key in ("href", "src"):
            value = attributes.get(key)
            if value == "/sandbox":
                attributes[key] = "./sandbox.html"
            elif value and value.startswith("/") and not value.startswith("//"):
                attributes[key] = "." + value
        if tag == "body":
            attributes["data-report-url"] = "./results.json"
        encoded = "".join(f' {key}' if value is None else f' {key}="{escape(value, quote=True)}"'
                          for key, value in attributes.items())
        self.parts.append(f"<{tag}{encoded}{' /' if closed else ''}>")

    def handle_starttag(self, tag, attrs):
        self.start(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self.start(tag, attrs, closed=True)

    def handle_endtag(self, tag):
        self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        self.parts.append(data)

    def handle_decl(self, decl):
        self.parts.append(f"<!{decl}>")

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")

    def handle_comment(self, data):
        self.parts.append(f"<!--{data}-->")


def static_html(source: str) -> str:
    parser = PagesHTML()
    parser.feed(source)
    parser.close()
    return "".join(parser.parts)


def build_pages(output_dir: Path, root: Path = ROOT) -> Path:
    web = root / "src" / "web"
    report_path = root / "reports" / "results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("evaluation_split") != "validation":
        raise ValueError("Expected a version-1 public validation report")
    if not isinstance(report.get("dataset"), dict) or not isinstance(report.get("runs"), list):
        raise ValueError("Public report is missing dataset or runs")
    for filename in (*ASSETS, "overview.html", "sandbox-static.html"):
        if not (web / filename).is_file():
            raise FileNotFoundError(f"Missing site asset: {filename}")
    if output_dir.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    if output_dir.exists():
        # Do not upload leftovers or recursively delete an arbitrary directory.
        unexpected = [p.name for p in output_dir.iterdir()
                      if p.name not in OUTPUT_FILES or not p.is_file() or p.is_symlink()]
        if unexpected:
            raise ValueError(f"Output contains unexpected files; choose a dedicated empty directory: {unexpected}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ASSETS:
        shutil.copyfile(web / filename, output_dir / filename)
    for source, destination in (("overview.html", "index.html"), ("sandbox-static.html", "sandbox.html")):
        page = static_html((web / source).read_text(encoding="utf-8"))
        (output_dir / destination).write_text(page, encoding="utf-8")
    shutil.copyfile(report_path, output_dir / "results.json")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static research overview for GitHub Pages; no model or raw data required")
    parser.add_argument("--output-dir", type=Path, default=Path("dist/pages"))
    args = parser.parse_args()
    try:
        output = build_pages(args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Pages build failed: {exc}\n")
    print(f"Built {len(OUTPUT_FILES)} public files in {output}")


if __name__ == "__main__":
    main()
