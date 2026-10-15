from __future__ import annotations

import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "public_papers"


def main() -> None:
    manifest_path = TARGET / "manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    TARGET.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        target = TARGET / entry["filename"]
        if target.exists() and target.read_bytes()[:4] == b"%PDF":
            print(f"exists: {target.name}")
            continue
        request = urllib.request.Request(
            entry["url"],
            headers={"User-Agent": "Mozilla/5.0 GeoRAG-Evaluation/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()
        if not content.startswith(b"%PDF"):
            raise ValueError(f"download is not a PDF: {entry['url']}")
        target.write_bytes(content)
        print(f"downloaded: {target.name} ({len(content) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
