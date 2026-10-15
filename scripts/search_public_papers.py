"""Inspect extracted public-paper text while authoring evidence-level evaluations."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from geo_rag.loaders import extract_sections_from_bytes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="substring of a PDF filename")
    parser.add_argument("pattern", nargs="?", default=".", help="regular expression")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    papers = list((ROOT / "data" / "public_papers").glob(f"*{args.source}*.pdf"))
    if len(papers) != 1:
        raise SystemExit(f"expected one matching PDF, found {len(papers)}")

    regex = re.compile(args.pattern, re.IGNORECASE)
    shown = 0
    for section in extract_sections_from_bytes(papers[0].name, papers[0].read_bytes()):
        for match in regex.finditer(section.text):
            start = max(0, match.start() - 180)
            end = min(len(section.text), match.end() + 260)
            excerpt = " ".join(section.text[start:end].split())
            print(f"page={section.metadata.get('page', '?')} {excerpt}")
            shown += 1
            if shown >= args.limit:
                return

    if not shown:
        raise SystemExit("no matches")


if __name__ == "__main__":
    main()
