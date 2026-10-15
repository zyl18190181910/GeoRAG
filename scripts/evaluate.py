import argparse
import json
from pathlib import Path

from geo_rag.config import Settings
from geo_rag.evaluation import evaluate, load_eval_cases
from geo_rag.service import RagService


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and refusal behavior.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "evaluation_latest.json",
        help="Path for the JSON report.",
    )
    args = parser.parse_args()
    service = RagService(Settings(storage_path=ROOT / "data" / "storage" / "knowledge.json"))
    if not service.chunks:
        raise SystemExit("knowledge store is empty; run scripts/seed_demo.py first")
    cases = load_eval_cases(ROOT / "data" / "eval" / "questions.jsonl")
    result = evaluate(service, cases)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()

