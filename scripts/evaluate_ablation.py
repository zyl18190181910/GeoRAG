import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geo_rag.ablation import run_ablation_evaluation  # noqa: E402
from geo_rag.config import Settings  # noqa: E402


def main() -> None:
    report = run_ablation_evaluation(Settings.from_env(), ROOT)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {ROOT / 'reports' / 'ablation_latest.json'}")


if __name__ == "__main__":
    main()
