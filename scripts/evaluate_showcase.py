import json
from pathlib import Path

from geo_rag.config import Settings
from geo_rag.showcase import run_showcase_evaluation


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    report = run_showcase_evaluation(Settings.from_env(), ROOT)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {ROOT / 'reports' / 'showcase_latest.json'}")
