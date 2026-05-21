#!/usr/bin/env python3
"""
Regenerate stakeholder HTML from ablation_results.json.

  python scripts/run_retrieval_ablation_offline.py
  python scripts/refresh_stakeholder_views.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "evaluation" / "reports" / "ablation_results.json"


def main() -> int:
    if not JSON_PATH.is_file():
        print("Running offline ablation first...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "run_retrieval_ablation_offline.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_ablation_stakeholder_report.py")], check=True)
    html = ROOT / "evaluation" / "reports" / "stakeholder_dashboard.html"
    print(f"Stakeholder HTML: {html}")
    print("Cursor canvas: documind-retrieval-scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
