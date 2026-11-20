from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sustai_energy.study import StudyConfig, run_study  # noqa: E402


if __name__ == "__main__":
    summary = run_study(StudyConfig(root=ROOT))
    print("Study complete.")
    print(f"Selected buildings: {', '.join(summary['selected_buildings'])}")
    print(f"Report: {ROOT / 'reports' / 'results.md'}")

