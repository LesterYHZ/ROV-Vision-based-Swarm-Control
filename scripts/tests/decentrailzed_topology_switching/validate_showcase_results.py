"""Assert the curated no-holdover dynamic-switching showcase criteria."""
from __future__ import annotations
import csv
import sys
from pathlib import Path


def read_metrics(path: Path):
    with path.open(newline="") as f:
        return {row[0]: row[1] for row in list(csv.reader(f))[1:]}


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    m = read_metrics(folder / "simulation_metrics.csv")
    checks = {
        "all three followers have zero holdover": m.get("all_followers_zero_holdover", "False") == "True",
        "no holdover samples": int(float(m["total_holdover_samples"])) == 0,
        "all selected links physically visible": m.get("all_selected_links_physically_visible", "False") == "True",
        "no collision samples": int(float(m["collision_samples"])) == 0,
        "all CBF QPs feasible": m.get("all_cbfs_feasible", "False") == "True",
        "F1 selected F2 throughout leader-link dropout": float(m["f1_selected_f2_fraction_during_dropout"]) >= 0.999,
        "F2 independently retained the leader": float(m["f2_selected_leader_fraction_during_dropout"]) >= 0.999,
        "at least six parent switches after initialization": int(float(m["parent_switches_after_initial_assignment"])) >= 6,
        "no inter-ROV messages": int(float(m["inter_agent_messages_sent"])) == 0,
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if not all(checks.values()):
        raise SystemExit("Showcase validation failed")
    print("\nAll showcase criteria passed.")


if __name__ == "__main__":
    main()
