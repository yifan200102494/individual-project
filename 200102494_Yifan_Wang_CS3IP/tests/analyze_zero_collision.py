"""
Report the strict zero-collision success rate from benchmark_raw_trials.csv.

A trial counts as a success iff:
    placed == 1  AND  collisions == 0  AND  time_taken < 45 s

This is the N=0 (true zero-tolerance) criterion that the threshold-sensitivity
script could not express via its `collisions < N` form (since no non-negative
integer is < 0). Re-uses the existing raw data, so no simulation is re-run.
"""

import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "result")
SRC = os.path.join(RESULTS_DIR, "benchmark_raw_trials_zero_2.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "zero_collision_success.csv")

TIMEOUT = 60.0  # Matches the convention used in benchmark_raw_zero.py and paper §6.2
SPEED_LABELS = {0.001: "1-Slow",
                0.003: "2-Medium",
                0.005: "3-Fast",
                0.007: "4-Extreme",
                0.017: "5-Insane"}


def is_strict_success(row):
    placed = int(row["placed"]) == 1
    zero_coll = int(row["collisions"]) == 0
    on_time = float(row["time_taken"]) < TIMEOUT
    return placed and zero_coll and on_time


def main():
    rows = []
    with open(SRC, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    speeds = sorted({float(r["speed"]) for r in rows})
    by_speed = {s: [r for r in rows if float(r["speed"]) == s] for s in speeds}

    print("\nStrict zero-collision success rate")
    print(f"(placed AND collisions == 0 AND time < {TIMEOUT:.0f} s)")
    print("-" * 60)
    print(f"{'Speed':<12}{'Trials':>8}{'Succ':>8}{'Rate (%)':>12}")
    print("-" * 60)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speed", "trials", "successes", "rate_pct"])

        total_trials = 0
        total_succ = 0
        for s in speeds:
            trials = by_speed[s]
            succ = sum(1 for r in trials if is_strict_success(r))
            rate = succ / len(trials) * 100
            total_trials += len(trials)
            total_succ += succ
            name = SPEED_LABELS.get(s, f"{s}")
            print(f"{name:<12}{len(trials):>8}{succ:>8}{rate:>11.1f}")
            w.writerow([name, len(trials), succ, f"{rate:.1f}"])

        mean_rate = total_succ / total_trials * 100
        w.writerow(["mean", total_trials, total_succ, f"{mean_rate:.1f}"])
        print("-" * 60)
        print(f"{'mean':<12}{total_trials:>8}{total_succ:>8}{mean_rate:>11.1f}")

    print(f"\n[saved] {OUT_CSV}")


if __name__ == "__main__":
    main()
