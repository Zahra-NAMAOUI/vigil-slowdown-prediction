#!/usr/bin/env python3
"""Pick one real slowdown episode out of the held-out test split for replay mode.

The demo only tells a story if the model first sees a long stretch where nothing
is coming. The model looks five minutes ahead, so the lead-in must be
continuously negative for the warm-up (120s) PLUS the whole prediction horizon
(300s) before the episode arrives.

Nothing is exported on trust: the shortlisted candidate is replayed offline
through the real pipeline — live_features recomputes the 223 features, the
frozen preprocessor and LightGBM V1 score them — and the resulting curve has to
start low and actually climb. If it does not, nothing is written.

Exports ONLY the raw collector columns: the engineered features and the label
never leave this script.

    python tools/build_replay_segment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.inference import load_inference_engine  # noqa: E402
from dashboard.live_features import (  # noqa: E402
    RAW_CANDIDATE_FEATURES, engineer_feature_history,
)

SOURCE_PATH = PROJECT_ROOT / "data/modeling/test.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/replay/replay_segment.csv"

TARGET = "slowdown_in_5min"
IDENTIFIERS = ["machine_id", "run_id", "segment_id", "timestamp"]
EXPORT_COLUMNS = ["machine_id", "run_id", "timestamp", *RAW_CANDIDATE_FEATURES]

# ── selection criteria ──────────────────────────────────────────────────────
MAX_GAP_SECONDS = 10.0        # the live gap rule; anything longer restarts warm-up
MIN_CALM_LEAD_SECONDS = 420.0 # 120s warm-up + 300s prediction horizon, target == 0 throughout
MIN_SUSTAINED_SECONDS = 60.0  # the transition must hold, not blip
MIN_TAIL_SECONDS = 120.0      # keep watching after it fires
MIN_ROWS, MAX_ROWS = 600, 900

# ── verification thresholds ─────────────────────────────────────────────────
REPLAY_INTERVAL_SECONDS = 2.0 # the cadence dashboard/replay.py plays rows back at
WARMUP_SECONDS = 120.0
SCORE_EVERY_ROWS = 5          # ~10s of replay time between scored points
PRINT_EVERY_SECONDS = 30.0
START_BELOW = 30.0            # first post-warm-up score
PEAK_ABOVE = 70.0             # somewhere around the transition
PEAK_WINDOW_BEFORE, PEAK_WINDOW_AFTER = 60.0, 300.0


# ════════════════════════════════════════════════════════════════════════════
# selection
# ════════════════════════════════════════════════════════════════════════════

def load_source() -> pd.DataFrame:
    if not SOURCE_PATH.exists():
        raise SystemExit(f"Source split is missing: {SOURCE_PATH}")
    frame = pd.read_csv(
        SOURCE_PATH,
        usecols=[*IDENTIFIERS, TARGET, "valid_5min_horizon", *RAW_CANDIDATE_FEATURES],
    )
    frame["_ts"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame.dropna(subset=["_ts"])


def continuous_blocks(frame: pd.DataFrame):
    """Same machine + run + segment, re-split wherever a real gap exceeds 10s.

    The modelling pipeline dropped rows (invalid horizon, insufficient history),
    so a segment_id in test.csv is not necessarily contiguous in wall-clock time.
    """
    for keys, group in frame.groupby(["machine_id", "run_id", "segment_id"], sort=False):
        ordered = group.sort_values("_ts").reset_index(drop=True)
        deltas = ordered["_ts"].diff().dt.total_seconds()
        block_number = deltas.gt(MAX_GAP_SECONDS).fillna(False).cumsum()
        for _, block in ordered.groupby(block_number, sort=False):
            yield keys, block.reset_index(drop=True)


def calm_lead_start(target: np.ndarray, position: int) -> int:
    """First index of the uninterrupted run of zeros ending just before `position`."""
    index = position - 1
    while index >= 0 and target[index] == 0:
        index -= 1
    return index + 1


def sustained_end(target: np.ndarray, position: int) -> int:
    """Last index of the uninterrupted run of ones starting at `position`."""
    index = position
    while index < len(target) and target[index] == 1:
        index += 1
    return index - 1


def find_candidates(frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    candidates, rejected = [], []

    for keys, block in continuous_blocks(frame):
        target = block[TARGET].to_numpy()
        seconds = (block["_ts"] - block["_ts"].iloc[0]).dt.total_seconds().to_numpy()
        rows = len(block)

        flips = [i for i in range(1, rows) if target[i - 1] == 0 and target[i] == 1]
        for position in flips:
            zero_start = calm_lead_start(target, position)
            calm_lead = float(seconds[position] - seconds[zero_start])
            hold_end = sustained_end(target, position)
            sustained = float(seconds[hold_end] - seconds[position])
            tail = float(seconds[-1] - seconds[position])

            note = {
                "run_id": str(keys[1])[:13], "machine_id": str(keys[0])[:12],
                "calm_lead_s": round(calm_lead), "sustained_s": round(sustained),
                "tail_s": round(tail), "block_rows": rows,
            }
            if calm_lead < MIN_CALM_LEAD_SECONDS:
                rejected.append(note | {"why": f"calm lead {calm_lead:.0f}s < {MIN_CALM_LEAD_SECONDS:.0f}s"})
                continue
            if sustained < MIN_SUSTAINED_SECONDS:
                rejected.append(note | {"why": f"sustained {sustained:.0f}s < {MIN_SUSTAINED_SECONDS:.0f}s"})
                continue
            if tail < MIN_TAIL_SECONDS:
                rejected.append(note | {"why": f"tail {tail:.0f}s < {MIN_TAIL_SECONDS:.0f}s"})
                continue

            window = fit_window(seconds, rows, position, zero_start)
            if window is None:
                rejected.append(note | {"why": f"no {MIN_ROWS}-{MAX_ROWS} row window fits"})
                continue

            start, end = window
            selection = block.iloc[start:end + 1].reset_index(drop=True)
            local = position - start
            lead = selection.iloc[:local]
            deltas = selection["_ts"].diff().dt.total_seconds().dropna()
            candidates.append({
                "keys": keys, "selection": selection, "transition_row": local,
                "rows": len(selection),
                "duration_s": round(float(seconds[end] - seconds[start]), 1),
                "calm_lead_s": round(float(seconds[position] - seconds[start]), 1),
                "sustained_s": round(sustained, 1),
                "tail_s": round(float(seconds[end] - seconds[position]), 1),
                "median_dt": round(float(deltas.median()), 2),
                "max_dt": round(float(deltas.max()), 2),
                "lead_cpu": round(float(lead["cpu_pct"].mean()), 2),
                "lead_swap": round(float(lead["swap_pct"].mean()), 2),
            })

    # Ranked by how calm the lead-in is: quietest CPU and swap first.
    for item in candidates:
        item["calmness"] = round((item["lead_cpu"] + item["lead_swap"]) / 2, 3)
    candidates.sort(key=lambda item: item["calmness"])
    return candidates, rejected


def fit_window(seconds: np.ndarray, rows: int, position: int, zero_start: int):
    """Smallest 600-900 row window whose lead-in stays inside the all-zero stretch.

    Shortest-first keeps the replay loop tight (600 rows = 20 minutes at 2s).
    The 420s calm-lead floor is enforced independently, so trimming the window
    can never eat into the run-up the model needs.
    """
    lead_ok = [i for i in range(zero_start, position + 1)
               if seconds[position] - seconds[i] >= MIN_CALM_LEAD_SECONDS]
    tail_ok = [j for j in range(position, rows)
               if seconds[j] - seconds[position] >= MIN_TAIL_SECONDS]
    if not lead_ok or not tail_ok:
        return None
    latest_start, earliest_end = max(lead_ok), min(tail_ok)

    for total in range(MIN_ROWS, MAX_ROWS + 1):              # prefer the tightest loop
        for start in range(min(latest_start, rows - total), zero_start - 1, -1):
            end = start + total - 1
            if end < earliest_end or end > rows - 1:
                continue
            return start, end
    return None


# ════════════════════════════════════════════════════════════════════════════
# verification: the real pipeline, offline
# ════════════════════════════════════════════════════════════════════════════

def replay_timeline(selection: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the frame exactly as dashboard/replay.py will write it.

    Replay rewrites every timestamp to a uniform 2s cadence, so the rolling
    windows the model will actually see are the ones built on THIS timeline,
    not on the recording's original spacing.
    """
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = selection[["machine_id", "run_id", *RAW_CANDIDATE_FEATURES]].copy()
    offsets = np.arange(len(frame)) * REPLAY_INTERVAL_SECONDS
    frame["timestamp"] = [(base + pd.Timedelta(seconds=float(s))).isoformat() for s in offsets]
    frame["segment_id"] = "replay-verify"        # one segment, as a replay run is
    return frame, offsets


def score_curve(selection: pd.DataFrame, engine) -> pd.DataFrame:
    """Score the candidate through live_features + the frozen artifacts."""
    frame, offsets = replay_timeline(selection)
    engineered = engineer_feature_history(frame)
    names = list(engine.input_feature_names)

    points = []
    for index in range(0, len(engineered), SCORE_EVERY_ROWS):
        elapsed = float(offsets[index])
        if elapsed < WARMUP_SECONDS:             # the live warm-up gate, honoured here too
            points.append({"elapsed_s": elapsed, "score": None})
            continue
        row = engineered.iloc[[index]][names]
        result = engine.predict(row)             # the same call the dashboard makes
        points.append({"elapsed_s": elapsed, "score": round(result.risk_score, 1)})
    return pd.DataFrame(points)


def assess(curve: pd.DataFrame, transition_elapsed: float) -> dict[str, object]:
    scored = curve.dropna(subset=["score"])
    if scored.empty:
        return {"passes": False, "reason": "no scored point after warm-up",
                "first_score": None, "peak_near_transition": None}
    first = float(scored["score"].iloc[0])
    near = scored[
        scored["elapsed_s"].between(transition_elapsed - PEAK_WINDOW_BEFORE,
                                    transition_elapsed + PEAK_WINDOW_AFTER)
    ]
    peak = float(near["score"].max()) if not near.empty else float("nan")
    peak_at = float(near.loc[near["score"].idxmax(), "elapsed_s"]) if not near.empty else float("nan")
    starts_low, rises = first < START_BELOW, peak > PEAK_ABOVE
    reasons = []
    if not starts_low:
        reasons.append(f"first score {first:.1f} is not below {START_BELOW:.0f}")
    if not rises:
        reasons.append(f"peak near the transition {peak:.1f} does not exceed {PEAK_ABOVE:.0f}")
    return {
        "passes": starts_low and rises, "reason": "; ".join(reasons) or "starts low and rises",
        "first_score": round(first, 1), "peak_near_transition": round(peak, 1),
        "peak_at_s": round(peak_at, 1),
        "min_score": round(float(scored["score"].min()), 1),
        "max_score": round(float(scored["score"].max()), 1),
    }


def print_curve(curve: pd.DataFrame, transition_elapsed: float) -> None:
    print(f"    {'t':>7}  {'score':>6}   (transition at {transition_elapsed:.0f}s)")
    step = max(1, int(round(PRINT_EVERY_SECONDS / (SCORE_EVERY_ROWS * REPLAY_INTERVAL_SECONDS))))
    for _, point in curve.iloc[::step].iterrows():
        elapsed, score = point["elapsed_s"], point["score"]
        if score is None or pd.isna(score):
            print(f"    {elapsed:6.0f}s  {'—':>6}   warm-up")
            continue
        bar = "#" * int(round(float(score) / 2.5))
        marker = "  <-- transition" if abs(elapsed - transition_elapsed) < PRINT_EVERY_SECONDS / 2 else ""
        print(f"    {elapsed:6.0f}s  {score:6.1f}   {bar}{marker}")


def describe(item: dict) -> None:
    print(f"    source run_id     : {item['keys'][1]}")
    print(f"    source machine_id : {item['keys'][0]}")
    replay_loop = item["rows"] * REPLAY_INTERVAL_SECONDS
    print(f"    rows              : {item['rows']}")
    print(f"    source recording  : {item['duration_s']:.0f}s "
          f"({item['duration_s'] / 60:.1f} min) at the original spacing")
    print(f"    replay loop       : {replay_loop:.0f}s "
          f"({replay_loop / 60:.1f} min) at {REPLAY_INTERVAL_SECONDS:.0f}s per row")
    print(f"    calm lead (0)     : {item['calm_lead_s']:.0f}s   "
          f"(need >= {MIN_CALM_LEAD_SECONDS:.0f}s)")
    print(f"    sustained (1)     : {item['sustained_s']:.0f}s   "
          f"(need >= {MIN_SUSTAINED_SECONDS:.0f}s)")
    print(f"    tail after        : {item['tail_s']:.0f}s")
    print(f"    lead-in calmness  : cpu {item['lead_cpu']:.1f}%  swap {item['lead_swap']:.1f}%")
    print(f"    native cadence    : {item['median_dt']:.2f}s median, {item['max_dt']:.2f}s worst "
          f"(replayed at {REPLAY_INTERVAL_SECONDS:.0f}s)")


# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    frame = load_source()
    print(f"Source: {SOURCE_PATH.relative_to(PROJECT_ROOT)} — {len(frame):,} rows")
    print("Criteria: "
          f"calm lead >= {MIN_CALM_LEAD_SECONDS:.0f}s (target continuously 0), "
          f"sustained >= {MIN_SUSTAINED_SECONDS:.0f}s, tail >= {MIN_TAIL_SECONDS:.0f}s, "
          f"{MIN_ROWS}-{MAX_ROWS} rows, gap <= {MAX_GAP_SECONDS:.0f}s\n")

    candidates, rejected = find_candidates(frame)
    if not candidates:
        print("No segment satisfies the selection criteria. They were NOT relaxed.\n")
        if rejected:
            table = pd.DataFrame(rejected).sort_values("calm_lead_s", ascending=False).head(10)
            print("Closest rejected transitions:")
            print(table.to_string(index=False))
        return 1

    print(f"{len(candidates)} candidate(s) satisfy the selection criteria, "
          f"ranked by lead-in calmness.\n")

    engine = load_inference_engine()
    print("Verifying through the real pipeline "
          "(live_features -> frozen preprocessor -> LightGBM V1).\n")

    verified = []
    for rank, item in enumerate(candidates, start=1):
        transition_elapsed = item["transition_row"] * REPLAY_INTERVAL_SECONDS
        curve = score_curve(item["selection"], engine)
        verdict = assess(curve, transition_elapsed)
        item |= {"curve": curve, "verdict": verdict, "transition_elapsed": transition_elapsed}
        verified.append(item)

        print(f"  [{rank}/{len(candidates)}] run {str(item['keys'][1])[:13]} — "
              f"calmness cpu {item['lead_cpu']:.1f}% / swap {item['lead_swap']:.1f}% — "
              f"{'PASS' if verdict['passes'] else 'FAIL'}: {verdict['reason']}")
        if verdict["passes"]:
            break

    winners = [item for item in verified if item["verdict"]["passes"]]

    if not winners:
        print("\nNo candidate starts below "
              f"{START_BELOW:.0f} and rises above {PEAK_ABOVE:.0f} around the transition.")
        print("NOTHING WAS EXPORTED. data/replay/replay_segment.csv is unchanged.\n")
        print("The three best candidates by lead-in calmness, with their score curves:\n")
        for rank, item in enumerate(verified[:3], start=1):
            print(f"  ── candidate {rank} " + "─" * 52)
            describe(item)
            print(f"    verdict           : {item['verdict']['reason']}")
            print(f"    first / min / max : {item['verdict']['first_score']} / "
                  f"{item['verdict']['min_score']} / {item['verdict']['max_score']}")
            print()
            print_curve(item["curve"], item["transition_elapsed"])
            print()
        print("Pick one and I will export it, or loosen a criterion deliberately.")
        return 2

    best = winners[0]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best["selection"][EXPORT_COLUMNS].to_csv(OUTPUT_PATH, index=False)

    print("\nSelected and verified:\n")
    describe(best)
    print(f"    transition at     : row {best['transition_row']} "
          f"— {best['transition_elapsed']:.0f}s into the replay")
    print(f"    first score       : {best['verdict']['first_score']} (needs < {START_BELOW:.0f})")
    print(f"    peak near it      : {best['verdict']['peak_near_transition']} "
          f"at {best['verdict']['peak_at_s']:.0f}s (needs > {PEAK_ABOVE:.0f})")
    print(f"    score range       : {best['verdict']['min_score']} .. {best['verdict']['max_score']}")
    print()
    print("  Score curve through the real pipeline:")
    print_curve(best["curve"], best["transition_elapsed"])
    print(f"\n  columns exported : {len(EXPORT_COLUMNS)} "
          f"(machine_id, run_id, timestamp + {len(RAW_CANDIDATE_FEATURES)} raw metrics)")
    print(f"  label exported   : no — {TARGET} locates the episode and is not shipped")
    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
