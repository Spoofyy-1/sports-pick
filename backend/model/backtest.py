"""Honest backtest with enforced train/test date split.

Usage:
    python -m model.backtest --split 2025-10-01
    python -m model.backtest --split 2024-07-01 --end 2025-06-30

Guarantees:
 - Training Elo only sees games with date < --split.
 - Test predictions are made from the frozen post-train Elo state.
   (Ratings are updated as we walk through the test window, but each
   prediction is computed *before* the game's result is applied — classic
   walk-forward, no leakage.)
 - Prints explicit date-range assertions so you can verify no overlap.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from data.history import fetch_range
from model.elo import EloState, MEAN_ELO, train as walk_forward_train

REPORT_DIR = Path(__file__).parent / "artifacts"
REPORT_DIR.mkdir(exist_ok=True)


@dataclass
class Prediction:
    date: str
    home: str
    away: str
    predicted_home_win: float
    home_won: bool


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _split_games(games, split: date):
    train, test = [], []
    for g in games:
        try:
            y, m, d = g.date.split("-")
            gd = date(int(y), int(m), int(d))
        except Exception:
            continue
        (train if gd < split else test).append(g)
    return train, test


def _walk_forward(state: EloState, games, min_games: int = 5) -> list[Prediction]:
    """Predict each game BEFORE applying its result. Returns predictions list."""
    preds: list[Prediction] = []
    for g in games:
        if (
            state.games_played.get(g.home_team, 0) >= min_games
            and state.games_played.get(g.away_team, 0) >= min_games
        ):
            p = state.expected_home_win_prob(g.home_team, g.away_team)
            preds.append(Prediction(g.date, g.home_team, g.away_team, p, g.home_won))
        state.update(g.home_team, g.away_team, g.home_score, g.away_score)
    return preds


def _metrics(preds: list[Prediction]) -> dict:
    n = len(preds)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for p in preds if (p.predicted_home_win > 0.5) == p.home_won)
    brier = sum((p.predicted_home_win - (1.0 if p.home_won else 0.0)) ** 2 for p in preds) / n
    ll = 0.0
    for p in preds:
        prob = min(max(p.predicted_home_win, 1e-6), 1 - 1e-6)
        y = 1.0 if p.home_won else 0.0
        ll += -(y * math.log(prob) + (1 - y) * math.log(1 - prob))
    ll /= n
    home_rate = sum(1 for p in preds if p.home_won) / n
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "home_win_rate": round(home_rate, 4),
        "baseline_always_home": round(max(home_rate, 1 - home_rate), 4),
    }


def _calibration_buckets(preds: list[Prediction], bins: int = 10) -> list[dict]:
    buckets = [[] for _ in range(bins)]
    for p in preds:
        i = min(int(p.predicted_home_win * bins), bins - 1)
        buckets[i].append(p)
    out = []
    for i, bucket in enumerate(buckets):
        lo, hi = i / bins, (i + 1) / bins
        if not bucket:
            out.append({"range": f"{lo:.1f}-{hi:.1f}", "n": 0})
            continue
        mean_pred = sum(b.predicted_home_win for b in bucket) / len(bucket)
        actual = sum(1 for b in bucket if b.home_won) / len(bucket)
        out.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "n": len(bucket),
            "mean_predicted": round(mean_pred, 3),
            "actual_hit_rate": round(actual, 3),
            "gap": round(actual - mean_pred, 3),
        })
    return out


def _assert_no_leakage(train_games, test_games, split: date):
    if train_games:
        max_train = max(_parse_date(g.date) for g in train_games)
        assert max_train < split, f"train set has game on {max_train} >= split {split}"
    if test_games:
        min_test = min(_parse_date(g.date) for g in test_games)
        assert min_test >= split, f"test set has game on {min_test} < split {split}"


async def run(split: date, end: date | None, years_back: int) -> dict:
    start = split - timedelta(days=365 * years_back)
    end = end or date.today()
    print(f"\nfetching games {start} → {end} from ESPN (per-day cached)…")
    games = await fetch_range(start, end)
    print(f"  {len(games)} total finished games")

    train_games, test_games = _split_games(games, split)
    print(f"  train: {len(train_games)}  test: {len(test_games)}")

    _assert_no_leakage(train_games, test_games, split)

    if train_games:
        train_dates = [g.date for g in train_games]
        print(f"  train date range: {min(train_dates)} → {max(train_dates)}")
    if test_games:
        test_dates = [g.date for g in test_games]
        print(f"  test date range:  {min(test_dates)} → {max(test_dates)}")
    print(f"  split boundary: {split} (exclusive on train side)\n")

    print("training Elo on train window only…")
    train_state, _ = walk_forward_train(train_games)
    print(f"  trained — {len(train_state.ratings)} teams, {sum(train_state.games_played.values())//2} games applied")

    print("\npredicting test games (walk-forward, frozen-then-evolving state)…")
    state_copy = EloState.from_dict(json.loads(json.dumps(train_state.to_dict())))
    preds = _walk_forward(state_copy, test_games)

    in_sample_preds = _walk_forward(EloState(), train_games)

    train_m = _metrics(in_sample_preds)
    test_m = _metrics(preds)
    calib = _calibration_buckets(preds)

    return {
        "split": split.isoformat(),
        "end": end.isoformat(),
        "train": train_m,
        "test": test_m,
        "calibration": calib,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="YYYY-MM-DD: first test-set date (exclusive on train)")
    ap.add_argument("--end", help="YYYY-MM-DD: last date to include (default: today)")
    ap.add_argument("--years-back", type=int, default=3, help="years of training history before --split")
    ap.add_argument("--save", default="backtest_report.json")
    args = ap.parse_args()

    split = _parse_date(args.split)
    end = _parse_date(args.end) if args.end else None

    result = asyncio.run(run(split, end, args.years_back))

    print("\n=== TRAIN (in-sample walk-forward) ===")
    print(json.dumps(result["train"], indent=2))
    print("\n=== TEST (out-of-sample) ===")
    print(json.dumps(result["test"], indent=2))
    print("\n=== CALIBRATION (test) ===")
    for b in result["calibration"]:
        if b.get("n", 0) == 0:
            continue
        print(
            f"  {b['range']}: n={b['n']:4d} predicted={b['mean_predicted']}  actual={b['actual_hit_rate']}  gap={b['gap']:+.3f}"
        )

    out_path = REPORT_DIR / args.save
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nreport saved to {out_path}")


if __name__ == "__main__":
    main()
