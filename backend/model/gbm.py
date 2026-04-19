"""Gradient-boosted team model + isotonic calibration.

Trained on walk-forward engineered features. At predict time, calibrator
maps raw GBM probability to a well-calibrated probability via isotonic
regression fit on held-out walk-forward predictions.
"""
from __future__ import annotations

import asyncio
import json
import math
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from data.history import fetch_range, load_cached
from model.features import FEATURE_NAMES, build_live_state, build_training_matrix
from model.elo import EloState, train as elo_walk_forward

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
GBM_PATH = ARTIFACT_DIR / "gbm.pkl"
CALIB_PATH = ARTIFACT_DIR / "calibrator.pkl"
GBM_METRICS_PATH = ARTIFACT_DIR / "gbm_metrics.json"
LIVE_STATE_PATH = ARTIFACT_DIR / "live_state.pkl"


def _brier(preds: list[float], actuals: list[int]) -> float:
    return sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(preds)


def _log_loss(preds: list[float], actuals: list[int]) -> float:
    loss = 0.0
    for p, a in zip(preds, actuals):
        p = min(max(p, 1e-6), 1 - 1e-6)
        loss += -(a * math.log(p) + (1 - a) * math.log(1 - p))
    return loss / len(preds)


def _accuracy(preds: list[float], actuals: list[int]) -> float:
    return sum(1 for p, a in zip(preds, actuals) if (p > 0.5) == (a == 1)) / len(preds)


def _calibration_table(preds: list[float], actuals: list[int], bins: int = 10) -> list[dict]:
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for p, a in zip(preds, actuals):
        i = min(int(p * bins), bins - 1)
        buckets[i].append((p, a))
    out = []
    for i, b in enumerate(buckets):
        lo, hi = i / bins, (i + 1) / bins
        if not b:
            out.append({"range": f"{lo:.1f}-{hi:.1f}", "n": 0})
            continue
        mean_p = sum(x[0] for x in b) / len(b)
        hit = sum(x[1] for x in b) / len(b)
        out.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "n": len(b),
            "mean_pred": round(mean_p, 3),
            "actual": round(hit, 3),
            "gap": round(hit - mean_p, 3),
        })
    return out


async def train_and_evaluate(years_back: int = 3, calib_frac: float = 0.15, test_frac: float = 0.15) -> dict:
    """Pull data, engineer features, train GBM, fit isotonic calibrator, evaluate."""
    today = date.today()
    start = today - timedelta(days=365 * years_back)
    print(f"fetching {start} → {today}…")
    games = await fetch_range(start, today)
    print(f"  {len(games)} games")

    print("engineering features (walk-forward)…")
    X, y, meta = build_training_matrix(games)
    print(f"  {len(X)} game-feature rows with {len(FEATURE_NAMES)} features each")

    # Chronological split: train | calib | test
    n = len(X)
    n_test = int(n * test_frac)
    n_calib = int(n * calib_frac)
    n_train = n - n_test - n_calib
    X_train, X_calib, X_test = X[:n_train], X[n_train:n_train + n_calib], X[n_train + n_calib:]
    y_train, y_calib, y_test = y[:n_train], y[n_train:n_train + n_calib], y[n_train + n_calib:]
    meta_test = meta[n_train + n_calib:]

    print(f"  train {len(X_train)} | calib {len(X_calib)} | test {len(X_test)}")
    print(f"  train dates {meta[0]['date']} → {meta[n_train-1]['date']}")
    print(f"  calib dates {meta[n_train]['date']} → {meta[n_train+n_calib-1]['date']}")
    print(f"  test  dates {meta[n_train+n_calib]['date']} → {meta[-1]['date']}")

    X_train_np = np.array(X_train, dtype=np.float64)
    y_train_np = np.array(y_train)
    X_calib_np = np.array(X_calib, dtype=np.float64)
    y_calib_np = np.array(y_calib)
    X_test_np = np.array(X_test, dtype=np.float64)

    print("\ntraining HistGradientBoostingClassifier…")
    gbm = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
        validation_fraction=0.1,
        early_stopping=True,
        n_iter_no_change=20,
    )
    gbm.fit(X_train_np, y_train_np)
    print(f"  trained ({gbm.n_iter_} iterations)")

    raw_calib = gbm.predict_proba(X_calib_np)[:, 1]
    print("\nfitting isotonic calibrator on held-out calib set…")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_calib, y_calib_np)

    # Evaluate on true holdout
    raw_test = gbm.predict_proba(X_test_np)[:, 1]
    calib_test = iso.predict(raw_test)

    # Benchmark: Elo alone (walk-forward through the same train set, then applied to test)
    # Rebuild Elo state as it was at start of test window.
    elo_train_games = games[:n_train + n_calib]
    elo_state, _ = elo_walk_forward(elo_train_games)
    test_games = games[n_train + n_calib:]
    elo_preds = []
    for g in test_games:
        # only score games that made it into the feature matrix
        elo_preds.append(elo_state.expected_home_win_prob(g.home_team, g.away_team))
        elo_state.update(g.home_team, g.away_team, g.home_score, g.away_score)
    # Trim elo_preds to match feature rows (skip cold-start games)
    elo_test = elo_preds[-len(X_test):]

    metrics = {
        "n_train": len(X_train),
        "n_calib": len(X_calib),
        "n_test": len(X_test),
        "elo": {
            "accuracy": round(_accuracy(elo_test, y_test), 4),
            "log_loss": round(_log_loss(elo_test, y_test), 4),
            "brier": round(_brier(elo_test, y_test), 4),
        },
        "gbm_raw": {
            "accuracy": round(_accuracy(raw_test.tolist(), y_test), 4),
            "log_loss": round(_log_loss(raw_test.tolist(), y_test), 4),
            "brier": round(_brier(raw_test.tolist(), y_test), 4),
        },
        "gbm_calibrated": {
            "accuracy": round(_accuracy(calib_test.tolist(), y_test), 4),
            "log_loss": round(_log_loss(calib_test.tolist(), y_test), 4),
            "brier": round(_brier(calib_test.tolist(), y_test), 4),
        },
        "calibration_gbm_raw": _calibration_table(raw_test.tolist(), y_test),
        "calibration_gbm_calibrated": _calibration_table(calib_test.tolist(), y_test),
        "feature_names": FEATURE_NAMES,
    }

    # Save artifacts: train a FINAL model on train+calib+test so live predictions are maximal-data
    X_all_np = np.array(X, dtype=np.float64)
    y_all_np = np.array(y)
    print("\ntraining final model on full history…")
    gbm_final = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    gbm_final.fit(X_all_np, y_all_np)
    GBM_PATH.write_bytes(pickle.dumps(gbm_final))
    CALIB_PATH.write_bytes(pickle.dumps(iso))

    live_state = build_live_state(games)
    LIVE_STATE_PATH.write_bytes(pickle.dumps(live_state))

    GBM_METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("\nsaved:")
    print(f"  {GBM_PATH}")
    print(f"  {CALIB_PATH}")
    print(f"  {LIVE_STATE_PATH}")
    print(f"  {GBM_METRICS_PATH}")

    return metrics


def _print_metrics(m: dict) -> None:
    print(f"\nholdout n={m['n_test']}")
    for name, v in [("Elo only", m["elo"]), ("GBM (raw)", m["gbm_raw"]), ("GBM (calibrated)", m["gbm_calibrated"])]:
        print(f"  {name:18s}  acc={v['accuracy']:.3f}  log_loss={v['log_loss']:.3f}  brier={v['brier']:.3f}")
    print("\ncalibration (GBM calibrated):")
    for row in m["calibration_gbm_calibrated"]:
        if not row.get("n"):
            continue
        print(f"  {row['range']}  n={row['n']:4d}  pred={row['mean_pred']:.3f}  actual={row['actual']:.3f}  gap={row['gap']:+.3f}")


async def main():
    m = await train_and_evaluate()
    _print_metrics(m)


if __name__ == "__main__":
    asyncio.run(main())
