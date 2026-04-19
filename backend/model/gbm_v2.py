"""GBM v2 — adds box-score-derived features (shooting, opponent-specific memory)."""
from __future__ import annotations

import asyncio
import json
import math
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from data.boxscores import fetch_box_scores, load_cached as load_box_cached
from data.history import fetch_range
from model.features_v2 import EXTENDED_FEATURE_NAMES, build_live_state, build_training_matrix

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
GBM_PATH = ARTIFACT_DIR / "gbm_v2.pkl"
CALIB_PATH = ARTIFACT_DIR / "calibrator_v2.pkl"
LIVE_STATE_PATH = ARTIFACT_DIR / "live_state_v2.pkl"
METRICS_PATH = ARTIFACT_DIR / "gbm_v2_metrics.json"


def _brier(p, y): return sum((a - b) ** 2 for a, b in zip(p, y)) / len(p)
def _log_loss(p, y):
    loss = 0.0
    for a, b in zip(p, y):
        a = min(max(a, 1e-6), 1 - 1e-6)
        loss += -(b * math.log(a) + (1 - b) * math.log(1 - a))
    return loss / len(p)
def _acc(p, y): return sum(1 for a, b in zip(p, y) if (a > 0.5) == (b == 1)) / len(p)


async def _ensure_box_scores(games, concurrency: int = 10) -> dict:
    have = {}
    missing: list[str] = []
    for g in games:
        eid = g.event_id
        if not eid:
            continue
        cached = load_box_cached(eid)
        if cached:
            have[eid] = cached
        else:
            missing.append(eid)

    print(f"  box scores: {len(have)} cached, {len(missing)} to fetch")
    if missing:
        fetched = await fetch_box_scores(missing, concurrency=concurrency)
        have.update(fetched)
    return have


async def train_and_evaluate(years_back: int = 3, calib_frac: float = 0.15, test_frac: float = 0.15):
    today = date.today()
    start = today - timedelta(days=365 * years_back)
    print(f"fetching scoreboard {start} → {today}…")
    games = await fetch_range(start, today)
    print(f"  {len(games)} games")

    print("fetching box scores…")
    box_scores = await _ensure_box_scores(games)
    print(f"  got box scores for {len(box_scores)} events")

    print("engineering extended features (walk-forward)…")
    X, y, meta = build_training_matrix(games, box_scores)
    print(f"  {len(X)} rows × {len(EXTENDED_FEATURE_NAMES)} features")

    n = len(X)
    n_test = int(n * test_frac)
    n_calib = int(n * calib_frac)
    n_train = n - n_test - n_calib
    X_train, X_calib, X_test = X[:n_train], X[n_train:n_train + n_calib], X[n_train + n_calib:]
    y_train, y_calib, y_test = y[:n_train], y[n_train:n_train + n_calib], y[n_train + n_calib:]
    print(f"  train {len(X_train)} | calib {len(X_calib)} | test {len(X_test)}")
    print(f"  train dates {meta[0]['date']} → {meta[n_train-1]['date']}")
    print(f"  test  dates {meta[n_train+n_calib]['date']} → {meta[-1]['date']}")

    Xn_tr, yn_tr = np.array(X_train, dtype=np.float64), np.array(y_train)
    Xn_ca = np.array(X_calib, dtype=np.float64)
    yn_ca = np.array(y_calib)
    Xn_te = np.array(X_test, dtype=np.float64)

    print("\ntraining GBM v2…")
    gbm = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.04, max_depth=6,
        min_samples_leaf=20, l2_regularization=1.5,
        random_state=42, validation_fraction=0.1,
        early_stopping=True, n_iter_no_change=25,
    )
    gbm.fit(Xn_tr, yn_tr)
    print(f"  trained ({gbm.n_iter_} iterations)")

    print("\nfitting isotonic calibrator…")
    raw_calib = gbm.predict_proba(Xn_ca)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_calib, yn_ca)

    raw_test = gbm.predict_proba(Xn_te)[:, 1]
    calib_test = iso.predict(raw_test)

    # Feature importance — permutation-style via sklearn's built-in
    importances = None
    try:
        from sklearn.inspection import permutation_importance
        print("\ncomputing permutation importance on test set (5 repeats)…")
        pi = permutation_importance(
            gbm, Xn_te, np.array(y_test), n_repeats=5, random_state=42, n_jobs=1
        )
        importances = sorted(
            [(name, float(m)) for name, m in zip(EXTENDED_FEATURE_NAMES, pi.importances_mean)],
            key=lambda x: x[1], reverse=True,
        )
    except Exception as e:
        print(f"  skipped importance: {e}")

    metrics = {
        "n_train": len(X_train),
        "n_calib": len(X_calib),
        "n_test": len(X_test),
        "gbm_raw": {
            "accuracy": round(_acc(raw_test.tolist(), y_test), 4),
            "log_loss": round(_log_loss(raw_test.tolist(), y_test), 4),
            "brier": round(_brier(raw_test.tolist(), y_test), 4),
        },
        "gbm_calibrated": {
            "accuracy": round(_acc(calib_test.tolist(), y_test), 4),
            "log_loss": round(_log_loss(calib_test.tolist(), y_test), 4),
            "brier": round(_brier(calib_test.tolist(), y_test), 4),
        },
        "top_features": importances[:15] if importances else None,
        "feature_names": EXTENDED_FEATURE_NAMES,
    }

    # Final model on full history
    X_all = np.array(X, dtype=np.float64)
    y_all = np.array(y)
    print("\ntraining final model on full history…")
    gbm_final = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.04, max_depth=6,
        min_samples_leaf=20, l2_regularization=1.5, random_state=42,
    )
    gbm_final.fit(X_all, y_all)
    GBM_PATH.write_bytes(pickle.dumps(gbm_final))
    CALIB_PATH.write_bytes(pickle.dumps(iso))
    live = build_live_state(games, box_scores)
    LIVE_STATE_PATH.write_bytes(pickle.dumps(live))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


def _print_metrics(m: dict):
    print(f"\nholdout n={m['n_test']}")
    for name, v in [("GBM v2 (raw)", m["gbm_raw"]), ("GBM v2 (calibrated)", m["gbm_calibrated"])]:
        print(f"  {name:22s}  acc={v['accuracy']:.3f}  log_loss={v['log_loss']:.3f}  brier={v['brier']:.3f}")
    if m.get("top_features"):
        print("\ntop 10 features by permutation importance:")
        for n, i in m["top_features"][:10]:
            print(f"  {n:28s}  {i:+.4f}")


async def main():
    m = await train_and_evaluate()
    _print_metrics(m)


if __name__ == "__main__":
    asyncio.run(main())
