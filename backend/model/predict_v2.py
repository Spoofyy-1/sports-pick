"""Runtime predictor using the calibrated GBM + live feature state."""
from __future__ import annotations

import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

from model.features import LiveFeatureState

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
GBM_PATH = ARTIFACT_DIR / "gbm.pkl"
CALIB_PATH = ARTIFACT_DIR / "calibrator.pkl"
LIVE_STATE_PATH = ARTIFACT_DIR / "live_state.pkl"

_gbm = None
_calib = None
_live: Optional[LiveFeatureState] = None


def loaded() -> bool:
    return GBM_PATH.exists() and CALIB_PATH.exists() and LIVE_STATE_PATH.exists()


def _load():
    global _gbm, _calib, _live
    if _gbm is None and GBM_PATH.exists():
        _gbm = pickle.loads(GBM_PATH.read_bytes())
    if _calib is None and CALIB_PATH.exists():
        _calib = pickle.loads(CALIB_PATH.read_bytes())
    if _live is None and LIVE_STATE_PATH.exists():
        _live = pickle.loads(LIVE_STATE_PATH.read_bytes())


def reload() -> None:
    global _gbm, _calib, _live
    _gbm = None
    _calib = None
    _live = None
    _load()


def predict(home: str, away: str, game_date_iso: str) -> dict:
    _load()
    try:
        gd = datetime.strptime(game_date_iso[:10], "%Y-%m-%d").date()
    except Exception:
        gd = date.today()
    if _gbm is None or _live is None:
        return {"error": "model_not_trained"}
    feats = _live.predict_features(home, away, gd)
    raw = float(_gbm.predict_proba(np.array([feats], dtype=np.float64))[0, 1])
    calibrated = float(_calib.predict([raw])[0]) if _calib is not None else raw
    # Use raw by default — outperformed isotonic on holdout. Calibrated exposed for inspection.
    return {
        "home_win_prob": round(raw, 4),
        "away_win_prob": round(1 - raw, 4),
        "home_win_prob_calibrated": round(calibrated, 4),
        "home_elo": round(_live.elo.rating(home), 1),
        "away_elo": round(_live.elo.rating(away), 1),
        "features": dict(zip(
            ["elo_diff", "h_rest", "a_rest", "h_b2b", "a_b2b",
             "h_ortg", "a_ortg", "h_drtg", "a_drtg", "h_form", "a_form", "h2h"],
            [round(x, 2) if isinstance(x, float) else x for x in feats],
        )),
        "model": "gbm_v1",
        "trained": True,
    }
