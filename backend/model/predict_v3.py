"""Runtime predictor using GBM v2 (27 features, opponent-specific memory)."""
from __future__ import annotations

import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

from model.features_v2 import EXTENDED_FEATURE_NAMES, LiveStateV2

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
GBM_PATH = ARTIFACT_DIR / "gbm_v2.pkl"
CALIB_PATH = ARTIFACT_DIR / "calibrator_v2.pkl"
LIVE_STATE_PATH = ARTIFACT_DIR / "live_state_v2.pkl"

_gbm = None
_calib = None
_live: Optional[LiveStateV2] = None


def loaded() -> bool:
    return GBM_PATH.exists() and LIVE_STATE_PATH.exists()


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
    _gbm, _calib, _live = None, None, None
    _load()


def predict(home: str, away: str, game_date_iso: str) -> dict:
    _load()
    if _gbm is None or _live is None:
        return {"error": "model_v2_not_trained"}
    try:
        gd = datetime.strptime(game_date_iso[:10], "%Y-%m-%d").date()
    except Exception:
        gd = date.today()
    feats = _live.features_for(home, away, gd)
    raw = float(_gbm.predict_proba(np.array([feats], dtype=np.float64))[0, 1])
    calibrated = float(_calib.predict([raw])[0]) if _calib is not None else raw
    return {
        "home_win_prob": round(raw, 4),
        "away_win_prob": round(1 - raw, 4),
        "home_win_prob_calibrated": round(calibrated, 4),
        "home_elo": round(_live.elo.rating(home), 1),
        "away_elo": round(_live.elo.rating(away), 1),
        "features": dict(zip(EXTENDED_FEATURE_NAMES, [
            round(x, 3) if isinstance(x, float) else x for x in feats
        ])),
        "model": "gbm_v2_boxscore",
        "trained": True,
    }
