"""Runtime predictor: loads latest Elo state and produces win probabilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .elo import EloState

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT = ARTIFACT_DIR / "elo.json"
PREDICTIONS = ARTIFACT_DIR / "predictions.json"

_state: Optional[EloState] = None
_preds: Optional[dict] = None


def state() -> EloState:
    global _state
    if _state is None:
        _state = EloState.load(ARTIFACT)
    return _state


def pregame_predictions() -> dict:
    """Pre-update walk-forward predictions made during training. Honest for backtest."""
    global _preds
    if _preds is None:
        _preds = json.loads(PREDICTIONS.read_text()) if PREDICTIONS.exists() else {}
    return _preds


def reload() -> None:
    global _state, _preds
    _state = EloState.load(ARTIFACT)
    _preds = None


def win_probabilities(home: str, away: str) -> dict:
    s = state()
    p_home = s.expected_home_win_prob(home, away)
    return {
        "home_win_prob": round(p_home, 4),
        "away_win_prob": round(1 - p_home, 4),
        "home_elo": round(s.rating(home), 1),
        "away_elo": round(s.rating(away), 1),
        "model": "elo_v1",
        "trained": bool(s.ratings),
    }


def pregame_win_prob(date_prefix: str, home: str, away: str) -> Optional[float]:
    """Pull the honest pre-game walk-forward prediction for a completed game."""
    preds = pregame_predictions()
    key = f"{date_prefix[:10]}|{home}|{away}"
    return preds.get(key)
