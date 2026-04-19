"""Runtime predictor: loads latest Elo state and produces win probabilities."""
from __future__ import annotations

from pathlib import Path

from .elo import EloState

ARTIFACT = Path(__file__).parent / "artifacts" / "elo.json"

_state: EloState | None = None


def state() -> EloState:
    global _state
    if _state is None:
        _state = EloState.load(ARTIFACT)
    return _state


def reload() -> None:
    global _state
    _state = EloState.load(ARTIFACT)


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
