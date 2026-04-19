"""Elo rating model for NBA — 538-style with home-court and margin-of-victory.

- K = 20 base, scaled by log margin-of-victory (MOV multiplier)
- Home-court advantage = +100 Elo
- Between seasons, ratings regress 25% toward 1505 mean
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

MEAN_ELO = 1505.0
HCA = 100.0
K_BASE = 20.0
CARRY_OVER = 0.75


@dataclass
class EloState:
    ratings: dict[str, float] = field(default_factory=dict)
    games_played: dict[str, int] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, MEAN_ELO)

    def expected_home_win_prob(self, home: str, away: str) -> float:
        diff = (self.rating(home) + HCA) - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    def update(self, home: str, away: str, home_score: int, away_score: int) -> tuple[float, bool]:
        expected = self.expected_home_win_prob(home, away)
        home_won = home_score > away_score
        actual = 1.0 if home_won else 0.0
        margin = abs(home_score - away_score)
        elo_diff = (self.rating(home) + HCA) - self.rating(away)
        signed_diff = elo_diff if home_won else -elo_diff
        mov_mult = math.log(max(margin, 1) + 1) * (2.2 / (signed_diff * 0.001 + 2.2))
        delta = K_BASE * mov_mult * (actual - expected)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta
        self.games_played[home] = self.games_played.get(home, 0) + 1
        self.games_played[away] = self.games_played.get(away, 0) + 1
        return expected, home_won

    def regress_to_mean(self) -> None:
        for team in list(self.ratings.keys()):
            self.ratings[team] = MEAN_ELO + (self.ratings[team] - MEAN_ELO) * CARRY_OVER

    def to_dict(self) -> dict:
        return {"ratings": self.ratings, "games_played": self.games_played}

    @classmethod
    def from_dict(cls, d: dict) -> "EloState":
        return cls(ratings=d.get("ratings", {}), games_played=d.get("games_played", {}))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "EloState":
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))


def train(games: Iterable, season_boundary_month: int = 8) -> tuple[EloState, list[dict]]:
    """Walk games chronologically. Returns final state + per-game predictions for backtest."""
    state = EloState()
    records: list[dict] = []
    current_season_year: int | None = None

    for g in games:
        try:
            y, m, _ = g.date.split("-")
            y, m = int(y), int(m)
        except Exception:
            y, m = 0, 0
        season_year = y if m >= season_boundary_month else y - 1
        if current_season_year is not None and season_year != current_season_year:
            state.regress_to_mean()
        current_season_year = season_year

        if state.games_played.get(g.home_team, 0) >= 5 and state.games_played.get(g.away_team, 0) >= 5:
            predicted = state.expected_home_win_prob(g.home_team, g.away_team)
            records.append({
                "date": g.date,
                "home": g.home_team,
                "away": g.away_team,
                "predicted_home_win": predicted,
                "home_won": g.home_won,
            })
        state.update(g.home_team, g.away_team, g.home_score, g.away_score)

    return state, records


def backtest_metrics(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    n = len(records)
    correct = sum(1 for r in records if (r["predicted_home_win"] > 0.5) == r["home_won"])
    log_loss = 0.0
    for r in records:
        p = min(max(r["predicted_home_win"], 1e-6), 1 - 1e-6)
        y = 1.0 if r["home_won"] else 0.0
        log_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    log_loss /= n
    base_rate = sum(1 for r in records if r["home_won"]) / n
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "log_loss": round(log_loss, 4),
        "home_win_rate": round(base_rate, 4),
        "baseline_accuracy_always_home": round(max(base_rate, 1 - base_rate), 4),
    }
