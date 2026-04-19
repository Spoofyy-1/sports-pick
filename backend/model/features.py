"""Feature engineering for NBA team games.

Walks the historical scoreboard cache chronologically and computes,
for each game *as of the moment before tipoff*, a feature vector:

  - elo_diff          Elo rating differential with HCA baked in
  - home_rest_days    days since home team's last game (capped at 5)
  - away_rest_days    days since away team's last game (capped at 5)
  - home_b2b          1 if home played yesterday, else 0
  - away_b2b          1 if away played yesterday, else 0
  - home_roll_ortg    home team's rolling points scored per game (last 10)
  - away_roll_ortg    away team's rolling points scored per game (last 10)
  - home_roll_drtg    home team's rolling opp points allowed per game (last 10)
  - away_roll_drtg    away team's rolling opp points allowed per game (last 10)
  - home_form         home win% over last 10 games
  - away_form         away win% over last 10 games
  - h2h_recent        home win% in last 5 H2H meetings (0.5 if <2 meetings)

Strictly walk-forward — each feature for game G uses only games before G.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable

from model.elo import EloState

FEATURE_NAMES = [
    "elo_diff",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "home_roll_ortg",
    "away_roll_ortg",
    "home_roll_drtg",
    "away_roll_drtg",
    "home_form",
    "away_form",
    "h2h_recent",
]


@dataclass
class TeamRollingState:
    last_game_date: date | None = None
    recent_points_for: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_points_against: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_wins: deque = field(default_factory=lambda: deque(maxlen=10))


def _parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def _rest_days(game_date: date, last: date | None) -> tuple[float, int]:
    if last is None:
        return 3.0, 0  # neutral default for first game of sample
    days = (game_date - last).days
    return float(min(days, 5)), 1 if days <= 1 else 0


def _mean(xs: Iterable[float], default: float) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else default


def build_training_matrix(games, hca_elo: float = 100.0) -> tuple[list[list[float]], list[int], list[dict]]:
    """Return (X, y, meta) — walk-forward engineered features + outcomes."""
    elo = EloState()
    state: dict[str, TeamRollingState] = defaultdict(TeamRollingState)
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=5))

    X: list[list[float]] = []
    y: list[int] = []
    meta: list[dict] = []

    league_pts = 113.0  # neutral prior for first couple of games per team

    for g in games:
        try:
            gd = _parse_date(g.date)
        except Exception:
            continue

        h_state = state[g.home_team]
        a_state = state[g.away_team]
        h_rest, h_b2b = _rest_days(gd, h_state.last_game_date)
        a_rest, a_b2b = _rest_days(gd, a_state.last_game_date)

        elo_diff = (elo.rating(g.home_team) + hca_elo) - elo.rating(g.away_team)

        h_ortg = _mean(h_state.recent_points_for, league_pts)
        a_ortg = _mean(a_state.recent_points_for, league_pts)
        h_drtg = _mean(h_state.recent_points_against, league_pts)
        a_drtg = _mean(a_state.recent_points_against, league_pts)
        h_form = _mean(h_state.recent_wins, 0.5)
        a_form = _mean(a_state.recent_wins, 0.5)

        key = tuple(sorted([g.home_team, g.away_team]))
        h2h_recent_wins = list(h2h[key])
        if len(h2h_recent_wins) >= 2:
            # value = 1 if the HOME team of *this* game won the prior meeting, else 0
            h2h_home_rate = sum(
                1 for winner in h2h_recent_wins if winner == g.home_team
            ) / len(h2h_recent_wins)
        else:
            h2h_home_rate = 0.5

        # Feature vector; only emit once both teams have ≥5 games to avoid cold-start noise
        if len(h_state.recent_points_for) >= 5 and len(a_state.recent_points_for) >= 5:
            X.append([
                elo_diff,
                h_rest,
                a_rest,
                float(h_b2b),
                float(a_b2b),
                h_ortg,
                a_ortg,
                h_drtg,
                a_drtg,
                h_form,
                a_form,
                h2h_home_rate,
            ])
            y.append(1 if g.home_won else 0)
            meta.append({
                "date": g.date,
                "home": g.home_team,
                "away": g.away_team,
                "home_score": g.home_score,
                "away_score": g.away_score,
            })

        # Update state AFTER feature snapshot
        h_state.recent_points_for.append(g.home_score)
        h_state.recent_points_against.append(g.away_score)
        h_state.recent_wins.append(1.0 if g.home_won else 0.0)
        h_state.last_game_date = gd

        a_state.recent_points_for.append(g.away_score)
        a_state.recent_points_against.append(g.home_score)
        a_state.recent_wins.append(0.0 if g.home_won else 1.0)
        a_state.last_game_date = gd

        winner = g.home_team if g.home_won else g.away_team
        h2h[key].append(winner)

        elo.update(g.home_team, g.away_team, g.home_score, g.away_score)

    return X, y, meta


@dataclass
class LiveFeatureState:
    """Frozen snapshot for predicting games that haven't happened yet."""
    elo: EloState
    team_state: dict[str, TeamRollingState]
    h2h: dict[tuple[str, str], deque]

    def predict_features(self, home: str, away: str, game_date: date, hca_elo: float = 100.0) -> list[float]:
        h = self.team_state.get(home, TeamRollingState())
        a = self.team_state.get(away, TeamRollingState())
        h_rest, h_b2b = _rest_days(game_date, h.last_game_date)
        a_rest, a_b2b = _rest_days(game_date, a.last_game_date)
        elo_diff = (self.elo.rating(home) + hca_elo) - self.elo.rating(away)
        h_ortg = _mean(h.recent_points_for, 113.0)
        a_ortg = _mean(a.recent_points_for, 113.0)
        h_drtg = _mean(h.recent_points_against, 113.0)
        a_drtg = _mean(a.recent_points_against, 113.0)
        h_form = _mean(h.recent_wins, 0.5)
        a_form = _mean(a.recent_wins, 0.5)
        key = tuple(sorted([home, away]))
        h2h_recent_wins = list(self.h2h.get(key, []))
        if len(h2h_recent_wins) >= 2:
            h2h_home_rate = sum(1 for w in h2h_recent_wins if w == home) / len(h2h_recent_wins)
        else:
            h2h_home_rate = 0.5
        return [
            elo_diff, h_rest, a_rest, float(h_b2b), float(a_b2b),
            h_ortg, a_ortg, h_drtg, a_drtg, h_form, a_form, h2h_home_rate,
        ]


def build_live_state(games) -> LiveFeatureState:
    """Replay all games to produce a frozen LiveFeatureState for prediction."""
    elo = EloState()
    state: dict[str, TeamRollingState] = defaultdict(TeamRollingState)
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=5))
    for g in games:
        try:
            gd = _parse_date(g.date)
        except Exception:
            continue
        h_state, a_state = state[g.home_team], state[g.away_team]
        h_state.recent_points_for.append(g.home_score)
        h_state.recent_points_against.append(g.away_score)
        h_state.recent_wins.append(1.0 if g.home_won else 0.0)
        h_state.last_game_date = gd
        a_state.recent_points_for.append(g.away_score)
        a_state.recent_points_against.append(g.home_score)
        a_state.recent_wins.append(0.0 if g.home_won else 1.0)
        a_state.last_game_date = gd
        winner = g.home_team if g.home_won else g.away_team
        h2h[tuple(sorted([g.home_team, g.away_team]))].append(winner)
        elo.update(g.home_team, g.away_team, g.home_score, g.away_score)
    return LiveFeatureState(elo=elo, team_state=dict(state), h2h=dict(h2h))
