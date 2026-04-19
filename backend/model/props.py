"""Player prop probability model.

For each player, we model each stat (points, rebounds, assists, threes) as
a normal distribution with mean = rolling weighted average (recent games
weighted heavier) and stdev = season stdev. That gives P(stat > line) for
any line. Simple but honest; matches well with how books price soft props.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Iterable

from data.players import PlayerGame, PlayerProfile


STAT_FIELDS = ["points", "rebounds", "assists", "threes"]


def _values(games: Iterable[PlayerGame], field: str, min_minutes: int = 10) -> list[int]:
    return [getattr(g, field) for g in games if g.minutes >= min_minutes]


def _weighted_mean(vals: list[int], halflife: float = 10.0) -> float:
    if not vals:
        return 0.0
    weights = [0.5 ** (i / halflife) for i in range(len(vals))]
    return sum(v * w for v, w in zip(vals, weights)) / sum(weights)


def _normal_over(line: float, mu: float, sigma: float) -> float:
    if sigma <= 0.01:
        return 1.0 if mu > line else 0.0
    z = (line - mu) / sigma
    return 0.5 * (1 - math.erf(z / math.sqrt(2)))


@dataclass
class PropQuote:
    player: str
    player_id: str
    team: str
    stat: str
    line: float
    model_prob_over: float
    model_prob_under: float
    mean: float
    stdev: float
    n_games: int
    recent: list[int]


def _player_stat_summary(profile: PlayerProfile, stat: str) -> tuple[float, float, list[int], int]:
    regular = profile.regular_season()
    regular_sorted = sorted(regular, key=lambda g: g.event_id or "", reverse=True)
    values = _values(regular_sorted, stat)
    if len(values) < 5:
        return 0.0, 0.0, [], 0
    mu = _weighted_mean(values)
    sigma = pstdev(values) if len(values) > 1 else 0.0
    return mu, sigma, values[:10], len(values)


def price_prop(
    profile: PlayerProfile, stat: str, line: float
) -> PropQuote | None:
    if stat not in STAT_FIELDS:
        return None
    mu, sigma, recent, n = _player_stat_summary(profile, stat)
    if n == 0:
        return None
    p_over = _normal_over(line, mu, sigma)
    return PropQuote(
        player=profile.name,
        player_id=profile.id,
        team=profile.team,
        stat=stat,
        line=line,
        model_prob_over=round(p_over, 4),
        model_prob_under=round(1 - p_over, 4),
        mean=round(mu, 2),
        stdev=round(sigma, 2),
        n_games=n,
        recent=recent,
    )


MIN_MEAN = {"points": 8.0, "rebounds": 3.0, "assists": 2.0, "threes": 1.0}


def suggest_lines_for_player(profile: PlayerProfile) -> list[PropQuote]:
    """One quote per stat at a book-fair-ish half-point line.
    Skips low-volume stats where a prop bet wouldn't exist at a book."""
    quotes: list[PropQuote] = []
    for stat in STAT_FIELDS:
        mu, sigma, recent, n = _player_stat_summary(profile, stat)
        if n < 15:
            continue
        if mu < MIN_MEAN.get(stat, 5.0):
            continue
        if sigma < 0.5:
            continue
        line = max(0.5, round(mu * 2) / 2 - 0.5)
        q = price_prop(profile, stat, line)
        if q:
            quotes.append(q)
    return quotes


def top_confidence_props(profiles: list[PlayerProfile], min_prob: float = 0.70) -> list[dict]:
    """Across all provided players, return the strongest over-bets vs. a typical line."""
    out: list[dict] = []
    for profile in profiles:
        for q in suggest_lines_for_player(profile):
            if q.model_prob_over >= min_prob:
                out.append({
                    "player": q.player,
                    "team": q.team,
                    "stat": q.stat,
                    "line": q.line,
                    "side": "over",
                    "model_prob": q.model_prob_over,
                    "mean": q.mean,
                    "stdev": q.stdev,
                    "n_games": q.n_games,
                    "recent": q.recent,
                })
            if q.model_prob_under >= min_prob:
                out.append({
                    "player": q.player,
                    "team": q.team,
                    "stat": q.stat,
                    "line": q.line,
                    "side": "under",
                    "model_prob": q.model_prob_under,
                    "mean": q.mean,
                    "stdev": q.stdev,
                    "n_games": q.n_games,
                    "recent": q.recent,
                })
    out.sort(key=lambda x: x["model_prob"], reverse=True)
    return out
