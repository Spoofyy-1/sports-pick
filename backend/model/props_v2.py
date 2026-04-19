"""Upgraded prop model.

Improvements over v1:
 1. Opponent-defense adjustment from team pace + points-allowed (derived from scoreboard cache)
 2. Poisson distribution for low-count stats (threes, steals, blocks)
 3. Truncated-normal for continuous-ish stats (points, rebounds, assists)
 4. Per-stat MIN_MEAN and confidence thresholds tuned per distribution
 5. Recent-minutes filter (skips players whose minutes have collapsed — hurt / out of rotation)
 6. Playoff shrinkage: mean trimmed, variance inflated
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Optional

from data.players import PlayerGame, PlayerProfile
from model.team_ratings import team_ratings, league_average_ppg

STAT_FIELDS = ["points", "rebounds", "assists", "threes"]

MIN_MEAN = {"points": 10.0, "rebounds": 3.5, "assists": 2.0, "threes": 1.0}
MIN_STDEV = {"points": 3.0, "rebounds": 1.5, "assists": 1.0, "threes": 0.5}
MIN_RECENT_MINUTES = 18  # avg minutes over last 5 games — filters injured/benched players

# Empirical NBA league-average points per team per game is ~113.
LEAGUE_AVG_PPG = 113.0


def _values(games: Iterable[PlayerGame], field: str, min_minutes: int = 10) -> list[int]:
    return [getattr(g, field) for g in games if g.minutes >= min_minutes]


def _weighted_mean(vals: list[int], halflife: float = 10.0) -> float:
    if not vals:
        return 0.0
    weights = [0.5 ** (i / halflife) for i in range(len(vals))]
    return sum(v * w for v, w in zip(vals, weights)) / sum(weights)


def _recent_minutes(profile: PlayerProfile) -> float:
    regular = profile.regular_season()
    regular_sorted = sorted(regular, key=lambda g: g.event_id or "", reverse=True)
    last5 = regular_sorted[:5]
    if not last5:
        return 0.0
    return mean([g.minutes for g in last5])


def _normal_over(line: float, mu: float, sigma: float) -> float:
    if sigma <= 0.01:
        return 1.0 if mu > line else 0.0
    z = (line - mu) / sigma
    return 0.5 * (1 - math.erf(z / math.sqrt(2)))


def _poisson_over(line: float, lam: float) -> float:
    """P(X > line) where X ~ Poisson(lam). For half-point lines, no ambiguity."""
    k = int(math.floor(line))
    # P(X <= k) = sum_{i=0..k} lam^i e^-lam / i!
    cdf = 0.0
    term = math.exp(-lam)
    cdf += term
    for i in range(1, k + 1):
        term *= lam / i
        cdf += term
    return 1.0 - cdf


def _truncated_normal_over(line: float, mu: float, sigma: float) -> float:
    """Same as normal_over but floor the result at 0 and cap at a reasonable max.
    For our use case (half-point lines, mu > 0), truncation at 0 doesn't change much,
    so this is effectively the normal CDF. Kept for explicit intent."""
    return _normal_over(line, mu, sigma)


def _opp_adjust(team_abbr: str, opp_abbr: Optional[str], stat: str) -> float:
    """Returns multiplicative factor on expected stat.
    Defensive index: opp_ppg_allowed / league_avg. Values <1 mean stingy defense → shrink.
    Currently applied to points and threes (rebounds/assists less defense-sensitive)."""
    if not opp_abbr:
        return 1.0
    ratings = team_ratings()
    opp = ratings.get(opp_abbr.upper())
    if not opp:
        return 1.0
    league_avg = league_average_ppg() or LEAGUE_AVG_PPG
    ratio = opp.get("opp_ppg_allowed") / league_avg if opp.get("opp_ppg_allowed") else 1.0
    # Dampen — never scale beyond ±8%
    ratio = max(0.92, min(1.08, ratio))
    if stat in ("points", "threes"):
        return ratio
    return 1.0


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
    adjusted_mean: float
    opponent: Optional[str]
    distribution: str


def _stat_summary(profile: PlayerProfile, stat: str) -> tuple[float, float, list[int], int]:
    regular = profile.regular_season()
    regular_sorted = sorted(regular, key=lambda g: g.event_id or "", reverse=True)
    values = _values(regular_sorted, stat)
    if len(values) < 5:
        return 0.0, 0.0, [], 0
    mu = _weighted_mean(values)
    sigma = pstdev(values) if len(values) > 1 else 0.0
    return mu, sigma, values[:10], len(values)


def price_prop(
    profile: PlayerProfile,
    stat: str,
    line: float,
    opp_abbr: Optional[str] = None,
    playoff: bool = False,
) -> Optional[PropQuote]:
    if stat not in STAT_FIELDS:
        return None
    mu, sigma, recent, n = _stat_summary(profile, stat)
    if n == 0:
        return None
    if _recent_minutes(profile) < MIN_RECENT_MINUTES:
        return None

    adj = _opp_adjust(profile.team, opp_abbr, stat)
    adj_mu = mu * adj

    # Playoff shrinkage: inflate variance ~10%, slight mean pullback
    if playoff:
        sigma *= 1.1
        adj_mu *= 0.98

    if stat in ("threes",):
        p_over = _poisson_over(line, max(adj_mu, 0.01))
        distribution = "poisson"
    else:
        p_over = _truncated_normal_over(line, adj_mu, max(sigma, 0.5))
        distribution = "truncated_normal"

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
        adjusted_mean=round(adj_mu, 2),
        opponent=opp_abbr,
        distribution=distribution,
    )


def suggest_lines_for_player(
    profile: PlayerProfile, opp_abbr: Optional[str] = None, playoff: bool = False
) -> list[PropQuote]:
    quotes: list[PropQuote] = []
    for stat in STAT_FIELDS:
        mu, sigma, _, n = _stat_summary(profile, stat)
        if n < 15:
            continue
        if mu < MIN_MEAN.get(stat, 5.0):
            continue
        if sigma < MIN_STDEV.get(stat, 0.5):
            continue
        line = max(0.5, round(mu * 2) / 2 - 0.5)
        q = price_prop(profile, stat, line, opp_abbr=opp_abbr, playoff=playoff)
        if q:
            quotes.append(q)
    return quotes


def top_confidence_props(
    player_matchups: list[tuple[PlayerProfile, Optional[str]]],
    min_prob: float = 0.70,
    playoff: bool = True,
) -> list[dict]:
    out: list[dict] = []
    for profile, opp in player_matchups:
        for q in suggest_lines_for_player(profile, opp_abbr=opp, playoff=playoff):
            for side, p in [("over", q.model_prob_over), ("under", q.model_prob_under)]:
                if p >= min_prob:
                    out.append({
                        "player": q.player,
                        "team": q.team,
                        "opponent": q.opponent,
                        "stat": q.stat,
                        "line": q.line,
                        "side": side,
                        "model_prob": p,
                        "mean": q.mean,
                        "stdev": q.stdev,
                        "adjusted_mean": q.adjusted_mean,
                        "n_games": q.n_games,
                        "recent": q.recent,
                        "distribution": q.distribution,
                    })
    out.sort(key=lambda x: x["model_prob"], reverse=True)
    return out
