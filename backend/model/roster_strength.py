"""Team roster strength index derived from player game logs.

For each team, summing the top-8 players by minutes per game gives a
"who's actually on the floor" score. Multiplying minutes × PPG gives
scoring contribution. We then subtract out injured players (from the
live ESPN feed) and compare to the team's season baseline to produce
a `roster_strength_delta` — how much this team's expected output shifts
when injuries are accounted for.

Star-player minutes trend: for a team's top scorer, check the last-5
game minutes trend. A sharp drop often signals load management or an
unreported injury.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Optional

from data.injuries import InjuryReport
from data.players import PlayerProfile


@dataclass
class PlayerContribution:
    name: str
    player_id: str
    mpg: float           # minutes per game (last 10 regular season)
    ppg: float           # points per game
    score_contribution: float  # mpg/48 * ppg — normalized contribution


@dataclass
class RosterStrength:
    team_abbr: str
    top_contributors: list[PlayerContribution]
    baseline_score: float      # sum of top-8 contributions
    injured_names: list[str]
    injured_contribution: float
    effective_score: float     # baseline minus injured
    strength_delta: float      # effective - baseline, negative if hurt
    top_scorer_minutes_trend: float  # slope of last-5 game minutes (neg = declining)
    top_scorer_name: str


def _recent_values(profile: PlayerProfile, field: str, n: int = 10) -> list[float]:
    regular = profile.regular_season()
    regular_sorted = sorted(regular, key=lambda g: g.event_id or "", reverse=True)
    return [float(getattr(g, field)) for g in regular_sorted[:n] if g.minutes >= 10]


def _contribution(profile: PlayerProfile) -> Optional[PlayerContribution]:
    mins = _recent_values(profile, "minutes", n=10)
    pts = _recent_values(profile, "points", n=10)
    if not mins or len(mins) < 3:
        return None
    mpg = mean(mins)
    ppg = mean(pts) if pts else 0.0
    return PlayerContribution(
        name=profile.name,
        player_id=profile.id,
        mpg=round(mpg, 2),
        ppg=round(ppg, 2),
        score_contribution=round((mpg / 48.0) * ppg, 2),
    )


def _minutes_trend(profile: PlayerProfile) -> float:
    """Simple slope over last 5 games: +1/game = +1.0 slope."""
    mins = _recent_values(profile, "minutes", n=5)
    if len(mins) < 3:
        return 0.0
    # reverse so index increases with time
    mins = list(reversed(mins))
    n = len(mins)
    x_mean = (n - 1) / 2
    y_mean = sum(mins) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(mins))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return round(num / den, 3)


def compute_roster_strength(
    profiles: list[PlayerProfile],
    injury_reports: list[InjuryReport],
    team_abbr: str,
) -> RosterStrength:
    """Score a team's playable roster given current injury state."""
    contribs: list[PlayerContribution] = []
    for p in profiles[:8]:
        c = _contribution(p)
        if c:
            contribs.append(c)
    contribs.sort(key=lambda c: c.score_contribution, reverse=True)

    injured_ids = {r.player_id for r in injury_reports if r.severity >= 0.5}
    injured_names_hit: list[str] = []
    injured_score = 0.0
    for c in contribs:
        if c.player_id in injured_ids:
            injured_names_hit.append(c.name)
            # scale contribution by severity (use highest severity found)
            sev = max((r.severity for r in injury_reports if r.player_id == c.player_id), default=1.0)
            injured_score += c.score_contribution * sev

    baseline = sum(c.score_contribution for c in contribs)
    effective = baseline - injured_score

    top_scorer = contribs[0] if contribs else None
    top_scorer_profile = next((p for p in profiles if top_scorer and p.id == top_scorer.player_id), None)
    trend = _minutes_trend(top_scorer_profile) if top_scorer_profile else 0.0

    return RosterStrength(
        team_abbr=team_abbr.upper(),
        top_contributors=contribs,
        baseline_score=round(baseline, 2),
        injured_names=injured_names_hit,
        injured_contribution=round(injured_score, 2),
        effective_score=round(effective, 2),
        strength_delta=round(effective - baseline, 2),
        top_scorer_minutes_trend=trend,
        top_scorer_name=top_scorer.name if top_scorer else "",
    )


def win_prob_adjustment(home_rs: RosterStrength, away_rs: RosterStrength) -> float:
    """Convert roster strength delta into a win-probability nudge.

    Rough heuristic: 1 point of per-game scoring ≈ 2% win prob shift
    (roughly NBA spread-to-moneyline rule-of-thumb). We clamp to ±15%.
    """
    home_delta = home_rs.strength_delta  # usually <= 0 (injured players cost)
    away_delta = away_rs.strength_delta
    net = (home_delta - away_delta) * 0.02
    return max(-0.15, min(0.15, net))
