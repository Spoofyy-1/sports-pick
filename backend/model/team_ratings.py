"""Derive team offensive/defensive ratings from the scoreboard cache.

No extra HTTP calls — reuses the cache we already have for training. For
each team in the last `window_days` of games, compute avg points scored
and avg points allowed. We expose these for prop opponent-adjustment.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Optional

from data.history import load_cached

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO", "New York Knicks": "NY",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SA", "Toronto Raptors": "TOR", "Utah Jazz": "UTAH",
    "Washington Wizards": "WSH",
}

_cache: dict | None = None


def _compute(window_days: int = 180) -> dict:
    """Walk the last `window_days` of cached games and aggregate per-team stats."""
    end = date.today()
    start = end - timedelta(days=window_days)
    games = load_cached(start, end)

    by_team: dict[str, dict] = defaultdict(lambda: {"pts_for": 0, "pts_against": 0, "n": 0})
    for g in games:
        h_abbr = TEAM_NAME_TO_ABBR.get(g.home_team)
        a_abbr = TEAM_NAME_TO_ABBR.get(g.away_team)
        if h_abbr:
            s = by_team[h_abbr]
            s["pts_for"] += g.home_score
            s["pts_against"] += g.away_score
            s["n"] += 1
        if a_abbr:
            s = by_team[a_abbr]
            s["pts_for"] += g.away_score
            s["pts_against"] += g.home_score
            s["n"] += 1

    out: dict[str, dict] = {}
    league_ppg_total, total_n = 0, 0
    for abbr, s in by_team.items():
        if s["n"] < 10:
            continue
        ppg = s["pts_for"] / s["n"]
        opp_ppg = s["pts_against"] / s["n"]
        league_ppg_total += ppg * s["n"]
        total_n += s["n"]
        out[abbr] = {
            "ppg_scored": round(ppg, 2),
            "opp_ppg_allowed": round(opp_ppg, 2),
            "games": s["n"],
        }
    league_avg = league_ppg_total / total_n if total_n else 113.0
    out["__league_avg__"] = {"ppg": round(league_avg, 2), "n": total_n}
    return out


def team_ratings() -> dict:
    global _cache
    if _cache is None:
        _cache = _compute()
    return _cache


def reload() -> None:
    global _cache
    _cache = None


def league_average_ppg() -> Optional[float]:
    r = team_ratings().get("__league_avg__")
    return r["ppg"] if r else None
