"""Extended features using box-score data.

Adds opponent-specific shooting memory — the thing the baseline model misses.
If Team A historically shoots 29% from 3 against Team B, we now carry that
signal forward into their next meeting.

New features (on top of the original 12):
  - h_efg_L10, a_efg_L10                    rolling effective FG%
  - h_3p_pct_L10, a_3p_pct_L10              rolling 3P%
  - h_3pa_L10, a_3pa_L10                    rolling 3PA volume
  - h_efg_allowed_L10, a_efg_allowed_L10    rolling opponent eFG% allowed
  - h_vs_opp_efg_last3, a_vs_opp_efg_last3  eFG% in last 3 meetings vs this opp
  - h_vs_opp_3p_last3, a_vs_opp_3p_last3    3P% in last 3 meetings vs this opp
  - h_vs_opp_margin_last3                   avg point margin (home PoV) last 3
  - h_momentum_L5                           sum of point differentials last 5
  - a_momentum_L5

All computed walk-forward: features for game G use only games strictly before G.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Optional

from data.boxscores import TeamBox
from model.elo import EloState

EXTENDED_FEATURE_NAMES = [
    # Base 12
    "elo_diff", "h_rest", "a_rest", "h_b2b", "a_b2b",
    "h_ortg_L10", "a_ortg_L10", "h_drtg_L10", "a_drtg_L10",
    "h_form_L10", "a_form_L10", "h2h_home_rate",
    # New shooting rolling (+8)
    "h_efg_L10", "a_efg_L10",
    "h_3p_pct_L10", "a_3p_pct_L10",
    "h_3pa_L10", "a_3pa_L10",
    "h_efg_allowed_L10", "a_efg_allowed_L10",
    # Opponent-specific memory (+5)
    "h_vs_opp_efg_last3", "a_vs_opp_efg_last3",
    "h_vs_opp_3p_last3", "a_vs_opp_3p_last3",
    "h_vs_opp_margin_last3",
    # Hot/cold momentum (+2)
    "h_momentum_L5", "a_momentum_L5",
]


@dataclass
class TeamRoll:
    last_game_date: Optional[date] = None
    pts_for: deque = field(default_factory=lambda: deque(maxlen=10))
    pts_against: deque = field(default_factory=lambda: deque(maxlen=10))
    wins: deque = field(default_factory=lambda: deque(maxlen=10))
    efg: deque = field(default_factory=lambda: deque(maxlen=10))
    tp_pct: deque = field(default_factory=lambda: deque(maxlen=10))
    tpa: deque = field(default_factory=lambda: deque(maxlen=10))
    efg_allowed: deque = field(default_factory=lambda: deque(maxlen=10))
    margins: deque = field(default_factory=lambda: deque(maxlen=5))  # for momentum


@dataclass
class H2HRoll:
    """Per-team-pair rolling history of actual box-score splits."""
    meetings: deque = field(default_factory=lambda: deque(maxlen=3))
    # each entry: {"home": name, "home_efg": x, "away_efg": y, "home_3p": z, "away_3p": w, "margin": home_pts - away_pts}


def _parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def _mean(xs, default):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else default


def _rest(gd: date, last: Optional[date]) -> tuple[float, int]:
    if last is None:
        return 3.0, 0
    days = (gd - last).days
    return float(min(days, 5)), 1 if days <= 1 else 0


def _h2h_vs_opp(team: str, meetings, metric: str) -> float:
    """metric ∈ {'efg', '3p', 'margin_home_pov'}"""
    if not meetings:
        if metric == "margin_home_pov":
            return 0.0
        return 0.45 if metric == "efg" else 0.35  # league-avg priors
    vals = []
    for m in meetings:
        if metric == "margin_home_pov":
            vals.append(m["margin"])
        else:
            # pick the row for this specific team
            if m["home"] == team:
                vals.append(m[f"home_{metric}"])
            else:
                vals.append(m[f"away_{metric}"])
    return sum(vals) / len(vals) if vals else 0.0


def build_training_matrix(
    games,
    box_scores: dict[str, list[TeamBox]],
    hca_elo: float = 100.0,
) -> tuple[list[list[float]], list[int], list[dict]]:
    elo = EloState()
    roll: dict[str, TeamRoll] = defaultdict(TeamRoll)
    h2h: dict[tuple[str, str], H2HRoll] = defaultdict(H2HRoll)

    X, y, meta = [], [], []
    league_ppg = 113.0

    for g in games:
        try:
            gd = _parse_date(g.date)
        except Exception:
            continue

        h, a = g.home_team, g.away_team
        hr, ar = roll[h], roll[a]

        h_rest, h_b2b = _rest(gd, hr.last_game_date)
        a_rest, a_b2b = _rest(gd, ar.last_game_date)
        elo_diff = (elo.rating(h) + hca_elo) - elo.rating(a)

        h_ortg = _mean(hr.pts_for, league_ppg)
        a_ortg = _mean(ar.pts_for, league_ppg)
        h_drtg = _mean(hr.pts_against, league_ppg)
        a_drtg = _mean(ar.pts_against, league_ppg)
        h_form = _mean(hr.wins, 0.5)
        a_form = _mean(ar.wins, 0.5)
        h_efg = _mean(hr.efg, 0.52)
        a_efg = _mean(ar.efg, 0.52)
        h_3p = _mean(hr.tp_pct, 0.36)
        a_3p = _mean(ar.tp_pct, 0.36)
        h_3pa = _mean(hr.tpa, 35.0)
        a_3pa = _mean(ar.tpa, 35.0)
        h_efg_allowed = _mean(hr.efg_allowed, 0.52)
        a_efg_allowed = _mean(ar.efg_allowed, 0.52)
        h_mom = sum(hr.margins) if hr.margins else 0.0
        a_mom = sum(ar.margins) if ar.margins else 0.0

        pair_key = tuple(sorted([h, a]))
        pair_meetings = h2h[pair_key].meetings
        h_vs_opp_efg = _h2h_vs_opp(h, pair_meetings, "efg")
        a_vs_opp_efg = _h2h_vs_opp(a, pair_meetings, "efg")
        h_vs_opp_3p = _h2h_vs_opp(h, pair_meetings, "3p")
        a_vs_opp_3p = _h2h_vs_opp(a, pair_meetings, "3p")
        # margin from home team's perspective — convert historical margins
        if pair_meetings:
            margins_home_pov = []
            for m in pair_meetings:
                if m["home"] == h:
                    margins_home_pov.append(m["margin"])
                else:
                    margins_home_pov.append(-m["margin"])
            h_vs_opp_margin = sum(margins_home_pov) / len(margins_home_pov)
        else:
            h_vs_opp_margin = 0.0

        # Old H2H win rate (last-5)
        win_meetings = list(h2h[pair_key].meetings)
        if len(win_meetings) >= 2:
            h2h_home_rate = sum(
                1 for m in win_meetings
                if (m["home"] == h and m["margin"] > 0) or (m["home"] == a and m["margin"] < 0)
            ) / len(win_meetings)
        else:
            h2h_home_rate = 0.5

        ready = (len(hr.pts_for) >= 5 and len(ar.pts_for) >= 5
                 and len(hr.efg) >= 3 and len(ar.efg) >= 3)

        if ready:
            X.append([
                elo_diff,
                h_rest, a_rest, float(h_b2b), float(a_b2b),
                h_ortg, a_ortg, h_drtg, a_drtg,
                h_form, a_form, h2h_home_rate,
                h_efg, a_efg,
                h_3p, a_3p,
                h_3pa, a_3pa,
                h_efg_allowed, a_efg_allowed,
                h_vs_opp_efg, a_vs_opp_efg,
                h_vs_opp_3p, a_vs_opp_3p,
                h_vs_opp_margin,
                h_mom, a_mom,
            ])
            y.append(1 if g.home_won else 0)
            meta.append({"date": g.date, "home": h, "away": a,
                         "event_id": g.event_id,
                         "home_score": g.home_score, "away_score": g.away_score})

        # === Update state AFTER snapshot ===
        hr.pts_for.append(g.home_score)
        hr.pts_against.append(g.away_score)
        hr.wins.append(1.0 if g.home_won else 0.0)
        hr.last_game_date = gd
        hr.margins.append(g.home_score - g.away_score)
        ar.pts_for.append(g.away_score)
        ar.pts_against.append(g.home_score)
        ar.wins.append(0.0 if g.home_won else 1.0)
        ar.last_game_date = gd
        ar.margins.append(g.away_score - g.home_score)

        boxes = box_scores.get(g.event_id)
        if boxes and len(boxes) == 2:
            hbox = next((b for b in boxes if b.home_away == "home"), None)
            abox = next((b for b in boxes if b.home_away == "away"), None)
            if hbox and abox:
                hr.efg.append(hbox.efg_pct)
                hr.tp_pct.append(hbox.tp_pct)
                hr.tpa.append(float(hbox.tpa))
                hr.efg_allowed.append(abox.efg_pct)
                ar.efg.append(abox.efg_pct)
                ar.tp_pct.append(abox.tp_pct)
                ar.tpa.append(float(abox.tpa))
                ar.efg_allowed.append(hbox.efg_pct)

                h2h[pair_key].meetings.append({
                    "home": h,
                    "home_efg": hbox.efg_pct,
                    "away_efg": abox.efg_pct,
                    "home_3p": hbox.tp_pct,
                    "away_3p": abox.tp_pct,
                    "margin": g.home_score - g.away_score,
                })

        elo.update(g.home_team, g.away_team, g.home_score, g.away_score)

    return X, y, meta


@dataclass
class LiveStateV2:
    elo: EloState
    roll: dict
    h2h: dict

    def features_for(self, home: str, away: str, gd: date, hca_elo: float = 100.0) -> list[float]:
        hr = self.roll.get(home, TeamRoll())
        ar = self.roll.get(away, TeamRoll())
        h_rest, h_b2b = _rest(gd, hr.last_game_date)
        a_rest, a_b2b = _rest(gd, ar.last_game_date)
        elo_diff = (self.elo.rating(home) + hca_elo) - self.elo.rating(away)
        pair_key = tuple(sorted([home, away]))
        pair_meetings = (self.h2h.get(pair_key) or H2HRoll()).meetings
        h_vs_opp_efg = _h2h_vs_opp(home, pair_meetings, "efg")
        a_vs_opp_efg = _h2h_vs_opp(away, pair_meetings, "efg")
        h_vs_opp_3p = _h2h_vs_opp(home, pair_meetings, "3p")
        a_vs_opp_3p = _h2h_vs_opp(away, pair_meetings, "3p")
        if pair_meetings:
            margins_home_pov = []
            for m in pair_meetings:
                margins_home_pov.append(m["margin"] if m["home"] == home else -m["margin"])
            h_vs_opp_margin = sum(margins_home_pov) / len(margins_home_pov)
        else:
            h_vs_opp_margin = 0.0
        win_meetings = list(pair_meetings)
        if len(win_meetings) >= 2:
            h2h_home_rate = sum(
                1 for m in win_meetings
                if (m["home"] == home and m["margin"] > 0) or (m["home"] == away and m["margin"] < 0)
            ) / len(win_meetings)
        else:
            h2h_home_rate = 0.5

        return [
            elo_diff, h_rest, a_rest, float(h_b2b), float(a_b2b),
            _mean(hr.pts_for, 113.0), _mean(ar.pts_for, 113.0),
            _mean(hr.pts_against, 113.0), _mean(ar.pts_against, 113.0),
            _mean(hr.wins, 0.5), _mean(ar.wins, 0.5),
            h2h_home_rate,
            _mean(hr.efg, 0.52), _mean(ar.efg, 0.52),
            _mean(hr.tp_pct, 0.36), _mean(ar.tp_pct, 0.36),
            _mean(hr.tpa, 35.0), _mean(ar.tpa, 35.0),
            _mean(hr.efg_allowed, 0.52), _mean(ar.efg_allowed, 0.52),
            h_vs_opp_efg, a_vs_opp_efg,
            h_vs_opp_3p, a_vs_opp_3p,
            h_vs_opp_margin,
            sum(hr.margins) if hr.margins else 0.0,
            sum(ar.margins) if ar.margins else 0.0,
        ]


def build_live_state(games, box_scores: dict[str, list[TeamBox]]) -> LiveStateV2:
    elo = EloState()
    roll: dict[str, TeamRoll] = defaultdict(TeamRoll)
    h2h: dict[tuple[str, str], H2HRoll] = defaultdict(H2HRoll)
    for g in games:
        try:
            gd = _parse_date(g.date)
        except Exception:
            continue
        hr, ar = roll[g.home_team], roll[g.away_team]
        hr.pts_for.append(g.home_score); hr.pts_against.append(g.away_score)
        hr.wins.append(1.0 if g.home_won else 0.0); hr.last_game_date = gd
        hr.margins.append(g.home_score - g.away_score)
        ar.pts_for.append(g.away_score); ar.pts_against.append(g.home_score)
        ar.wins.append(0.0 if g.home_won else 1.0); ar.last_game_date = gd
        ar.margins.append(g.away_score - g.home_score)

        boxes = box_scores.get(g.event_id)
        if boxes and len(boxes) == 2:
            hbox = next((b for b in boxes if b.home_away == "home"), None)
            abox = next((b for b in boxes if b.home_away == "away"), None)
            if hbox and abox:
                hr.efg.append(hbox.efg_pct); hr.tp_pct.append(hbox.tp_pct); hr.tpa.append(float(hbox.tpa))
                hr.efg_allowed.append(abox.efg_pct)
                ar.efg.append(abox.efg_pct); ar.tp_pct.append(abox.tp_pct); ar.tpa.append(float(abox.tpa))
                ar.efg_allowed.append(hbox.efg_pct)
                pair_key = tuple(sorted([g.home_team, g.away_team]))
                h2h[pair_key].meetings.append({
                    "home": g.home_team,
                    "home_efg": hbox.efg_pct,
                    "away_efg": abox.efg_pct,
                    "home_3p": hbox.tp_pct,
                    "away_3p": abox.tp_pct,
                    "margin": g.home_score - g.away_score,
                })

        elo.update(g.home_team, g.away_team, g.home_score, g.away_score)
    return LiveStateV2(elo=elo, roll=dict(roll), h2h=dict(h2h))
