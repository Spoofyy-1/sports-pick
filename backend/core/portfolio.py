"""Simulated AI-managed parlay portfolio.

Starts with $1000. Each tick grades any settled bets (all legs final) and
places new +EV parlays from the live parlay generator, staking a small
fraction of the current bankroll per bet.

State is persisted to a JSON file (PORTFOLIO_PATH env override, else
backend/data/portfolio.json). Railway's filesystem is ephemeral across
deploys — the portfolio resets when the container is recreated. That's
fine for a demo; to harden it, swap the JSON load/save for a DB.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .ev import american_to_decimal

STARTING_BALANCE = 1000.0
STAKE_FRACTION = 0.02          # stake 2% of current bankroll per parlay
MAX_OPEN_BETS = 3
MIN_BALANCE_TO_BET = 50.0
MIN_TICK_INTERVAL = 60.0        # seconds — soft guard against spammy ticks
DEFAULT_LEGS = 3
CANDIDATE_PARLAYS_TO_CONSIDER = 5


def _default_path() -> Path:
    override = os.getenv("PORTFOLIO_PATH")
    if override:
        return Path(override)
    return Path(__file__).parent.parent / "data" / "portfolio.json"


class PortfolioManager:
    def __init__(self, path: Path | None = None):
        self.path = path or _default_path()
        self._state = self._load()

    # ---------- persistence ----------

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                pass
        return self._fresh_state()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2))

    @staticmethod
    def _fresh_state() -> dict:
        return {
            "starting_balance": STARTING_BALANCE,
            "balance": STARTING_BALANCE,
            "open_bets": [],
            "closed_bets": [],
            "last_tick_ts": 0.0,
            "created_ts": time.time(),
        }

    # ---------- public API ----------

    def reset(self) -> dict:
        self._state = self._fresh_state()
        self._save()
        return self.summary()

    def summary(self) -> dict:
        s = self._state
        closed = s["closed_bets"]
        wins = sum(1 for b in closed if b["status"] == "won")
        losses = sum(1 for b in closed if b["status"] == "lost")
        staked = sum(b["stake"] for b in closed)
        returned = sum(b.get("payout", 0.0) for b in closed)
        pnl = s["balance"] - s["starting_balance"]
        roi = pnl / s["starting_balance"] if s["starting_balance"] else 0.0
        return {
            "starting_balance": s["starting_balance"],
            "balance": round(s["balance"], 2),
            "pnl": round(pnl, 2),
            "roi": round(roi, 4),
            "open_bets": s["open_bets"],
            "closed_bets": sorted(
                closed, key=lambda b: b.get("settled_ts", 0), reverse=True
            )[:25],
            "total_bets_placed": len(closed) + len(s["open_bets"]),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / (wins + losses), 4) if (wins + losses) else None,
            "total_staked": round(staked, 2),
            "total_returned": round(returned, 2),
            "last_tick_ts": s["last_tick_ts"],
        }

    def tick(self, games: list[dict], candidate_parlays: list[dict]) -> dict:
        """Grade any settled bets, then consider placing new ones.

        games: list from _load_games() — used to determine winners
        candidate_parlays: output of /parlays/suggested — pre-ranked by EV
        """
        now = time.time()
        # soft rate limit
        if now - self._state["last_tick_ts"] < MIN_TICK_INTERVAL:
            return {"skipped": "too soon", **self.summary()}

        self._grade_open_bets(games)
        self._place_new_bets(candidate_parlays)
        self._state["last_tick_ts"] = now
        self._save()
        return self.summary()

    # ---------- grading ----------

    def _grade_open_bets(self, games: list[dict]) -> None:
        by_id = {g.get("event_id"): g for g in games}
        still_open: list[dict] = []
        for bet in self._state["open_bets"]:
            outcome = _grade_bet(bet, by_id)
            if outcome is None:
                still_open.append(bet)
                continue
            bet["status"] = outcome["status"]
            bet["payout"] = outcome["payout"]
            bet["settled_ts"] = time.time()
            bet["leg_results"] = outcome["leg_results"]
            self._state["balance"] += outcome["payout"]
            self._state["closed_bets"].append(bet)
        self._state["open_bets"] = still_open

    # ---------- placing ----------

    def _place_new_bets(self, candidate_parlays: list[dict]) -> None:
        if self._state["balance"] < MIN_BALANCE_TO_BET:
            return
        open_event_ids = {
            leg.get("event_id")
            for b in self._state["open_bets"]
            for leg in b.get("legs", [])
        }
        slots = MAX_OPEN_BETS - len(self._state["open_bets"])
        if slots <= 0:
            return

        for parlay in candidate_parlays[:CANDIDATE_PARLAYS_TO_CONSIDER]:
            if slots <= 0:
                break
            if parlay.get("ev_per_dollar", 0) <= 0:
                continue
            leg_event_ids = {leg.get("event_id") for leg in parlay.get("legs", [])}
            if not all(leg_event_ids):
                continue                          # can't grade if any leg lacks event_id
            if leg_event_ids & open_event_ids:
                continue                          # avoid overlap with open bets

            stake = round(self._state["balance"] * STAKE_FRACTION, 2)
            if stake < 1.0 or stake > self._state["balance"]:
                continue

            bet = {
                "id": uuid.uuid4().hex[:10],
                "placed_ts": time.time(),
                "stake": stake,
                "combined_american": parlay["combined_american"],
                "combined_decimal": parlay["combined_decimal"],
                "model_prob": parlay["model_prob"],
                "ev_per_dollar_at_placement": parlay["ev_per_dollar"],
                "potential_win": round(
                    stake * (parlay["combined_decimal"] - 1), 2
                ),
                "legs": [
                    {
                        "label": leg["label"],
                        "event_id": leg.get("event_id"),
                        "team": leg.get("team"),
                        "american": leg["american"],
                        "true_prob": leg.get("true_prob"),
                    }
                    for leg in parlay["legs"]
                ],
                "status": "open",
            }
            self._state["balance"] -= stake         # reserve stake
            self._state["open_bets"].append(bet)
            open_event_ids |= leg_event_ids
            slots -= 1


def _grade_bet(bet: dict, by_event_id: dict[str, dict]) -> dict | None:
    """Return {'status','payout','leg_results'} if settleable, else None."""
    leg_results = []
    for leg in bet["legs"]:
        g = by_event_id.get(leg.get("event_id"))
        if not g or g.get("status") != "final":
            return None                               # not all legs settled yet
        home_won = g.get("home_won")
        if home_won is None:
            # final but no winner field — fall back on score
            hs, as_ = g.get("home_score"), g.get("away_score")
            if hs is None or as_ is None:
                return None
            home_won = hs > as_
        winner = g["home_team"] if home_won else g["away_team"]
        leg_won = leg.get("team") == winner
        leg_results.append(
            {
                "event_id": leg["event_id"],
                "team": leg["team"],
                "winner": winner,
                "leg_won": leg_won,
            }
        )

    all_won = all(r["leg_won"] for r in leg_results)
    if all_won:
        # payout returns stake × decimal (stake was already deducted at placement)
        payout = round(bet["stake"] * bet["combined_decimal"], 2)
        return {"status": "won", "payout": payout, "leg_results": leg_results}
    return {"status": "lost", "payout": 0.0, "leg_results": leg_results}


# Singleton accessor — keeps a single instance per process
_manager: PortfolioManager | None = None


def get_portfolio() -> PortfolioManager:
    global _manager
    if _manager is None:
        _manager = PortfolioManager()
    return _manager
