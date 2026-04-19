"""ESPN-sourced NBA moneylines (ESPN aggregates DraftKings lines).

Uses two public ESPN endpoints — no key required:
  - scoreboard (events for a date range)
  - core API odds (moneyline/spread/total per event)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
    "events/{eid}/competitions/{eid}/odds"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


@dataclass
class MoneylineGame:
    event_id: str
    name: str
    start_date: str
    home_team: str
    away_team: str
    home_ml: int | None
    away_ml: int | None
    spread: float | None
    total: float | None
    provider: str
    status: str = "scheduled"
    home_score: int | None = None
    away_score: int | None = None
    home_won: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _parse_american(s: Any) -> int | None:
    if s is None:
        return None
    try:
        return int(str(s).replace("−", "-").replace("+", "").strip())
    except (ValueError, TypeError):
        return None


async def _fetch_scoreboard(client: httpx.AsyncClient, date_yyyymmdd: str) -> list[dict]:
    r = await client.get(SCOREBOARD_URL, params={"dates": date_yyyymmdd})
    r.raise_for_status()
    return r.json().get("events", []) or []


async def _fetch_odds(client: httpx.AsyncClient, event_id: str) -> dict | None:
    try:
        r = await client.get(ODDS_URL.format(eid=event_id))
        r.raise_for_status()
        items = r.json().get("items", []) or []
        return items[0] if items else None
    except Exception:
        return None


def _extract_moneylines(odds_item: dict) -> tuple[int | None, int | None]:
    def ml(side: dict) -> int | None:
        current = (side or {}).get("current") or {}
        amer = (current.get("moneyLine") or {}).get("american")
        if amer is not None:
            return _parse_american(amer)
        raw = side.get("moneyLine")
        return _parse_american(raw)

    return ml(odds_item.get("homeTeamOdds") or {}), ml(odds_item.get("awayTeamOdds") or {})


def _status_name(ev: dict) -> str:
    comp = (ev.get("competitions") or [{}])[0]
    t = (comp.get("status") or {}).get("type") or {}
    name = t.get("name", "").lower()
    if "final" in name or t.get("completed"):
        return "final"
    if "in" in name or "progress" in name:
        return "live"
    return "scheduled"


def _score(ev: dict, side: str) -> int | None:
    comp = (ev.get("competitions") or [{}])[0]
    c = next((c for c in comp.get("competitors", []) if c.get("homeAway") == side), {})
    try:
        return int(c.get("score"))
    except (TypeError, ValueError):
        return None


async def fetch_nba_moneylines(days_back: int = 2, days_ahead: int = 3) -> list[MoneylineGame]:
    today = datetime.now(timezone.utc).date()
    dates = [
        (today + timedelta(days=i)).strftime("%Y%m%d")
        for i in range(-days_back, days_ahead + 1)
    ]

    async with httpx.AsyncClient(timeout=20.0, headers=HEADERS, follow_redirects=True) as client:
        scoreboards = await asyncio.gather(
            *[_fetch_scoreboard(client, d) for d in dates], return_exceptions=True
        )
        events: dict[str, dict] = {}
        for sb in scoreboards:
            if isinstance(sb, Exception):
                continue
            for ev in sb:
                events[str(ev["id"])] = ev

        odds_results = await asyncio.gather(
            *[_fetch_odds(client, eid) for eid in events],
            return_exceptions=True,
        )

    games: list[MoneylineGame] = []
    for (eid, ev), odds in zip(events.items(), odds_results):
        status = _status_name(ev)
        comp = (ev.get("competitions") or [{}])[0]
        home = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), {})
        away = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), {})

        home_ml, away_ml = (None, None)
        spread, total, provider = (None, None, "")
        if not isinstance(odds, Exception) and odds:
            home_ml, away_ml = _extract_moneylines(odds)
            spread = odds.get("spread")
            total = odds.get("overUnder")
            provider = (odds.get("provider") or {}).get("name", "")

        hs, as_ = _score(ev, "home"), _score(ev, "away")
        home_won = None
        if status == "final" and hs is not None and as_ is not None:
            home_won = hs > as_

        games.append(
            MoneylineGame(
                event_id=eid,
                name=ev.get("name", ""),
                start_date=ev.get("date", ""),
                home_team=(home.get("team") or {}).get("displayName", ""),
                away_team=(away.get("team") or {}).get("displayName", ""),
                home_ml=home_ml,
                away_ml=away_ml,
                spread=spread,
                total=total,
                provider=provider,
                status=status,
                home_score=hs,
                away_score=as_,
                home_won=home_won,
            )
        )
    games.sort(key=lambda g: g.start_date)
    return games
