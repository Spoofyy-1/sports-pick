"""Pull historical NBA game results from ESPN's public scoreboard API.

No key required. Returns finished games with scores for any date range.
Results are cached to data/cache/<YYYYMMDD>.json to avoid refetching.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json",
}


@dataclass
class FinishedGame:
    date: str
    event_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_won: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def _cache_path(d: date) -> Path:
    return CACHE_DIR / f"{d.strftime('%Y%m%d')}.json"


FRESH_DAYS = 3  # recent days always refetch (games may still be live / just completed)


async def _fetch_day(client: httpx.AsyncClient, d: date, force: bool = False) -> dict:
    path = _cache_path(d)
    today = date.today()
    is_recent = (today - d).days < FRESH_DAYS
    if path.exists() and not force and not is_recent:
        return json.loads(path.read_text())
    r = await client.get(SCOREBOARD_URL, params={"dates": d.strftime("%Y%m%d")})
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data))
    return data


def _parse_day(raw: dict) -> list[FinishedGame]:
    out: list[FinishedGame] = []
    for ev in raw.get("events", []) or []:
        comp = (ev.get("competitions") or [{}])[0]
        status = (comp.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            continue
        home = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), None)
        away = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            hs = int(home.get("score"))
            as_ = int(away.get("score"))
        except (TypeError, ValueError):
            continue
        out.append(
            FinishedGame(
                date=ev.get("date", "")[:10],
                event_id=str(ev.get("id")),
                home_team=(home.get("team") or {}).get("displayName", ""),
                away_team=(away.get("team") or {}).get("displayName", ""),
                home_score=hs,
                away_score=as_,
                home_won=hs > as_,
            )
        )
    return out


async def fetch_range(start: date, end: date, concurrency: int = 8) -> list[FinishedGame]:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=20.0, headers=HEADERS, follow_redirects=True) as client:
        async def _one(d: date) -> list[FinishedGame]:
            async with sem:
                try:
                    raw = await _fetch_day(client, d)
                    return _parse_day(raw)
                except Exception:
                    return []

        chunks = await asyncio.gather(*[_one(d) for d in days])

    games = [g for chunk in chunks for g in chunk]
    games.sort(key=lambda g: g.date)
    return games


def load_cached(start: date, end: date) -> list[FinishedGame]:
    games: list[FinishedGame] = []
    d = start
    while d <= end:
        path = _cache_path(d)
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                games.extend(_parse_day(raw))
            except Exception:
                pass
        d += timedelta(days=1)
    games.sort(key=lambda g: g.date)
    return games
