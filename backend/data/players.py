"""Player game-log fetcher using ESPN's public athlete API.

Each team roster has ~15 players; we keep only the top 8 by minutes-per-game
(heuristic: last-N-game average) for prop-relevance. No key required.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

CACHE_DIR = Path(__file__).parent / "cache" / "players"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{abbr}/roster"
GAMELOG_URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{pid}/gamelog"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

TEAM_ABBR_OVERRIDE = {"utah jazz": "utah", "utah": "utah"}


@dataclass
class PlayerGame:
    season_type: str
    event_id: str | None
    minutes: int
    points: int
    rebounds: int
    assists: int
    threes: int
    steals: int
    blocks: int


@dataclass
class PlayerProfile:
    id: str
    name: str
    team: str
    position: str = ""
    games: list[PlayerGame] = field(default_factory=list)

    def regular_season(self) -> list[PlayerGame]:
        return [g for g in self.games if "regular" in g.season_type.lower()]

    def postseason(self) -> list[PlayerGame]:
        return [g for g in self.games if "post" in g.season_type.lower()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "team": self.team,
            "position": self.position,
            "games": [g.__dict__ for g in self.games],
        }


async def _roster(client: httpx.AsyncClient, abbr: str) -> list[dict]:
    url = ROSTER_URL.format(abbr=abbr.lower())
    r = await client.get(url)
    r.raise_for_status()
    d = r.json()
    athletes = d.get("athletes", [])
    flat: list[dict] = []
    if athletes and isinstance(athletes[0], dict) and "items" in athletes[0]:
        for grp in athletes:
            flat.extend(grp.get("items", []))
    else:
        flat = list(athletes)
    return flat


async def _gamelog(client: httpx.AsyncClient, pid: str, season: str = "2026") -> list[PlayerGame]:
    url = GAMELOG_URL.format(pid=pid)
    try:
        r = await client.get(url, params={"season": season})
        r.raise_for_status()
    except Exception:
        return []
    d = r.json()
    labels = d.get("names", [])
    games: list[PlayerGame] = []
    if not labels:
        return games
    try:
        idx = {n: labels.index(n) for n in [
            "minutes",
            "points",
            "totalRebounds",
            "assists",
            "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
            "steals",
            "blocks",
        ]}
    except ValueError:
        return games

    def _int(s: Any, default: int = 0) -> int:
        try:
            return int(str(s).split("-")[0])
        except (ValueError, IndexError, TypeError):
            return default

    for st in d.get("seasonTypes", []):
        st_name = st.get("displayName") or st.get("name") or ""
        for cat in st.get("categories", []):
            for ev in cat.get("events", []):
                stats = ev.get("stats", [])
                if not stats:
                    continue
                games.append(PlayerGame(
                    season_type=st_name,
                    event_id=str(ev.get("eventId")) if ev.get("eventId") is not None else None,
                    minutes=_int(stats[idx["minutes"]]),
                    points=_int(stats[idx["points"]]),
                    rebounds=_int(stats[idx["totalRebounds"]]),
                    assists=_int(stats[idx["assists"]]),
                    threes=_int(stats[idx["threePointFieldGoalsMade-threePointFieldGoalsAttempted"]]),
                    steals=_int(stats[idx["steals"]]),
                    blocks=_int(stats[idx["blocks"]]),
                ))
    return games


async def fetch_team_players(
    team_abbr: str, top_n: int = 8, season: str = "2026"
) -> list[PlayerProfile]:
    abbr = TEAM_ABBR_OVERRIDE.get(team_abbr.lower(), team_abbr.lower())
    cache_path = CACHE_DIR / f"{abbr}_{season}.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        profiles = []
        for p in raw:
            games = [PlayerGame(**g) for g in p["games"]]
            profiles.append(
                PlayerProfile(id=p["id"], name=p["name"], team=p["team"],
                              position=p.get("position", ""), games=games)
            )
        return profiles[:top_n]

    async with httpx.AsyncClient(timeout=20.0, headers=HEADERS, follow_redirects=True) as client:
        roster = await _roster(client, abbr)
        if not roster:
            return []
        gamelogs = await asyncio.gather(
            *[_gamelog(client, str(p["id"]), season) for p in roster],
            return_exceptions=True,
        )

    profiles: list[PlayerProfile] = []
    for p, gl in zip(roster, gamelogs):
        if isinstance(gl, Exception):
            gl = []
        pos = ""
        if isinstance(p.get("position"), dict):
            pos = p["position"].get("abbreviation", "")
        profiles.append(
            PlayerProfile(
                id=str(p.get("id")),
                name=p.get("displayName") or p.get("fullName") or "",
                team=abbr.upper(),
                position=pos,
                games=gl,
            )
        )

    profiles.sort(
        key=lambda pr: (
            sum(g.minutes for g in pr.regular_season()[:15]) / max(len(pr.regular_season()[:15]), 1)
        ),
        reverse=True,
    )
    cache_path.write_text(json.dumps([p.to_dict() for p in profiles]))
    return profiles[:top_n]
