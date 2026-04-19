"""ESPN injury feed.

Single endpoint returns every team's injury list. We cache the response
for 15 minutes (it updates slowly, no need to hammer it).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

CACHE_PATH = Path(__file__).parent / "cache" / "injuries.json"
CACHE_PATH.parent.mkdir(exist_ok=True)
TTL_SECONDS = 15 * 60

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# ESPN status IDs (higher = more severe)
STATUS_WEIGHT = {
    "Out": 1.0,
    "OUT": 1.0,
    "Out For Season": 1.0,
    "Doubtful": 0.75,
    "Questionable": 0.40,
    "Day-To-Day": 0.30,
    "Probable": 0.10,
}


@dataclass
class InjuryReport:
    player_id: str
    player_name: str
    team_abbr: str
    team_name: str
    status: str
    severity: float  # 0-1 impact weight
    short_comment: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


async def _fetch_raw(force: bool = False) -> dict:
    if CACHE_PATH.exists() and not force:
        try:
            raw = json.loads(CACHE_PATH.read_text())
            if time.time() - raw.get("_fetched_at", 0) < TTL_SECONDS:
                return raw
        except Exception:
            pass
    async with httpx.AsyncClient(timeout=20.0, headers=HEADERS, follow_redirects=True) as client:
        r = await client.get(URL)
        r.raise_for_status()
        data = r.json()
    data["_fetched_at"] = time.time()
    CACHE_PATH.write_text(json.dumps(data))
    return data


def _parse(raw: dict) -> list[InjuryReport]:
    from model.team_ratings import TEAM_NAME_TO_ABBR
    out: list[InjuryReport] = []
    for team in raw.get("injuries", []) or []:
        team_name = team.get("displayName") or team.get("name") or ""
        team_abbr = (team.get("abbreviation") or "").upper()
        if not team_abbr:
            team_abbr = TEAM_NAME_TO_ABBR.get(team_name, "").upper()
        for inj in team.get("injuries", []) or []:
            athlete = inj.get("athlete") or {}
            status = inj.get("status") or ""
            severity = STATUS_WEIGHT.get(status, 0.0)
            out.append(InjuryReport(
                player_id=str(athlete.get("id", "")),
                player_name=athlete.get("displayName", ""),
                team_abbr=team_abbr,
                team_name=team_name,
                status=status,
                severity=severity,
                short_comment=inj.get("shortComment", "") or inj.get("longComment", "") or "",
            ))
    return out


async def fetch_all(force: bool = False) -> list[InjuryReport]:
    raw = await _fetch_raw(force)
    return _parse(raw)


async def by_team(team_abbrs: list[str]) -> dict[str, list[InjuryReport]]:
    reports = await fetch_all()
    out: dict[str, list[InjuryReport]] = {a.upper(): [] for a in team_abbrs}
    for r in reports:
        if r.team_abbr in out:
            out[r.team_abbr].append(r)
    return out
