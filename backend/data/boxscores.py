"""ESPN box score fetcher + cache.

For each event_id we fetch /summary once and extract per-team shooting splits.
Cached to data/cache/boxscores/{event_id}.json. ~200ms per event, safe to
batch-fetch with asyncio concurrency.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import httpx

CACHE_DIR = Path(__file__).parent / "cache" / "boxscores"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


@dataclass
class TeamBox:
    team: str
    home_away: str
    fgm: int
    fga: int
    tpm: int
    tpa: int
    ftm: int
    fta: int
    rebounds: int
    offensive_rebounds: int
    assists: int
    steals: int
    blocks: int
    turnovers: int
    fast_break_points: int
    points_in_paint: int
    score: int

    @property
    def fg_pct(self) -> float:
        return self.fgm / self.fga if self.fga else 0.0

    @property
    def tp_pct(self) -> float:
        return self.tpm / self.tpa if self.tpa else 0.0

    @property
    def ft_pct(self) -> float:
        return self.ftm / self.fta if self.fta else 0.0

    @property
    def efg_pct(self) -> float:
        """Effective FG% = (FGM + 0.5 * 3PM) / FGA — accounts for 3pt value."""
        return (self.fgm + 0.5 * self.tpm) / self.fga if self.fga else 0.0

    @property
    def ts_pct(self) -> float:
        """True shooting % = PTS / (2 * (FGA + 0.44 * FTA))."""
        denom = 2 * (self.fga + 0.44 * self.fta)
        return self.score / denom if denom else 0.0

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "home_away": self.home_away,
            "fgm": self.fgm, "fga": self.fga,
            "tpm": self.tpm, "tpa": self.tpa,
            "ftm": self.ftm, "fta": self.fta,
            "rebounds": self.rebounds,
            "offensive_rebounds": self.offensive_rebounds,
            "assists": self.assists,
            "steals": self.steals,
            "blocks": self.blocks,
            "turnovers": self.turnovers,
            "fast_break_points": self.fast_break_points,
            "points_in_paint": self.points_in_paint,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TeamBox":
        return cls(**d)


def _split_made_att(s: str) -> tuple[int, int]:
    try:
        m, a = s.split("-")
        return int(m), int(a)
    except (ValueError, AttributeError):
        return 0, 0


def _int_stat(stats: list[dict], name: str, default: int = 0) -> int:
    for s in stats:
        if s.get("name") == name:
            try:
                return int(s.get("displayValue", default))
            except (ValueError, TypeError):
                return default
    return default


def _parse_box(raw: dict) -> list[TeamBox]:
    teams_box = raw.get("boxscore", {}).get("teams", [])
    header_comp = raw.get("header", {}).get("competitions", [{}])[0]
    score_by_team = {}
    for c in header_comp.get("competitors", []):
        try:
            score_by_team[c.get("team", {}).get("displayName", "")] = int(c.get("score"))
        except (TypeError, ValueError):
            pass

    out: list[TeamBox] = []
    for tb in teams_box:
        stats = tb.get("statistics", [])
        name = tb.get("team", {}).get("displayName", "")
        fgm, fga = _split_made_att(next((s["displayValue"] for s in stats if s.get("name") == "fieldGoalsMade-fieldGoalsAttempted"), ""))
        tpm, tpa = _split_made_att(next((s["displayValue"] for s in stats if s.get("name") == "threePointFieldGoalsMade-threePointFieldGoalsAttempted"), ""))
        ftm, fta = _split_made_att(next((s["displayValue"] for s in stats if s.get("name") == "freeThrowsMade-freeThrowsAttempted"), ""))
        out.append(TeamBox(
            team=name,
            home_away=tb.get("homeAway", ""),
            fgm=fgm, fga=fga,
            tpm=tpm, tpa=tpa,
            ftm=ftm, fta=fta,
            rebounds=_int_stat(stats, "totalRebounds"),
            offensive_rebounds=_int_stat(stats, "offensiveRebounds"),
            assists=_int_stat(stats, "assists"),
            steals=_int_stat(stats, "steals"),
            blocks=_int_stat(stats, "blocks"),
            turnovers=_int_stat(stats, "turnovers"),
            fast_break_points=_int_stat(stats, "fastBreakPoints"),
            points_in_paint=_int_stat(stats, "pointsInPaint"),
            score=score_by_team.get(name, 0),
        ))
    return out


def _cache_path(event_id: str) -> Path:
    return CACHE_DIR / f"{event_id}.json"


def load_cached(event_id: str) -> Optional[list[TeamBox]]:
    p = _cache_path(event_id)
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    return [TeamBox.from_dict(t) for t in raw]


async def _fetch_one(client: httpx.AsyncClient, event_id: str, force: bool = False) -> Optional[list[TeamBox]]:
    cached = load_cached(event_id)
    if cached is not None and not force:
        return cached
    try:
        r = await client.get(SUMMARY_URL, params={"event": event_id})
        r.raise_for_status()
        raw = r.json()
    except Exception:
        return None
    boxes = _parse_box(raw)
    if boxes:
        _cache_path(event_id).write_text(json.dumps([b.to_dict() for b in boxes]))
    return boxes


async def fetch_box_scores(
    event_ids: Iterable[str], concurrency: int = 8, force: bool = False
) -> dict[str, list[TeamBox]]:
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, list[TeamBox]] = {}

    async with httpx.AsyncClient(timeout=20.0, headers=HEADERS, follow_redirects=True) as client:
        async def _one(eid: str):
            async with sem:
                boxes = await _fetch_one(client, eid, force)
                if boxes:
                    results[eid] = boxes

        await asyncio.gather(*[_one(e) for e in event_ids])
    return results
