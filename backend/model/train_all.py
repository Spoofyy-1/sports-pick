"""Master training: retrain Elo + pull player game logs for every team with
an upcoming game. Run this to refresh everything."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from data.players import fetch_team_players
from scrapers.odds import fetch_nba_moneylines
from model import train as elo_train

OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

TEAM_ABBR = {
    "Atlanta Hawks": "atl", "Boston Celtics": "bos", "Brooklyn Nets": "bkn",
    "Charlotte Hornets": "cha", "Chicago Bulls": "chi", "Cleveland Cavaliers": "cle",
    "Dallas Mavericks": "dal", "Denver Nuggets": "den", "Detroit Pistons": "det",
    "Golden State Warriors": "gs", "Houston Rockets": "hou", "Indiana Pacers": "ind",
    "LA Clippers": "lac", "Los Angeles Clippers": "lac", "Los Angeles Lakers": "lal",
    "Memphis Grizzlies": "mem", "Miami Heat": "mia", "Milwaukee Bucks": "mil",
    "Minnesota Timberwolves": "min", "New Orleans Pelicans": "no", "New York Knicks": "ny",
    "Oklahoma City Thunder": "okc", "Orlando Magic": "orl", "Philadelphia 76ers": "phi",
    "Phoenix Suns": "phx", "Portland Trail Blazers": "por", "Sacramento Kings": "sac",
    "San Antonio Spurs": "sa", "Toronto Raptors": "tor", "Utah Jazz": "utah",
    "Washington Wizards": "wsh",
}


async def main():
    print("=== Step 1: Retraining Elo ===\n")
    await elo_train.main(years=2)

    print("\n=== Step 2: Fetching rosters + game logs for teams with upcoming games ===\n")
    games = await fetch_nba_moneylines()
    teams_to_fetch: set[str] = set()
    for g in games:
        for name in (g.home_team, g.away_team):
            if name in TEAM_ABBR:
                teams_to_fetch.add(TEAM_ABBR[name])

    print(f"teams with games: {sorted(teams_to_fetch)}")

    total_players = 0
    for abbr in sorted(teams_to_fetch):
        try:
            profiles = await fetch_team_players(abbr, top_n=8)
            ngames = sum(len(p.games) for p in profiles)
            print(f"  {abbr}: {len(profiles)} players, {ngames} game rows")
            total_players += len(profiles)
        except Exception as e:
            print(f"  {abbr}: FAILED {e}")

    print(f"\n=== Done. Cached {total_players} player profiles across {len(teams_to_fetch)} teams ===")
    manifest = {
        "teams": sorted(teams_to_fetch),
        "players_cached": total_players,
    }
    (OUT_DIR / "train_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
