"""Train Elo on historical NBA games pulled from ESPN and save ratings."""
from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path

from data.history import fetch_range
from model.elo import EloState, backtest_metrics, train

OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(exist_ok=True)


async def main(years: int = 2) -> None:
    today = date.today()
    start = today - timedelta(days=365 * years)
    print(f"fetching games {start} → {today} from ESPN (cached per-day)…")
    games = await fetch_range(start, today, concurrency=8)
    print(f"got {len(games)} finished games")
    if not games:
        print("no games fetched; aborting")
        return

    split = int(len(games) * 0.8)
    train_games, test_games = games[:split], games[split:]

    print(f"training Elo on {len(train_games)} games, testing on {len(test_games)}…")
    warmup_state, _ = train(train_games)
    warmup_state.save(OUT_DIR / "elo_after_train.json")

    final_state, train_records = train(games)
    final_state.save(OUT_DIR / "elo.json")

    test_records = []
    for g in test_games:
        if final_state.games_played.get(g.home_team, 0) >= 5 and final_state.games_played.get(g.away_team, 0) >= 5:
            predicted = warmup_state.expected_home_win_prob(g.home_team, g.away_team)
            test_records.append({
                "date": g.date,
                "home": g.home_team,
                "away": g.away_team,
                "predicted_home_win": predicted,
                "home_won": g.home_won,
            })

    train_m = backtest_metrics(train_records)
    test_m = backtest_metrics(test_records)
    print("\n=== Training (walk-forward, full history) ===")
    print(json.dumps(train_m, indent=2))
    print("\n=== Holdout (last 20% of games) ===")
    print(json.dumps(test_m, indent=2))

    (OUT_DIR / "metrics.json").write_text(json.dumps({"train": train_m, "test": test_m}, indent=2))
    print(f"\nsaved Elo to {OUT_DIR / 'elo.json'}")


if __name__ == "__main__":
    asyncio.run(main())
