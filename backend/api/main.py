from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv(Path(__file__).parent.parent / ".env")
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.ev import (
    american_to_decimal,
    american_to_implied_prob,
    ev_per_dollar,
    kelly_fraction,
    remove_vig_two_way,
)
from core.kimi import analyze as kimi_analyze, available as kimi_available
from core.parlay import summarize_parlay
from data.players import fetch_team_players
from model import predict as model_predict_v1
from model import predict_v2 as model_predict
from model.props_v2 import (
    STAT_FIELDS,
    price_prop,
    suggest_lines_for_player,
    top_confidence_props,
)
from model.team_ratings import TEAM_NAME_TO_ABBR, team_ratings
from model.train_all import TEAM_ABBR
from scrapers.odds import fetch_nba_moneylines

app = FastAPI(title="Sports Pick")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_odds_cache: dict[str, Any] = {"t": 0.0, "data": []}
ODDS_TTL = 60.0


async def _load_games() -> list[dict]:
    now = time.time()
    if now - _odds_cache["t"] > ODDS_TTL:
        raw = await fetch_nba_moneylines()
        _odds_cache["data"] = [g.to_dict() for g in raw]
        _odds_cache["t"] = now
    return list(_odds_cache["data"])


def _enrich(game: dict) -> dict:
    h, a = game.get("home_ml"), game.get("away_ml")
    out = dict(game)
    if h is not None and a is not None:
        fh, fa = remove_vig_two_way(h, a)
        out["home_implied"] = round(american_to_implied_prob(h), 4)
        out["away_implied"] = round(american_to_implied_prob(a), 4)
        out["home_fair_prob"] = round(fh, 4)
        out["away_fair_prob"] = round(fa, 4)

    if model_predict.loaded():
        probs = model_predict.predict(
            game["home_team"], game["away_team"], game.get("start_date", "")
        )
    else:
        probs = model_predict_v1.win_probabilities(game["home_team"], game["away_team"])
    out["model"] = probs

    if h is not None:
        out["home_ev"] = round(ev_per_dollar(probs["home_win_prob"], h), 4)
    if a is not None:
        out["away_ev"] = round(ev_per_dollar(probs["away_win_prob"], a), 4)

    out["status"] = game.get("status", "scheduled")
    out["home_score"] = game.get("home_score")
    out["away_score"] = game.get("away_score")
    return out


@app.get("/")
async def root():
    return {
        "ok": True,
        "service": "sports-pick",
        "gbm_loaded": model_predict.loaded(),
        "elo_loaded": bool(model_predict_v1.state().ratings),
        "model_trained": model_predict.loaded() or bool(model_predict_v1.state().ratings),
        "kimi_enabled": kimi_available(),
    }


@app.get("/games")
async def games(status: Optional[str] = None):
    raw = await _load_games()
    enriched = [_enrich(g) for g in raw]
    if status:
        enriched = [g for g in enriched if g.get("status") == status]
    return {"count": len(enriched), "games": enriched}


@app.get("/graded")
async def graded(limit: int = 20):
    """Recent completed games with model prediction vs. actual outcome.

    Uses the honest pre-game prediction recorded during training (if available).
    Falls back to current state otherwise (and flags `honest=false`).
    """
    raw = await _load_games()
    out = []
    for g in raw:
        if g.get("status") != "final" or g.get("home_won") is None:
            continue
        e = _enrich(g)
        pre = model_predict_v1.pregame_win_prob(e["start_date"], e["home_team"], e["away_team"])
        honest = pre is not None
        model_home = pre if honest else e["model"]["home_win_prob"]
        picked_home = model_home > 0.5
        actual_home = bool(g["home_won"])
        out.append({
            "event_id": e["event_id"],
            "matchup": e["name"],
            "date": e["start_date"],
            "home_team": e["home_team"],
            "away_team": e["away_team"],
            "home_score": g["home_score"],
            "away_score": g["away_score"],
            "model_home_prob": round(model_home, 4),
            "honest": honest,
            "model_pick": e["home_team"] if picked_home else e["away_team"],
            "actual_winner": e["home_team"] if actual_home else e["away_team"],
            "correct": picked_home == actual_home,
            "home_ml": e.get("home_ml"),
            "away_ml": e.get("away_ml"),
        })
    out.sort(key=lambda x: x["date"], reverse=True)
    honest_games = [x for x in out if x["honest"]]
    correct_all = sum(1 for x in out if x["correct"])
    correct_honest = sum(1 for x in honest_games if x["correct"])
    return {
        "count": len(out),
        "honest_count": len(honest_games),
        "accuracy_all": round(correct_all / len(out), 4) if out else None,
        "accuracy_honest": round(correct_honest / len(honest_games), 4) if honest_games else None,
        "games": out[:limit],
    }


@app.get("/picks")
async def picks(min_ev: float = 0.0, limit: int = 20):
    """Top moneyline picks by model EV (model prob vs. market odds)."""
    raw = await _load_games()
    out: list[dict] = []
    for g in raw:
        if g.get("status") == "final":
            continue
        e = _enrich(g)
        for side in ("home", "away"):
            ml = e.get(f"{side}_ml")
            ev = e.get(f"{side}_ev")
            if ml is None or ev is None:
                continue
            if ev < min_ev:
                continue
            out.append({
                "event_id": e["event_id"],
                "matchup": e["name"],
                "start_date": e["start_date"],
                "side": side,
                "team": e[f"{side}_team"],
                "opponent": e["away_team" if side == "home" else "home_team"],
                "american": ml,
                "decimal": round(american_to_decimal(ml), 3),
                "model_prob": e["model"][f"{side}_win_prob"],
                "market_implied": e.get(f"{side}_implied"),
                "fair_prob": e.get(f"{side}_fair_prob"),
                "ev_per_dollar": ev,
                "kelly_quarter": round(kelly_fraction(e["model"][f"{side}_win_prob"], ml, 0.25), 4),
            })
    out.sort(key=lambda p: p["ev_per_dollar"], reverse=True)
    return {"count": len(out), "picks": out[:limit]}


@app.get("/bigodds")
async def big_odds(min_american: int = 200, limit: int = 20):
    """Underdogs with the biggest payouts (lottery tickets, sorted by payout)."""
    raw = await _load_games()
    out: list[dict] = []
    for g in raw:
        e = _enrich(g)
        for side in ("home", "away"):
            ml = e.get(f"{side}_ml")
            if ml is None or ml < min_american:
                continue
            out.append({
                "matchup": e["name"],
                "start_date": e["start_date"],
                "team": e[f"{side}_team"],
                "american": ml,
                "decimal": round(american_to_decimal(ml), 3),
                "model_prob": e["model"][f"{side}_win_prob"],
                "market_implied": e.get(f"{side}_implied"),
                "ev_per_dollar": e.get(f"{side}_ev"),
            })
    out.sort(key=lambda p: p["american"], reverse=True)
    return {"count": len(out), "picks": out[:limit]}


@app.get("/parlays/suggested")
async def suggested_parlays(legs: int = 3, top: int = 5):
    """Build parlays from the top +EV single picks, rank by combined EV."""
    raw = await _load_games()
    singles: list[dict] = []
    for g in raw:
        if g.get("status") == "final":
            continue
        e = _enrich(g)
        for side in ("home", "away"):
            ml = e.get(f"{side}_ml")
            ev = e.get(f"{side}_ev")
            if ml is None or ev is None or ev <= 0:
                continue
            singles.append({
                "label": f"{e[f'{side}_team']} ML ({e['name']})",
                "event_id": e["event_id"],
                "team": e[f"{side}_team"],
                "american": ml,
                "true_prob": e["model"][f"{side}_win_prob"],
                "ev": ev,
            })
    singles.sort(key=lambda s: s["ev"], reverse=True)
    pool = singles[: max(legs * 3, 8)]

    built: list[dict] = []
    for combo in itertools.combinations(pool, legs):
        if len({s["event_id"] for s in combo}) != legs:
            continue
        summary = summarize_parlay([
            {"label": s["label"], "american": s["american"], "true_prob": s["true_prob"],
             "event_id": s["event_id"], "team": s["team"]}
            for s in combo
        ])
        built.append(summary)

    built.sort(key=lambda p: p["ev_per_dollar"], reverse=True)
    return {"count": len(built), "parlays": built[:top]}


@app.get("/parlays/longshots")
async def longshot_parlays(legs: int = 3, top: int = 5, min_american: int = 150):
    """Highest-payout parlays from underdog picks — big odds, low probability."""
    raw = await _load_games()
    singles: list[dict] = []
    for g in raw:
        e = _enrich(g)
        for side in ("home", "away"):
            ml = e.get(f"{side}_ml")
            if ml is None or ml < min_american:
                continue
            singles.append({
                "label": f"{e[f'{side}_team']} ML ({e['name']})",
                "event_id": e["event_id"],
                "american": ml,
                "true_prob": e["model"][f"{side}_win_prob"],
            })
    singles.sort(key=lambda s: s["american"], reverse=True)
    pool = singles[: max(legs * 3, 8)]

    built: list[dict] = []
    for combo in itertools.combinations(pool, legs):
        if len({s["event_id"] for s in combo}) != legs:
            continue
        built.append(summarize_parlay([
            {"label": s["label"], "american": s["american"], "true_prob": s["true_prob"]}
            for s in combo
        ]))

    built.sort(key=lambda p: p["combined_american"], reverse=True)
    return {"count": len(built), "parlays": built[:top]}


class EVQuery(BaseModel):
    true_prob: float
    american: int


@app.post("/ev")
async def ev(q: EVQuery):
    return {
        "decimal_odds": round(american_to_decimal(q.american), 4),
        "implied_prob": round(american_to_implied_prob(q.american), 4),
        "ev_per_dollar": round(ev_per_dollar(q.true_prob, q.american), 4),
        "kelly_quarter": round(kelly_fraction(q.true_prob, q.american, 0.25), 4),
    }


class ParlayLegIn(BaseModel):
    label: str
    american: int
    true_prob: float
    event_id: Optional[str] = None
    team: Optional[str] = None


class ParlayQuery(BaseModel):
    legs: list[ParlayLegIn]


@app.post("/parlay")
async def parlay(q: ParlayQuery):
    if not q.legs:
        raise HTTPException(status_code=400, detail="no legs")
    return summarize_parlay([leg.model_dump() for leg in q.legs])


class AnalyzeQuery(BaseModel):
    bet: str


@app.post("/analyze")
async def analyze(q: AnalyzeQuery):
    raw = await _load_games()
    enriched = [_enrich(g) for g in raw]
    context = {
        "games": [
            {
                "matchup": g["name"],
                "start": g["start_date"],
                "home": g["home_team"],
                "away": g["away_team"],
                "home_ml": g.get("home_ml"),
                "away_ml": g.get("away_ml"),
                "model_home_prob": g["model"].get("home_win_prob"),
                "model_away_prob": g["model"].get("away_win_prob"),
                "home_ev": g.get("home_ev"),
                "away_ev": g.get("away_ev"),
            }
            for g in enriched
        ],
        "model": "Elo v1 trained on past ~2 seasons of ESPN results",
    }
    result = await kimi_analyze(q.bet, context)
    return result


@app.post("/model/reload")
async def model_reload():
    model_predict_v1.reload()
    model_predict.reload()
    return {"reloaded": True, "gbm_loaded": model_predict.loaded()}


@app.get("/model/metrics")
async def model_metrics():
    import json
    from pathlib import Path
    path = Path(__file__).parent.parent / "model" / "artifacts" / "gbm_metrics.json"
    if not path.exists():
        return {"error": "no metrics yet — run python -m model.gbm"}
    return json.loads(path.read_text())


@app.get("/props")
async def props(min_prob: float = 0.65, limit: int = 30, playoff: bool = True):
    """High-confidence player stat-line picks with opponent-defense adjustment."""
    raw = await _load_games()
    upcoming = [g for g in raw if g.get("status") != "final"]

    matchup_map: dict[str, str] = {}  # team_abbr -> opponent_abbr
    team_abbrs: set[str] = set()
    for g in upcoming:
        h_abbr = TEAM_ABBR.get(g.get("home_team", ""))
        a_abbr = TEAM_ABBR.get(g.get("away_team", ""))
        if h_abbr and a_abbr:
            matchup_map[h_abbr] = a_abbr.upper()
            matchup_map[a_abbr] = h_abbr.upper()
            team_abbrs.add(h_abbr)
            team_abbrs.add(a_abbr)

    player_matchups: list[tuple] = []
    for abbr in team_abbrs:
        try:
            profiles = await fetch_team_players(abbr, top_n=6)
            opp = matchup_map.get(abbr)
            for p in profiles:
                player_matchups.append((p, opp))
        except Exception:
            continue

    top = top_confidence_props(player_matchups, min_prob=min_prob, playoff=playoff)
    return {
        "count": len(top),
        "props": top[:limit],
        "teams": sorted(team_abbrs),
        "playoff_adjustment": playoff,
    }


class PropQuery(BaseModel):
    player_id: str
    team_abbr: str
    stat: str
    line: float


@app.post("/props/price")
async def price_one_prop(q: PropQuery):
    if q.stat not in STAT_FIELDS:
        raise HTTPException(status_code=400, detail=f"stat must be one of {STAT_FIELDS}")
    profiles = await fetch_team_players(q.team_abbr.lower(), top_n=15)
    profile = next((p for p in profiles if p.id == q.player_id), None)
    if not profile:
        raise HTTPException(status_code=404, detail="player not found")
    quote = price_prop(profile, q.stat, q.line)
    if not quote:
        raise HTTPException(status_code=404, detail="insufficient sample")
    return quote.__dict__


@app.get("/team_ratings")
async def team_ratings_endpoint():
    """Offensive / defensive ratings derived from the scoreboard cache."""
    return team_ratings()
