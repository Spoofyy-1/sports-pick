"""Parlay combination math. Assumes independent legs."""
from __future__ import annotations

from math import prod

from .ev import american_to_decimal, decimal_to_american, ev_per_dollar


def parlay_decimal(american_odds: list[int]) -> float:
    return prod(american_to_decimal(o) for o in american_odds)


def parlay_american(american_odds: list[int]) -> int:
    return decimal_to_american(parlay_decimal(american_odds))


def parlay_implied_prob(american_odds: list[int]) -> float:
    return 1 / parlay_decimal(american_odds)


def parlay_true_prob(true_probs: list[float]) -> float:
    return prod(true_probs)


def parlay_ev(true_probs: list[float], american_odds: list[int]) -> float:
    combined_prob = parlay_true_prob(true_probs)
    combined_american = parlay_american(american_odds)
    return ev_per_dollar(combined_prob, combined_american)


def summarize_parlay(legs: list[dict]) -> dict:
    """legs: [{'label': str, 'american': int, 'true_prob': float}, ...]"""
    odds = [leg["american"] for leg in legs]
    probs = [leg["true_prob"] for leg in legs]
    return {
        "legs": legs,
        "combined_american": parlay_american(odds),
        "combined_decimal": round(parlay_decimal(odds), 4),
        "book_implied_prob": round(parlay_implied_prob(odds), 4),
        "model_prob": round(parlay_true_prob(probs), 4),
        "ev_per_dollar": round(parlay_ev(probs, odds), 4),
    }
