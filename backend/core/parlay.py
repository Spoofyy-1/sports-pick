"""Parlay math with same-game correlation adjustment.

For legs marked with the same event_id/team, we apply a conservative
correlation boost (positive correlation within the same team's success,
negative between opposing teams in the same game).
"""
from __future__ import annotations

from math import prod, sqrt

from .ev import american_to_decimal, decimal_to_american, ev_per_dollar


def parlay_decimal(american_odds: list[int]) -> float:
    return prod(american_to_decimal(o) for o in american_odds)


def parlay_american(american_odds: list[int]) -> int:
    return decimal_to_american(parlay_decimal(american_odds))


def parlay_implied_prob(american_odds: list[int]) -> float:
    return 1 / parlay_decimal(american_odds)


def parlay_true_prob_independent(true_probs: list[float]) -> float:
    return prod(true_probs)


def _pair_correlation(a: dict, b: dict) -> float:
    """Return correlation coefficient for two legs (0 if independent)."""
    if not a.get("event_id") or not b.get("event_id"):
        return 0.0
    if a["event_id"] != b["event_id"]:
        return 0.0  # different games → independent
    # Same game: check side
    same_side = a.get("team") and a.get("team") == b.get("team")
    opp_side = a.get("team") and b.get("team") and a["team"] != b["team"]
    if same_side:
        return 0.30  # same team's success correlates across props/ML
    if opp_side:
        return -0.30
    return 0.15  # same game, no clear side — slight correlation via pace/total


def _pair_joint_prob(p1: float, p2: float, rho: float) -> float:
    """Gaussian-copula-ish joint probability for two correlated binary events."""
    cov = rho * sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    return max(0.0, min(min(p1, p2), p1 * p2 + cov))


def parlay_true_prob(legs: list[dict]) -> float:
    """Sequentially accumulate joint probability accounting for pairwise correlation.

    Simplified approach: start with first leg's prob, multiply by each subsequent
    leg's conditional probability given the average correlation to prior legs.
    Not exact but captures SGP boost/bust reasonably.
    """
    if not legs:
        return 0.0
    if len(legs) == 1:
        return legs[0]["true_prob"]
    joint = legs[0]["true_prob"]
    for i in range(1, len(legs)):
        p2 = legs[i]["true_prob"]
        rhos = [_pair_correlation(legs[j], legs[i]) for j in range(i)]
        rho_avg = sum(rhos) / len(rhos) if rhos else 0.0
        prev_marginal = joint ** (1 / i)  # rough approximation of average prior prob
        joint_i = _pair_joint_prob(prev_marginal, p2, rho_avg)
        if prev_marginal > 0:
            joint *= joint_i / prev_marginal
    return max(0.0, min(joint, 1.0))


def parlay_ev(legs: list[dict], american_odds: list[int]) -> float:
    return ev_per_dollar(parlay_true_prob(legs), parlay_american(american_odds))


def summarize_parlay(legs: list[dict]) -> dict:
    """legs: [{'label': str, 'american': int, 'true_prob': float, 'event_id'?: str, 'team'?: str}]"""
    odds = [leg["american"] for leg in legs]
    probs = [leg["true_prob"] for leg in legs]
    indep = parlay_true_prob_independent(probs)
    corr = parlay_true_prob(legs)
    return {
        "legs": legs,
        "combined_american": parlay_american(odds),
        "combined_decimal": round(parlay_decimal(odds), 4),
        "book_implied_prob": round(parlay_implied_prob(odds), 4),
        "model_prob_independent": round(indep, 4),
        "model_prob": round(corr, 4),
        "correlation_adjustment": round(corr - indep, 4),
        "ev_per_dollar": round(ev_per_dollar(corr, parlay_american(odds)), 4),
    }
