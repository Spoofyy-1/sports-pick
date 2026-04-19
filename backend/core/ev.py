"""Odds conversion and expected value math."""
from __future__ import annotations


def american_to_decimal(american: int) -> float:
    if american > 0:
        return 1 + american / 100
    return 1 + 100 / abs(american)


def american_to_implied_prob(american: int) -> float:
    if american > 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def decimal_to_american(decimal: float) -> int:
    if decimal >= 2.0:
        return round((decimal - 1) * 100)
    return round(-100 / (decimal - 1))


def remove_vig_two_way(odds_a: int, odds_b: int) -> tuple[float, float]:
    pa = american_to_implied_prob(odds_a)
    pb = american_to_implied_prob(odds_b)
    total = pa + pb
    return pa / total, pb / total


def ev_per_dollar(true_prob: float, american: int) -> float:
    """Expected profit per $1 staked."""
    dec = american_to_decimal(american)
    return true_prob * (dec - 1) - (1 - true_prob)


def kelly_fraction(true_prob: float, american: int, fraction: float = 0.25) -> float:
    """Fractional Kelly stake as fraction of bankroll. Returns 0 if -EV."""
    dec = american_to_decimal(american)
    b = dec - 1
    q = 1 - true_prob
    f_star = (b * true_prob - q) / b if b > 0 else 0.0
    return max(0.0, f_star * fraction)
