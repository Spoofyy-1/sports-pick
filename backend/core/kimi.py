"""Kimi (Moonshot AI) client. Uses the OpenAI-compatible chat endpoint.

Set MOONSHOT_API_KEY to enable. Without a key, analyze() returns a stub.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

API_URL = "https://api.moonshot.ai/v1/chat/completions"
MODEL = os.environ.get("MOONSHOT_MODEL", "kimi-k2-0905-preview")


def available() -> bool:
    return bool(os.environ.get("MOONSHOT_API_KEY"))


async def analyze(bet_description: str, context: dict[str, Any]) -> dict[str, Any]:
    """Ask Kimi to reason about a bet/parlay and return a structured view."""
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        return {
            "enabled": False,
            "message": "Set MOONSHOT_API_KEY on the backend to enable AI analysis.",
            "bet": bet_description,
        }

    system = (
        "You are a sharp NBA sports-betting analyst. Given a bet or parlay described by the user, "
        "and the context data provided (Elo ratings, market lines, recent results), produce: "
        "1) a calibrated probability that the bet hits (0-1), "
        "2) a 2-3 sentence rationale citing specifics, "
        "3) the single biggest risk, "
        "4) a verdict (BET / PASS / LEAN)."
        " Be direct. Do not hedge with disclaimers."
    )
    user_prompt = (
        f"Bet: {bet_description}\n\nContext JSON:\n{context}\n\n"
        "Respond as compact JSON with keys: probability, rationale, risk, verdict."
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
        )
        r.raise_for_status()
        data = r.json()

    content = data["choices"][0]["message"]["content"]
    return {"enabled": True, "model": MODEL, "raw": content}
