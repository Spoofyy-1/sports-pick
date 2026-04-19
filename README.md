# Sports Pick

Free NBA moneyline edge finder. Pulls live DraftKings odds (via ESPN), predicts win probability with an Elo model trained on 2+ seasons of ESPN results, ranks positive-EV picks and parlays, and can hand off a pasted bet to Kimi (Moonshot AI) for qualitative analysis.

## Stack

- **Backend:** FastAPI (Python), SQLite-free JSON artifacts, deployable to Railway
- **Frontend:** Next.js 14 (App Router) + Tailwind, deployable to Vercel
- **Model:** Elo with home-court + margin-of-victory, 538-style
- **AI:** Kimi (Moonshot AI) via OpenAI-compatible chat API

## Run locally

### Backend

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m model.train        # pulls history + trains Elo (~30s first run)
.venv/bin/uvicorn api.main:app --port 8765 --reload
```

Optional: `export MOONSHOT_API_KEY=sk-...` to enable the AI analyzer.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

## Endpoints

| Path | Purpose |
| --- | --- |
| `GET /` | Health + model/Kimi status |
| `GET /games` | All upcoming games with odds, implied prob, fair prob, model prob, EV |
| `GET /picks?min_ev=0&limit=20` | Top moneyline picks by model EV |
| `GET /bigodds?min_american=150` | Biggest-payout underdogs |
| `GET /parlays/suggested?legs=3` | Highest +EV parlays from +EV singles |
| `GET /parlays/longshots?legs=3` | Highest-payout parlays |
| `POST /ev` | Compute EV for `{true_prob, american}` |
| `POST /parlay` | Summarize a custom parlay |
| `POST /analyze` | Send a bet to Kimi for qualitative analysis |
| `POST /model/reload` | Reload Elo artifact from disk |

## Deploy

- **Backend → Railway**: point at `backend/`, set start command `uvicorn api.main:app --host 0.0.0.0 --port $PORT`, schedule `python -m model.train` daily.
- **Frontend → Vercel**: point at `frontend/`, set `NEXT_PUBLIC_API_BASE` to your Railway URL.
