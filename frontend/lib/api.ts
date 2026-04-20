const RAW_API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8765";
export const API_BASE = /^https?:\/\//i.test(RAW_API_BASE)
  ? RAW_API_BASE.replace(/\/$/, "")
  : `https://${RAW_API_BASE.replace(/\/$/, "")}`;

export async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

export type Pick = {
  event_id: string;
  matchup: string;
  start_date: string;
  side: "home" | "away";
  team: string;
  opponent: string;
  american: number;
  decimal: number;
  model_prob: number;
  market_implied: number;
  fair_prob: number;
  ev_per_dollar: number;
  kelly_quarter: number;
};

export type BigOddsPick = {
  matchup: string;
  start_date: string;
  team: string;
  american: number;
  decimal: number;
  model_prob: number;
  market_implied: number;
  ev_per_dollar: number | null;
};

export type ParlayLeg = {
  label: string;
  american: number;
  true_prob: number;
};

export type ParlaySummary = {
  legs: ParlayLeg[];
  combined_american: number;
  combined_decimal: number;
  book_implied_prob: number;
  model_prob: number;
  ev_per_dollar: number;
};

export type AnalyzeResult = {
  enabled: boolean;
  message?: string;
  model?: string;
  raw?: string;
  bet?: string;
};

export type GradedGame = {
  event_id: string;
  matchup: string;
  date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  model_home_prob: number;
  honest: boolean;
  model_pick: string;
  actual_winner: string;
  correct: boolean;
  home_ml: number | null;
  away_ml: number | null;
};

export type GradedResponse = {
  count: number;
  honest_count: number;
  accuracy_all: number | null;
  accuracy_honest: number | null;
  games: GradedGame[];
};

export type PropPick = {
  player: string;
  team: string;
  stat: "points" | "rebounds" | "assists" | "threes";
  line: number;
  side: "over" | "under";
  model_prob: number;
  mean: number;
  stdev: number;
  n_games: number;
  recent: number[];
};

export type PropsResponse = {
  count: number;
  props: PropPick[];
  teams: string[];
};

export type PortfolioBetLeg = {
  label: string;
  event_id: string | null;
  team: string | null;
  american: number;
  true_prob: number | null;
};

export type PortfolioBetLegResult = {
  event_id: string;
  team: string;
  winner: string;
  leg_won: boolean;
};

export type PortfolioBet = {
  id: string;
  placed_ts: number;
  settled_ts?: number;
  stake: number;
  combined_american: number;
  combined_decimal: number;
  model_prob: number;
  ev_per_dollar_at_placement: number;
  potential_win: number;
  legs: PortfolioBetLeg[];
  status: "open" | "won" | "lost";
  payout?: number;
  leg_results?: PortfolioBetLegResult[];
};

export type PortfolioSummary = {
  starting_balance: number;
  balance: number;
  pnl: number;
  roi: number;
  open_bets: PortfolioBet[];
  closed_bets: PortfolioBet[];
  total_bets_placed: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  total_staked: number;
  total_returned: number;
  last_tick_ts: number;
  skipped?: string;
};

export function formatAmerican(n: number): string {
  return n > 0 ? `+${n}` : `${n}`;
}

export function fmtPct(p: number | null | undefined, digits = 1): string {
  if (p == null) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

export function fmtEV(ev: number | null | undefined): string {
  if (ev == null) return "—";
  const sign = ev >= 0 ? "+" : "";
  return `${sign}${(ev * 100).toFixed(1)}%`;
}

export function fmtDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}
