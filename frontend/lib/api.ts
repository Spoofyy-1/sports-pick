export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8765";

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
