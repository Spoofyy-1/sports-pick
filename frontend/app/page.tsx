"use client";

import { useEffect, useMemo, useState } from "react";
import {
  AnalyzeResult,
  BigOddsPick,
  ParlaySummary,
  Pick,
  fmtDate,
  fmtEV,
  fmtPct,
  formatAmerican,
  getJSON,
  postJSON,
} from "../lib/api";

type Tab = "picks" | "parlays" | "bigodds" | "analyzer";

export default function Page() {
  const [tab, setTab] = useState<Tab>("picks");
  const [status, setStatus] = useState<{
    ok: boolean;
    model_trained: boolean;
    kimi_enabled: boolean;
  } | null>(null);

  useEffect(() => {
    getJSON<typeof status>("/").then(setStatus).catch(() => setStatus(null));
  }, []);

  return (
    <main className="mx-auto max-w-6xl px-5 py-8">
      <Header status={status} />
      <Tabs tab={tab} onChange={setTab} />
      <div className="mt-6">
        {tab === "picks" && <PicksTab />}
        {tab === "parlays" && <ParlaysTab />}
        {tab === "bigodds" && <BigOddsTab />}
        {tab === "analyzer" && <AnalyzerTab kimiEnabled={!!status?.kimi_enabled} />}
      </div>
      <footer className="mt-16 border-t border-border pt-6 text-xs text-zinc-500">
        Odds from DraftKings via ESPN · Elo v1 model · Gamble responsibly.
      </footer>
    </main>
  );
}

function Header({ status }: { status: { model_trained: boolean; kimi_enabled: boolean } | null }) {
  return (
    <header className="flex flex-col gap-2 border-b border-border pb-6 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-accent">
          <span className="inline-block h-2 w-2 rounded-full bg-accent" /> Live NBA Edge
        </div>
        <h1 className="mt-1 text-3xl font-bold tracking-tight">Sports Pick</h1>
        <p className="mt-1 max-w-2xl text-sm text-zinc-400">
          Model-priced NBA moneylines. Compares Elo win probability against live DraftKings odds to surface
          positive-EV bets, parlays, and longshots.
        </p>
      </div>
      <div className="flex gap-2 text-[11px]">
        <Badge ok={!!status?.ok} label="API" />
        <Badge ok={!!status?.model_trained} label="Model" />
        <Badge ok={!!status?.kimi_enabled} label="Kimi" okText="on" offText="no key" />
      </div>
    </header>
  );
}

function Badge({
  ok,
  label,
  okText = "ok",
  offText = "off",
}: {
  ok: boolean;
  label: string;
  okText?: string;
  offText?: string;
}) {
  return (
    <span
      className={`rounded-full border px-2.5 py-1 font-mono ${
        ok ? "border-accent/40 bg-accent/10 text-accent" : "border-border bg-panel text-zinc-400"
      }`}
    >
      {label}: {ok ? okText : offText}
    </span>
  );
}

function Tabs({ tab, onChange }: { tab: Tab; onChange: (t: Tab) => void }) {
  const tabs: { k: Tab; label: string }[] = [
    { k: "picks", label: "Top Picks" },
    { k: "parlays", label: "Parlays" },
    { k: "bigodds", label: "Big Odds" },
    { k: "analyzer", label: "AI Analyzer" },
  ];
  return (
    <nav className="mt-6 flex gap-1 overflow-x-auto rounded-xl border border-border bg-panel p-1">
      {tabs.map((t) => (
        <button
          key={t.k}
          onClick={() => onChange(t.k)}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
            tab === t.k ? "bg-white text-black" : "text-zinc-300 hover:bg-white/5"
          }`}
        >
          {t.label}
        </button>
      ))}
    </nav>
  );
}

function Loading() {
  return <div className="py-20 text-center text-sm text-zinc-500">Loading…</div>;
}

function Empty({ msg }: { msg: string }) {
  return (
    <div className="rounded-xl border border-border bg-panel p-10 text-center text-sm text-zinc-500">{msg}</div>
  );
}

function PicksTab() {
  const [picks, setPicks] = useState<Pick[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getJSON<{ picks: Pick[] }>("/picks?min_ev=0&limit=20")
      .then((d) => setPicks(d.picks))
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <Empty msg={`Error: ${err}`} />;
  if (!picks) return <Loading />;
  if (picks.length === 0) return <Empty msg="No positive-EV moneylines in the next few days." />;

  return (
    <section className="grid gap-3">
      {picks.map((p) => (
        <article
          key={`${p.event_id}-${p.side}`}
          className="grid grid-cols-[1fr_auto] items-center gap-4 rounded-xl border border-border bg-panel p-4 transition hover:border-accent/40"
        >
          <div>
            <div className="text-xs text-zinc-400">{fmtDate(p.start_date)}</div>
            <div className="mt-0.5 text-lg font-semibold">{p.team}</div>
            <div className="text-xs text-zinc-500">vs {p.opponent} · {p.matchup}</div>
            <div className="mt-2 flex flex-wrap gap-3 text-xs text-zinc-300">
              <span>Model: <strong className="text-white">{fmtPct(p.model_prob)}</strong></span>
              <span>Market: <strong className="text-white">{fmtPct(p.market_implied)}</strong></span>
              <span>Fair: <strong className="text-white">{fmtPct(p.fair_prob)}</strong></span>
              <span>¼-Kelly: <strong className="text-white">{fmtPct(p.kelly_quarter, 2)}</strong></span>
            </div>
          </div>
          <div className="text-right">
            <div className="font-mono text-2xl font-bold">{formatAmerican(p.american)}</div>
            <div className={`mt-1 text-sm font-semibold ${p.ev_per_dollar >= 0 ? "text-accent" : "text-danger"}`}>
              EV {fmtEV(p.ev_per_dollar)}
            </div>
          </div>
        </article>
      ))}
    </section>
  );
}

function ParlaysTab() {
  const [ev, setEv] = useState<ParlaySummary[] | null>(null);
  const [long, setLong] = useState<ParlaySummary[] | null>(null);
  const [legs, setLegs] = useState(3);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setEv(null);
    setLong(null);
    Promise.all([
      getJSON<{ parlays: ParlaySummary[] }>(`/parlays/suggested?legs=${legs}&top=6`),
      getJSON<{ parlays: ParlaySummary[] }>(`/parlays/longshots?legs=${legs}&top=4`),
    ])
      .then(([a, b]) => {
        setEv(a.parlays);
        setLong(b.parlays);
      })
      .catch((e) => setErr(String(e)));
  }, [legs]);

  if (err) return <Empty msg={`Error: ${err}`} />;

  return (
    <section className="grid gap-6">
      <div className="flex items-center justify-between">
        <div className="text-sm text-zinc-400">Building parlays from +EV singles.</div>
        <div className="flex gap-1 rounded-lg border border-border bg-panel p-1 text-xs">
          {[2, 3, 4, 5].map((n) => (
            <button
              key={n}
              onClick={() => setLegs(n)}
              className={`rounded px-3 py-1 font-mono ${
                legs === n ? "bg-white text-black" : "text-zinc-300 hover:bg-white/5"
              }`}
            >
              {n} legs
            </button>
          ))}
        </div>
      </div>

      <ParlayGroup title="Top +EV Parlays" parlays={ev} />
      <ParlayGroup title="Highest Payout (Longshots)" parlays={long} />
    </section>
  );
}

function ParlayGroup({ title, parlays }: { title: string; parlays: ParlaySummary[] | null }) {
  return (
    <div>
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-zinc-400">{title}</h2>
      {!parlays ? (
        <Loading />
      ) : parlays.length === 0 ? (
        <Empty msg="None available." />
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">
          {parlays.map((p, i) => (
            <div key={i} className="rounded-xl border border-border bg-panel p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="font-mono text-2xl font-bold">{formatAmerican(p.combined_american)}</div>
                  <div className="text-xs text-zinc-500">
                    @ {p.combined_decimal.toFixed(2)} · book {fmtPct(p.book_implied_prob)} · model{" "}
                    {fmtPct(p.model_prob)}
                  </div>
                </div>
                <div className={`text-right text-sm font-semibold ${p.ev_per_dollar >= 0 ? "text-accent" : "text-danger"}`}>
                  EV {fmtEV(p.ev_per_dollar)}
                </div>
              </div>
              <ul className="mt-3 space-y-2 text-sm">
                {p.legs.map((l, j) => (
                  <li key={j} className="flex items-center justify-between gap-3 border-t border-border pt-2">
                    <span className="truncate text-zinc-300">{l.label}</span>
                    <span className="shrink-0 font-mono text-zinc-400">
                      {formatAmerican(l.american)} · {fmtPct(l.true_prob)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function BigOddsTab() {
  const [data, setData] = useState<BigOddsPick[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getJSON<{ picks: BigOddsPick[] }>("/bigodds?min_american=150&limit=25")
      .then((d) => setData(d.picks))
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <Empty msg={`Error: ${err}`} />;
  if (!data) return <Loading />;
  if (data.length === 0) return <Empty msg="No high-payout underdogs available." />;

  return (
    <section className="grid gap-3">
      {data.map((p, i) => (
        <article
          key={i}
          className="grid grid-cols-[1fr_auto] items-center gap-4 rounded-xl border border-border bg-panel p-4"
        >
          <div>
            <div className="text-xs text-zinc-400">{fmtDate(p.start_date)}</div>
            <div className="mt-0.5 text-lg font-semibold">{p.team}</div>
            <div className="text-xs text-zinc-500">{p.matchup}</div>
            <div className="mt-2 flex flex-wrap gap-3 text-xs text-zinc-300">
              <span>Model: <strong className="text-white">{fmtPct(p.model_prob)}</strong></span>
              <span>Market: <strong className="text-white">{fmtPct(p.market_implied)}</strong></span>
              <span>Payout: <strong className="text-white">{p.decimal.toFixed(2)}x</strong></span>
            </div>
          </div>
          <div className="text-right">
            <div className="font-mono text-3xl font-bold text-warn">{formatAmerican(p.american)}</div>
            {p.ev_per_dollar != null && (
              <div className={`mt-1 text-xs ${p.ev_per_dollar >= 0 ? "text-accent" : "text-zinc-500"}`}>
                EV {fmtEV(p.ev_per_dollar)}
              </div>
            )}
          </div>
        </article>
      ))}
    </section>
  );
}

function AnalyzerTab({ kimiEnabled }: { kimiEnabled: boolean }) {
  const [bet, setBet] = useState("Lakers ML + Nuggets ML + Thunder -14.5");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const parsed = useMemo(() => {
    if (!result?.raw) return null;
    try {
      return JSON.parse(result.raw);
    } catch {
      return null;
    }
  }, [result]);

  async function submit() {
    setLoading(true);
    setErr(null);
    setResult(null);
    try {
      const r = await postJSON<AnalyzeResult>("/analyze", { bet });
      setResult(r);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="grid gap-6">
      <div className="rounded-xl border border-border bg-panel p-5">
        <label className="text-sm font-medium text-zinc-300">
          Paste a moneyline or parlay
        </label>
        <textarea
          value={bet}
          onChange={(e) => setBet(e.target.value)}
          rows={4}
          className="mt-2 w-full rounded-lg border border-border bg-surface p-3 font-mono text-sm outline-none focus:border-accent"
          placeholder="e.g. Celtics ML @ -800, Thunder ML @ -1100, Lakers +4.5"
        />
        <div className="mt-3 flex items-center justify-between">
          <div className="text-xs text-zinc-500">
            {kimiEnabled
              ? "Kimi will reason over current lines, Elo ratings, and recent results."
              : "Set MOONSHOT_API_KEY on the backend to enable AI reasoning."}
          </div>
          <button
            onClick={submit}
            disabled={loading || !bet.trim()}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black disabled:opacity-40"
          >
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
      </div>

      {err && <Empty msg={`Error: ${err}`} />}

      {result && !result.enabled && (
        <div className="rounded-xl border border-warn/40 bg-warn/5 p-5 text-sm text-warn">
          {result.message}
        </div>
      )}

      {parsed && (
        <div className="rounded-xl border border-accent/30 bg-panel p-5">
          <div className="mb-3 flex items-center gap-2">
            <span className="rounded-full bg-accent/20 px-2 py-0.5 text-[10px] font-bold uppercase text-accent">
              {parsed.verdict || "Verdict"}
            </span>
            <span className="font-mono text-sm text-zinc-300">
              Probability: {fmtPct(parsed.probability)}
            </span>
          </div>
          <div className="text-sm leading-relaxed text-zinc-200">{parsed.rationale}</div>
          {parsed.risk && (
            <div className="mt-3 rounded-lg border border-border bg-surface p-3 text-xs text-zinc-400">
              <span className="font-semibold text-danger">Biggest risk: </span>
              {parsed.risk}
            </div>
          )}
        </div>
      )}

      {result?.enabled && !parsed && (
        <pre className="overflow-auto rounded-xl border border-border bg-panel p-5 text-xs text-zinc-300">
          {result.raw}
        </pre>
      )}
    </section>
  );
}
