"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

export type LatLng = [number, number];
export type BBox = [number, number, number, number]; // south, west, north, east
export type PlannerStop = { id: string; lat: number; lng: number; name: string; demand: number };
type Stop = { id: string; lat: number; lng: number; name: string; road: string; key: boolean; routes: number[] };
type Metrics = {
  score: number;
  att: number;
  unserved: number;
  d0: number;
  d1: number;
  d2: number;
  dun: number;
  fleet: number;
  total_km: number;
  route_km: number[];
  route_min: number[];
  route_min_back: number[];
  peak_hour_load: number[];
  category: (keyof typeof CATEGORY)[];
  headway: number[]; // minutes between buses per route: the planner's headway, or less where a route is busy
  buses: number[];
  express: boolean[][]; // per route, per link between key stops: skips the stops between them
  walk: number; // minutes walked to and from stops, per trip (included in att)
  riders?: number; // score v3 (since 2026-10-29): bus trips a day, people near a served stop, the bus budget
  boardings?: number;
  covered?: number;
  budget?: number;
  periods?: (number | null)[][]; // Phase 3 (2026-10-29): per route, minutes between buses in each period (null = none then)
  fleet_k?: number[]; // buses on the road in each period
  hubs?: number[]; // key stops where changes are timed (buses meet)
  empty_km_day?: number; // empty runs from depots and back
  electric?: boolean[]; // an electric bus can run the route on its range plus charging in layovers
  changed?: number; // Phase 4 (2026-11-01): routes changed since training started (each 0-1)
};
// Phase 4: why a route exists (engine sim.explain); names of key stops
type Why = {
  role: keyof typeof ROLE;
  boardings: number; // riders boarding it a day
  direct: number; // of them, riding it the whole way
  pairs: [string, string, number][]; // the busiest of those trips, both ways
  lost_riders: number; // without it (its buses on the other routes); below 0 = its buses would carry more elsewhere
  lost_covered: number; // people whose only stop near home it has
  only: string[]; // key stops no other route serves
};
type Sumo = {
  congestion: { path: LatLng[]; c: number }[];
  buses: { route: number; dir: number; trips: number; minutes: number; delay: number }[];
  cars: number;
  teleports: number;
  scale: number;
};
export type Design = {
  id: string;
  routes: { stops: string[]; path: LatLng[] }[];
  stops: Stop[];
  coverage: number;
  metrics: Metrics;
  max_route_min: number; // 0 = no cap
  people: number;
  trips: number;
  dosm_year: string;
  planner_snap_m: number[];
  periods?: [string, number, number][]; // name, from, to (hour of the day)
  sumo?: Sumo;
  why?: Why[]; // per route; left out while training (added when paused)
};
type Run = {
  id: string;
  name: string;
  optimizer: string;
  routes: number;
  stops: number;
  coverage: number;
  score: number;
  att: number;
  unserved: number;
  d0: number;
  fleet: number;
  total_km: number;
  riders?: number | null;
  covered?: number | null;
  trained_s: number;
  training: boolean;
};
type Form = {
  name: string;
  n_routes: number;
  n_stops: number;
  headways: number[]; // shuttle, local, inter-town
  max_route_min: number;
  optimizer: string;
  seconds: number;
  max_buses: number;
  max_km: number;
};
type Train = {
  running: boolean;
  mode?: "routes" | "stops"; // what the running training changes: routes only, or stop spots too
  generation: number;
  seconds: number;
  history: [number, number][]; // [training seconds, best cost RM/trip]: generation 0, then every improvement
  error: string | null;
  design?: Omit<Design, "id">; // the best design so far, sent when newer than what we have
};
type School = {
  running: boolean;
  cities?: number;
  updates?: number;
  seconds?: number;
  last?: number; // latest check: AI cost / baseline cost on maps it never trains on
  history: [number, number][]; // [training seconds, best so far], every time the saved policy improves
  found?: [number, string][]; // [training seconds, how its designs changed], at each improvement
  live?: { n: number } & Partial<Live>; // just { n } when we already have frame n
};
// The AI's own design of the Kelantan network while it trains (engine rl.frame), redrawn every few seconds.
type Live = {
  n: number;
  seconds: number;
  now: Omit<Design, "id">;
  best: Omit<Design, "id">;
  best_at: number;
};

const MapView = dynamic(() => import("./MapView"), {
  ssr: false,
  loading: () => <div className="grid h-full place-items-center text-sm text-neutral-500">Loading map…</div>,
});

const TABS = [
  ["design", "1. Design"],
  ["result", "2. Result"],
  ["saved", "Saved"],
  ["ai", "AI training"],
] as const;
const KELANTAN: BBox = [4.5, 101.3, 6.3, 102.7]; // the whole state, as engine/server.py KELANTAN
const STATE_NETWORK = "Kelantan network"; // the one saved run the "Kelantan network" button always opens
const COLORS = ["#60a5fa", "#f87171", "#4ade80", "#c084fc", "#fb923c", "#22d3ee", "#f472b6", "#a3e635"];
const CATEGORY = { shuttle: "Shuttle", local: "Local", intertown: "Inter-town" }; // engine order: sim.CATEGORIES
const ROLE = {
  trunk: ["Trunk", "a main line: a bus every 15 min or better at its busiest"],
  feeder: ["Feeder", "brings riders to other buses: 30% or more of its riders change"],
  coverage: ["Coverage", "a bus for the people along it, even where few ride"],
} as const; // engine sim.TRUNK_MIN, sim.FEEDER_SHARE
const n0 = (x: number) => Math.round(x).toLocaleString();
function whyText(w: Why) {
  const pairs = w.pairs.map(([a, b]) => (a === b ? `along ${a}` : `${a} ↔ ${b}`));
  const only = [...new Set(w.only)];
  return (
    `${n0(w.boardings)} riders a day, ${n0(w.direct)} of them the whole way` +
    (pairs.length ? ` (most: ${pairs.join("; ")})` : "") +
    `. Without it: ${w.lost_riders >= 0 ? `${n0(w.lost_riders)} fewer riders` : `its buses would carry ${n0(-w.lost_riders)} more riders on other routes`}` +
    `, ${n0(w.lost_covered)} people lose their only stop near home` +
    (only.length ? `; no other bus at ${only.join(", ")}` : "") +
    "."
  );
}
const OPTIMIZERS: Record<string, string> = {
  baseline: "Baseline (busiest trips first)",
  ga: "Genetic algorithm",
  rl: "Deep RL (learns this map)",
  "rl+ga": "RL + GA hybrid (best)",
};
const NONE = new Set<number>();
const input = "rounded border border-neutral-700 bg-transparent px-2 py-1";
const button = "rounded border border-neutral-700 px-3 py-1.5 hover:bg-neutral-900 disabled:opacity-50";
const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
const when = (id: string) => `${id.slice(4, 8)}-${id.slice(8, 10)}-${id.slice(10, 12)} ${id.slice(13, 15)}:${id.slice(15, 17)}`;
const dur = (s: number) =>
  s < 60 ? `${Math.round(s)} s` : s < 3600 ? `${Math.floor(s / 60)} min ${Math.floor(s % 60)} s` : `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    throw new Error(
      detail
        ? typeof detail === "string"
          ? detail
          : JSON.stringify(detail)
        : `Engine returned ${res.status}. Is it running on port 8000? (see README)`,
    );
  }
  return res.json();
}

function Num({ label, value, onChange, min, max, step = 1 }: { label: string; value: number; onChange: (v: number) => void; min: number; max: number; step?: number }) {
  return (
    <label className="flex items-center justify-between gap-2">
      {label}
      <input type="number" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className={`w-24 ${input}`} />
    </label>
  );
}

// Best cost so far over training time. A step line: the best only moves when the AI finds a better design. A rise means
// the rules changed and re-scored the design; only the part since then compares like with like, so only that is drawn.
function Curve({ h: all, seconds, fmt = (v) => `score ${v.toFixed(3)}`, what = "Best score" }: { h: [number, number][]; seconds: number; fmt?: (v: number) => string; what?: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const h = all.slice(Math.max(0, all.findLastIndex(([, s], i) => i > 0 && s > all[i - 1][1])));
  const t0 = h[0][0];
  const T = Math.max(seconds, h[h.length - 1][0], t0 + 1);
  const hi = h[0][1];
  const lo = h[h.length - 1][1];
  const x = (t: number) => ((t - t0) / (T - t0)) * 300;
  const y = (s: number) => (hi > lo ? 6 + ((hi - s) / (hi - lo)) * 76 : 44);
  const points = h
    .flatMap(([t, s], i) => (i ? [[t, h[i - 1][1]], [t, s]] : [[t, s]]))
    .concat([[T, lo]])
    .map(([t, s]) => `${x(t)},${y(s)}`)
    .join(" ");
  const at = h[hover ?? h.length - 1];
  return (
    <figure className="flex flex-col gap-1">
      <svg
        viewBox="0 0 300 90"
        className="w-full touch-none"
        role="img"
        aria-label={`${what} went from ${fmt(hi)} to ${fmt(lo)} in ${dur(T - t0)} of training`}
        onPointerMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const t = t0 + ((e.clientX - r.left) / r.width) * (T - t0);
          setHover(Math.max(0, h.findLastIndex(([ht]) => ht <= t)));
        }}
        onPointerLeave={() => setHover(null)}
      >
        <line x1="0" x2="300" y1="89.5" y2="89.5" stroke="#262626" vectorEffect="non-scaling-stroke" />
        {hover !== null && <line x1={x(at[0])} x2={x(at[0])} y1="0" y2="90" stroke="#525252" vectorEffect="non-scaling-stroke" />}
        <polyline points={points} fill="none" stroke="#3987e5" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
        <circle cx={x(hover === null ? T : at[0])} cy={y(at[1])} r={4} fill="#3987e5" stroke="#0a0a0a" strokeWidth={2} />
      </svg>
      <figcaption className="flex justify-between gap-2 text-neutral-400">
        <span>{hover === null ? `${fmt(hi)} → ${fmt(lo)}` : `${fmt(at[1])} at ${dur(at[0])}`}</span>
        <span>{dur(T)} trained</span>
      </figcaption>
      <details>
        <summary className="cursor-pointer text-neutral-400">Every improvement ({h.length})</summary>
        <div className="max-h-40 overflow-y-auto">
          <table className="w-full text-left tabular-nums">
            <thead>
              <tr className="text-neutral-400">
                <th className="font-normal">Training time</th>
                <th className="font-normal">{what}</th>
              </tr>
            </thead>
            <tbody>
              {h.map(([t, s], i) => (
                <tr key={i}>
                  <td>{dur(t)}</td>
                  <td>{fmt(s)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

export default function Designer() {
  const [view, setView] = useState<BBox | null>(null);
  const [focus, setFocus] = useState<BBox | null>(null);
  const [form, setForm] = useState<Form>({ name: "", n_routes: 8, n_stops: 40, headways: [30, 30, 60], max_route_min: 90, optimizer: "rl+ga", seconds: 20, max_buses: 0, max_km: 0 });
  const [planner, setPlanner] = useState<PlannerStop[]>([]);
  const [design, setDesign] = useState<Design | null>(null);
  const [locked, setLocked] = useState<Set<number>>(new Set());
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [traffic, setTraffic] = useState(1);
  const [showTraffic, setShowTraffic] = useState(true);
  const [runs, setRuns] = useState<Run[]>([]);
  const [compare, setCompare] = useState<string[]>([]);
  const [train, setTrain] = useState<Train | null>(null);
  const [school, setSchool] = useState<School | null>(null);
  const [live, setLive] = useState<Live | null>(null);
  const [liveView, setLiveView] = useState<"now" | "best">("now");
  const [tab, setTab] = useState<(typeof TABS)[number][0]>("design");
  const [showBuses, setShowBuses] = useState(true);
  const [toast, setToast] = useState<string | null>(null);
  const [desktop, setDesktop] = useState("");
  const [shown, setShown] = useState<{ id: string; routes: Set<number> } | null>(null); // route filter; another design = all

  const set = (patch: Partial<Form>) => setForm({ ...form, ...patch });
  const refreshRuns = () => call<Run[]>("/engine/runs").then(setRuns).catch(() => {});
  useEffect(() => {
    refreshRuns();
  }, []);

  // The deep-RL designer's own training (python rl.py <minutes>), live: every 5 s while it trains, else every minute.
  // Each improvement while the page is open pops up, and on the desktop too once the planner allowed it.
  useEffect(() => {
    let alive = true;
    let timer = 0;
    let seen = -1; // improvements already shown
    let frame = 0; // the live design we have
    const tick = async () => {
      const s = await call<School>(`/engine/rl/training?live=${frame}`).catch(() => null);
      if (!alive) return;
      if (s) {
        if (seen > 0 && s.history.length > seen) {
          const what = s.found?.[s.found.length - 1]?.[1];
          const text = `AI improved: ${s.history[seen - 1][1].toFixed(3)}× → ${s.history[s.history.length - 1][1].toFixed(3)}× the baseline's cost.${what ? ` Its designs changed: ${what}.` : ""}`;
          setToast(text);
          window.setTimeout(() => setToast((t) => (t === text ? null : t)), 30000);
          if (typeof Notification !== "undefined" && Notification.permission === "granted") new Notification("LAJU AI got better", { body: text });
        }
        seen = s.history.length;
        if (s.live) frame = s.live.n;
        if (s.live?.now) setLive(s.live as Live);
        setSchool(s);
      }
      timer = window.setTimeout(tick, s?.running ? 5000 : 60000);
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  // While training runs, poll its progress; the map follows the best design so far.
  const runId = design?.id;
  const training = !!train?.running;
  useEffect(() => {
    if (!runId || !training) return;
    let since = 0;
    let alive = true;
    let timer = 0;
    const tick = async () => {
      try {
        const s = await call<Train>(`/engine/runs/${runId}/train?since=${since}`);
        if (!alive) return;
        since = s.history.length;
        if (s.design) setDesign({ id: runId, ...s.design });
        setTrain(s);
        if (s.running) timer = window.setTimeout(tick, 2000);
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      }
    };
    timer = window.setTimeout(tick, 500);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [runId, training]);

  async function run(label: string, fn: () => Promise<void>) {
    setBusy(label);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  function designNetwork() {
    const byId = new Map(design?.stops.map((s) => [s.id, s]));
    const keep = design ? [...locked].sort((a, b) => a - b).map((i) => design.routes[i]) : [];
    const lockedStops = keep.map((r) =>
      r.stops.flatMap((id) => {
        const s = byId.get(id);
        return s?.key ? [{ lat: s.lat, lng: s.lng, name: s.name }] : [];
      }),
    );
    run("Designing network…", async () => {
      const d = await call<Design>("/engine/design", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          bbox: view,
          forced: planner.map(({ lat, lng, name, demand }) => ({ lat, lng, name, demand })),
          locked: lockedStops,
        }),
      });
      setDesign(d);
      setTab("result");
      setTrain(null);
      setLocked(new Set(lockedStops.map((_, i) => i)));
      refreshRuns();
    });
  }

  // One network for the whole state: designed the first time, then always the same run, improving live until paused.
  function openStateNetwork() {
    const saved = runs.find((r) => r.name === STATE_NETWORK);
    run(saved ? "Opening the Kelantan network…" : "Designing the Kelantan network (first time only, about 2 min)…", async () => {
      const id =
        saved?.id ??
        (
          await call<Design>("/engine/design", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...form, name: STATE_NETWORK, bbox: KELANTAN, n_stops: 150, n_routes: 60, optimizer: "rl+ga", seconds: 60, forced: [], locked: [] }),
          })
        ).id;
      const r = await call<{ output: Omit<Design, "id"> }>(`/engine/runs/${id}`);
      setDesign({ id, ...r.output });
      setTab("result");
      setLocked(new Set());
      setFocus(KELANTAN);
      setTrain(await call<Train>(`/engine/runs/${id}/train`, { method: "POST" }));
      refreshRuns();
    });
  }

  function startTraining(stops = false) {
    if (!design) return;
    run(stops ? "Starting to move stops…" : "Starting training…", async () => {
      setTrain(await call<Train>(`/engine/runs/${design.id}/train${stops ? "?stops=true" : ""}`, { method: "POST" }));
      refreshRuns();
    });
  }

  function pauseTraining() {
    if (!design) return;
    run("Pausing and saving…", async () => {
      const s = await call<Train>(`/engine/runs/${design.id}/pause?since=-1`, { method: "POST" }); // always the design: why each route exists
      if (s.design) setDesign({ id: design.id, ...s.design });
      setTrain(s);
      refreshRuns();
    });
  }

  function simulate() {
    if (!design) return;
    run("Simulating traffic with SUMO…", async () => {
      const sumo = await call<Sumo>(`/engine/runs/${design.id}/simulate?scale=${traffic}`, { method: "POST" });
      setDesign({ ...design, sumo });
      setShowTraffic(true);
    });
  }

  function load(id: string) {
    run("Loading scenario…", async () => {
      const r = await call<{ input: Form & { bbox: BBox; forced: Omit<PlannerStop, "id">[] }; output: Omit<Design, "id"> }>(`/engine/runs/${id}`);
      const { bbox, forced, ...rest } = r.input;
      setDesign({ id, ...r.output });
      setTab("result");
      setTrain(await call<Train>(`/engine/runs/${id}/train`));
      setLocked(new Set());
      setForm({ ...form, ...Object.fromEntries(Object.keys(form).map((k) => [k, rest[k as keyof Form] ?? form[k as keyof Form]])) });
      setPlanner(forced.map((s) => ({ ...s, id: crypto.randomUUID() })));
      setFocus(bbox);
    });
  }

  const m = design?.metrics;
  const visible = (i: number) => !shown || shown.id !== design?.id || shown.routes.has(i);
  const toggleRoute = (i: number) => {
    if (!design) return;
    const next = new Set(design.routes.map((_, k) => k).filter(visible));
    if (next.has(i)) next.delete(i);
    else next.add(i);
    setShown({ id: design.id, routes: next });
  };
  const pair = compare.map((id) => runs.find((r) => r.id === id)).filter((r): r is Run => !!r);
  const state = runs.find((r) => r.name === STATE_NETWORK);
  const onMap = tab === "ai" && live ? { id: `ai-${liveView}`, ...live[liveView] } : null; // the AI's live design, else ours
  const quiet = train?.history.length ? train.seconds - train.history[train.history.length - 1][0] : 0; // training time since the last better design

  return (
    <main className="flex h-screen flex-col md:flex-row print:block print:h-auto">
      <aside className="flex w-full shrink-0 flex-col gap-4 overflow-y-auto border-neutral-800 p-4 text-sm md:w-[26rem] md:border-r print:w-full print:overflow-visible">
        <div>
          <h1 className="text-xl font-bold">LAJU</h1>
          <p className="text-neutral-400">AI bus network designer · Kelantan</p>
        </div>

        <nav role="tablist" aria-label="Sections" className="flex gap-1 border-b border-neutral-800 print:hidden">
          {TABS.map(([k, label]) => (
            <button
              key={k}
              role="tab"
              aria-selected={tab === k}
              onClick={() => {
                setTab(k);
                if (k === "ai" && live) {
                  setFocus([...KELANTAN]);
                  refreshRuns();
                }
              }}
              className={`rounded-t px-3 py-1.5 ${tab === k ? "bg-neutral-800 font-semibold text-white" : "text-neutral-400 hover:text-white"}`}
            >
              {label}
              {((k === "result" && training) || (k === "ai" && school?.running)) && <span className="text-emerald-400" title="live"> ●</span>}
            </button>
          ))}
        </nav>
        {busy && <p className="text-blue-300 print:hidden">{busy}</p>}
        {error && <p className="text-red-400 print:hidden">{error}</p>}

        {tab === "design" && (
          <>
            <ol className="list-inside list-decimal text-neutral-400">
              <li>Whole state: press the green button. One network, kept and improved live.</li>
              <li>One area: move the map there, then press the blue button.</li>
              <li>The result opens; press Train there to keep improving it.</li>
            </ol>

            <button className="rounded bg-emerald-700 px-3 py-2 font-semibold text-white hover:bg-emerald-800 disabled:opacity-50" disabled={!!busy || training} onClick={openStateNetwork}>
              Kelantan network: open and keep improving
            </button>

            <section className="flex flex-col gap-2 rounded border border-neutral-800 p-3">
              <Num label="Bus routes" value={form.n_routes} min={1} max={60} onChange={(v) => set({ n_routes: v })} />
              <label className="flex items-center justify-between gap-2">
                Route designer
                <select value={form.optimizer} onChange={(e) => set({ optimizer: e.target.value })} className={`w-44 ${input} bg-neutral-950`}>
                  {Object.entries(OPTIMIZERS).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
              {form.optimizer !== "baseline" && <Num label="Search time (s)" value={form.seconds} min={1} max={600} onChange={(v) => set({ seconds: v })} />}
              <label className="flex flex-col gap-1">
                <span>
                  Buses to share: <b>{form.max_buses || 64}</b> <span className="text-neutral-500">(BAS.MY Kelantan has 64)</span>
                </span>
                <input type="range" min={16} max={400} value={form.max_buses || 64} onChange={(e) => set({ max_buses: Number(e.target.value) })} />
              </label>
              <button className="rounded bg-blue-600 px-3 py-2 font-semibold text-white hover:bg-blue-700 disabled:opacity-50" disabled={!!busy || !view || training} onClick={designNetwork}>
                {training ? "Training… pause it to design again" : "Design network for this map view"}
              </button>
            </section>

            <details>
              <summary className="cursor-pointer text-neutral-400">More settings</summary>
              <div className="mt-2 flex flex-col gap-2">
                <p className="text-neutral-500">
                  The AI places key stops where people live and travel, then intermediate stops along the routes, only on roads
                  where a bus can safely stop: never on expressways, ramps, roundabouts or bridges; small roads are left to cars.
                </p>
                <label className="flex items-center justify-between gap-2">
                  Scenario name
                  <input value={form.name} onChange={(e) => set({ name: e.target.value })} placeholder="optional" className={`w-44 ${input}`} />
                </label>
                <Num label="AI key stops" value={form.n_stops} min={5} max={150} onChange={(v) => set({ n_stops: v })} />
                {Object.values(CATEGORY).map((label, c) => (
                  <Num
                    key={label}
                    label={`${label}: a bus at least every (min)`}
                    value={form.headways[c]}
                    min={2}
                    max={120}
                    onChange={(v) => set({ headways: form.headways.map((h, k) => (k === c ? v : h)) })}
                  />
                ))}
                <p className="text-neutral-500">
                  A route&apos;s length picks its bus: shuttle (minibus) under 20 km and 25 stops, local (city bus) up to 40 km and
                  65 stops, inter-town (coach) longer. Inter-town buses run express, key stops only, where another bus still stops
                  or where their riders save more time than the people near the skipped stops then walk. The AI learns which mix
                  works best. Busy routes run more often: as often as their busiest hour needs, or as their riders&apos; waiting
                  is worth an extra bus. A trip needing 3 or more changes counts as no bus: riders would not make it.
                </p>
                <Num label="Longest route, one way (min, 0 = none)" value={form.max_route_min} min={0} max={600} onChange={(v) => set({ max_route_min: v })} />
                <Num label="Budget: max route km (0 = none)" value={form.max_km} min={0} max={10000} onChange={(v) => set({ max_km: v })} />
              </div>
            </details>

            {/* Stops the AI must use, from a loaded scenario: clicking the map no longer adds one (it shows stop info). */}
            {planner.length > 0 && (
              <section className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold">Your stops ({planner.length})</h2>
                  <button
                    className={button}
                    onClick={() => {
                      setPlanner([]);
                      setDesign(null);
                      setTrain(null);
                      setLocked(new Set());
                    }}
                  >
                    Clear
                  </button>
                </div>
                {planner.map((s) => (
                  <div key={s.id} className="flex items-center gap-2">
                    <input
                      aria-label="Stop name"
                      value={s.name}
                      placeholder="Stop name"
                      onChange={(e) => setPlanner(planner.map((x) => (x.id === s.id ? { ...x, name: e.target.value } : x)))}
                      className={`min-w-0 flex-1 ${input}`}
                    />
                    <input
                      aria-label="Extra trips per day"
                      title="Extra trips/day this place generates (e.g. a new housing estate)"
                      type="number"
                      min={0}
                      value={s.demand}
                      onChange={(e) => setPlanner(planner.map((x) => (x.id === s.id ? { ...x, demand: Math.max(0, Number(e.target.value) || 0) } : x)))}
                      className={`w-24 ${input}`}
                    />
                    <button aria-label="Remove stop" className="px-1 text-neutral-500 hover:text-red-400" onClick={() => setPlanner(planner.filter((x) => x.id !== s.id))}>
                      ✕
                    </button>
                  </div>
                ))}
              </section>
            )}
          </>
        )}

        {tab === "ai" && live && (
          <section className="flex flex-col gap-2 rounded border border-neutral-800 p-3 print:hidden">
            <h2 className="font-semibold">The AI designing Kelantan{school?.running ? " · live" : " · last training"}</h2>
            <p className="text-neutral-400">
              The map shows the AI&apos;s own design of the {STATE_NETWORK} ({live.now.routes.length} routes), redrawn every few
              seconds from what it has learned so far: the lines move as it learns. This is the AI alone, before any genetic
              algorithm.
            </p>
            <div className="flex gap-1">
              {(["now", "best"] as const).map((k) => (
                <button key={k} aria-pressed={liveView === k} className={`${button} ${liveView === k ? "bg-neutral-800 text-white" : "text-neutral-400"}`} onClick={() => setLiveView(k)}>
                  {k === "now" ? `Now (${dur(live.seconds)} trained)` : `Best so far (at ${dur(live.best_at)})`}
                </button>
              ))}
            </div>
            <p>
              <b>score {live[liveView].metrics.score.toFixed(3)}</b> · no bus within 2 changes {pct(live[liveView].metrics.unserved)} · direct{" "}
              {pct(live[liveView].metrics.d0)} · {live[liveView].metrics.fleet} buses · {pct(live[liveView].coverage)} of people within 500 m
            </p>
            {state && (
              <p className="text-neutral-400">
                Your {STATE_NETWORK} (green button): score {state.score.toFixed(3)}, no bus within 2 changes {pct(state.unserved)}. Its training
                takes in the AI&apos;s latest designs every 10 minutes.
              </p>
            )}
          </section>
        )}

        {tab === "ai" && !school?.history.length && (
          <p className="text-neutral-400">
            No AI training recorded yet. To train the deep-RL designer: in <code>engine/</code>, run{" "}
            <code>.venv\Scripts\python rl.py 60</code> (minutes). Its progress shows here live.
          </p>
        )}
        {tab === "ai" && school && school.history.length > 0 && (
          <section className="flex flex-col gap-2 rounded border border-neutral-800 p-3 print:hidden">
            <h2 className="font-semibold">Deep RL training{school.running ? " · live" : ""}</h2>
            <p className="text-neutral-400">
              The deep-RL designer practises on {school.cities} random Kelantan map views ({(school.updates ?? 0).toLocaleString()} rounds of
              16 designs so far). Below: its cost on maps it never practises on, against the baseline (1.000 = as good as the
              baseline, lower is better). The best version so far designs your networks.
              {school.running && school.last !== undefined && ` Latest check: ${school.last.toFixed(3)}×.`}
            </p>
            <Curve h={school.history} seconds={school.seconds ?? 0} fmt={(v) => `${v.toFixed(3)}× baseline`} what="Best cost vs baseline" />
            {!!school.found?.length && (
              <div>
                <h3 className="font-semibold">What it found (its designs on 6 test maps)</h3>
                <ul className="list-disc pl-4 text-neutral-300">
                  {school.found
                    .slice(-5)
                    .reverse()
                    .map(([t, what]) => (
                      <li key={t}>
                        at {dur(t)}: {what}
                      </li>
                    ))}
                </ul>
              </div>
            )}
            <p className="text-neutral-400">Each improvement pops up here while this page is open.</p>
            {typeof Notification !== "undefined" && (
              <button className={button} onClick={() => Notification.requestPermission().then((p) => setDesktop(p))}>
                {desktop === "granted" ? "Desktop notifications on" : desktop === "denied" ? "Desktop notifications blocked by the browser" : "Also notify me on the desktop"}
              </button>
            )}
          </section>
        )}

        {tab === "result" && !design && <p className="text-neutral-400">No design yet: make one in 1. Design, or load one from Saved.</p>}
        {tab === "result" && design && m && (
          <section className="flex flex-col gap-2 rounded border border-neutral-800 p-3">
            <h2 className="font-semibold">{runs.find((r) => r.id === design.id)?.name || "Result"} · {when(design.id)}</h2>
            {m.riders !== undefined ? (
              <p>
                <b>{Math.round(m.riders).toLocaleString()} riders a day</b> ({Math.round(m.boardings ?? 0).toLocaleString()} boardings) ·{" "}
                <b>{Math.round(m.covered ?? 0).toLocaleString()}</b> people near a stop with a bus · {m.budget} buses to share{" "}
                <span className="text-neutral-500">(score {m.score.toFixed(3)}: lower is better)</span>
              </p>
            ) : (
              <p>
                Cost: <b>RM {m.score.toFixed(2)} per trip</b> <span className="text-neutral-500">(passenger time + bus operation + unserved trips; lower is better)</span>
              </p>
            )}
            <p>
              Average trip <b>{m.att.toFixed(1)} min</b> incl. {m.walk.toFixed(1)} min walking, waiting · direct <b>{pct(m.d0)}</b> · 1 transfer {pct(m.d1)} · 2 transfers {pct(m.d2)} · no bus within 2 changes <b>{pct(m.dun)}</b>
            </p>
            <p>
              <b>{pct(design.coverage)}</b> of {Math.round(design.people).toLocaleString()} people live within 500 m of a stop ·{" "}
              {Math.round(design.trips).toLocaleString()} {m.riders !== undefined ? "trips/day between stops, all modes" : "bus trips/day"} modelled
              (DOSM {design.dosm_year} population)
            </p>
            <p>
              <b>{m.fleet} buses</b>{" "}
              {`(${Object.entries(CATEGORY)
                .map(([k, label]) => [label, m.category.reduce((n, c, i) => n + (c === k ? m.buses[i] : 0), 0)] as const)
                .filter(([, n]) => n > 0)
                .map(([label, n]) => `${n} ${label.toLowerCase()}`)
                .join(", ")})`}{" "}
              · <b>{m.total_km.toFixed(1)} km</b> of routes · {design.stops.length} stops
            </p>
            {m.fleet_k && design.periods && (
              <p>
                On the road: {design.periods.map(([name, a, b], k) => `${m.fleet_k![k]} ${name} (${a}-${b})`).join(" · ")} · changes
                timed at <b>{m.hubs?.length ?? 0} hubs</b> · {Math.round(m.empty_km_day ?? 0).toLocaleString()} km a day empty from
                depots · {m.electric?.filter(Boolean).length ?? 0} routes an electric bus can run
              </p>
            )}
            {design.why && (
              <p>
                Routes:{" "}
                {Object.entries(ROLE)
                  .map(([k, [label, what]]) => `${design.why!.filter((w) => w.role === k).length} ${label.toLowerCase()} (${what.split(":")[0]})`)
                  .join(" · ")}
              </p>
            )}
            <div className="flex flex-col gap-2 border-y border-neutral-800 py-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-neutral-400">
                  {train?.history.length
                    ? training && train.mode === "stops"
                      ? `Moving stops onto better roads · try ${train.generation.toLocaleString()}`
                      : `${training ? "Training" : "Paused"} · generation ${train.generation.toLocaleString()}` +
                        (m.changed ? ` · ${m.changed.toFixed(1)} of ${design.routes.length} routes changed since it started` : "")
                    : "Train: the AI keeps improving this design until you pause it. It keeps a route unless changing it wins about 1% more riders."}
                </span>
                <div className="flex shrink-0 gap-1 print:hidden">
                  <button className={button} disabled={!!busy} onClick={training ? pauseTraining : () => startTraining()}>
                    {training ? "Pause (saves progress)" : train?.history.length ? "Resume training" : "Train"}
                  </button>
                  {!training && (
                    <button
                      className={button}
                      disabled={!!busy}
                      title="Also move each AI stop onto a better nearby road (main road vs side road) and re-plan the routes; slow: a few seconds per try"
                      onClick={() => startTraining(true)}
                    >
                      Move stops too
                    </button>
                  )}
                </div>
              </div>
              {train && train.history.length > 0 && (
                <>
                  <Curve h={train.history} seconds={train.seconds} />
                  <p>
                    {quiet > Math.max(120, train.seconds / 2) ? (
                      <>
                        No better design in the last <b>{dur(quiet)}</b>: about as good as this setup gets. More training rarely helps now;
                        more routes, more key stops or a smaller area will.
                      </>
                    ) : (
                      <>Last better design found {dur(quiet)} ago: still improving.</>
                    )}
                  </p>
                </>
              )}
              {train?.error && <p className="text-red-400">{train.error}</p>}
            </div>
            <div className="flex flex-wrap items-center gap-2 print:hidden">
              <span className="text-neutral-400">Show on map:</span>
              <button className={button} onClick={() => setShown(null)}>
                All routes
              </button>
              <button className={button} onClick={() => setShown({ id: design.id, routes: new Set() })}>
                None
              </button>
              <label className="flex items-center gap-1">
                <input type="checkbox" checked={showBuses} onChange={(e) => setShowBuses(e.target.checked)} />
                Moving buses
              </label>
            </div>
            {design.periods && m.periods && (
              <p className="text-neutral-500">Bus every … min: {design.periods.map(([name, a, b]) => `${name} ${a}-${b}`).join(" / ")}.</p>
            )}
            <p className="text-neutral-500 print:hidden">
              Tick a route to show it on the map. Each moving yellow dot is a bus, as often as its route runs; 1 s on screen =
              1 min. &quot;Lock&quot; keeps a route as it is in the next design (dashed on the map).
              {training && !design.why && " Pause training to see why each route exists."}
            </p>
            <ul className="flex flex-col gap-1">
              {design.routes.map((r, i) => {
                const long = !!design.max_route_min && Math.max(m.route_min[i], m.route_min_back[i]) > design.max_route_min;
                return (
                  <li key={i} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      aria-label={`Show route ${i + 1} on the map`}
                      title="Show this route on the map"
                      checked={visible(i)}
                      onChange={() => toggleRoute(i)}
                      className="h-4 w-4 shrink-0 print:hidden"
                      style={{ accentColor: COLORS[i % COLORS.length] }}
                    />
                    <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                    <span className={`flex-1 ${visible(i) ? "" : "text-neutral-500"}`}>
                      Route {i + 1}
                      {design.why?.[i] && (
                        <>
                          {" "}
                          · <b title={ROLE[design.why[i].role][1]}>{ROLE[design.why[i].role][0]}</b>
                        </>
                      )}{" "}
                      · <b>{CATEGORY[m.category[i]]}</b>
                      {m.express[i].some(Boolean) && (
                        <b title="Key stops only, where another bus still stops or where riders gain more than walkers lose">
                          {" "}
                          express{m.express[i].every(Boolean) ? "" : " in parts"}
                        </b>
                      )}
                      : {r.stops.length} stops, {m.route_km[i].toFixed(1)} km,{" "}
                      <span className={long ? "text-red-400" : undefined}>
                        {Math.round(m.route_min[i])} min{long && ` (over ${design.max_route_min})`}
                      </span>{" "}
                      ·{" "}
                      <span className="text-neutral-400">
                        bus every {m.periods ? m.periods[i].map((h) => h ?? "–").join(" / ") : m.headway[i]} min, {m.buses[i]} buses · peak{" "}
                        {m.peak_hour_load[i]} pax/h
                      </span>
                      {design.why?.[i] && <span className="block text-xs text-neutral-400">{whyText(design.why[i])}</span>}
                    </span>
                    <button
                      aria-pressed={locked.has(i)}
                      title="Lock: keep this route as it is in the next design"
                      onClick={() => {
                        const next = new Set(locked);
                        if (next.has(i)) next.delete(i);
                        else next.add(i);
                        setLocked(next);
                      }}
                      className={`shrink-0 rounded border px-1.5 text-xs print:hidden ${locked.has(i) ? "border-amber-500 text-amber-400" : "border-neutral-700 text-neutral-400 hover:text-white"}`}
                    >
                      {locked.has(i) ? "Locked" : "Lock"}
                    </button>
                  </li>
                );
              })}
            </ul>
            {design.planner_snap_m.some((d) => d > 50) && (
              <p className="text-neutral-400">
                Some of your stops were moved up to {Math.max(...design.planner_snap_m)} m to the nearest place a bus may stop.
              </p>
            )}
            <div className="flex flex-wrap gap-2 print:hidden">
              <a className={button} href={`/engine/runs/${design.id}/geojson`} download>
                GeoJSON (QGIS)
              </a>
              <a className={button} href={`/engine/runs/${design.id}/gtfs`} download>
                GTFS
              </a>
              <button className={button} onClick={() => window.print()}>
                PDF report
              </button>
            </div>

            <div className="flex items-center gap-2 border-t border-neutral-800 pt-2 print:hidden">
              <select value={traffic} onChange={(e) => setTraffic(Number(e.target.value))} className={`${input} bg-neutral-950`} aria-label="Traffic level">
                {[0.5, 1, 2, 4].map((x) => (
                  <option key={x} value={x}>
                    {x}× traffic
                  </option>
                ))}
              </select>
              <button className={button} disabled={!!busy || training} onClick={simulate}>
                Simulate peak hour (SUMO)
              </button>
            </div>
            {design.sumo && (
              <div className="flex flex-col gap-1">
                <label className="flex items-center gap-2 print:hidden">
                  <input type="checkbox" checked={showTraffic} onChange={(e) => setShowTraffic(e.target.checked)} />
                  Show congestion (green free-flow → red jammed)
                </label>
                <p className="text-neutral-400">
                  {design.sumo.cars.toLocaleString()} cars ({design.sumo.scale}× traffic) · {design.sumo.teleports} gridlock teleports
                </p>
                {design.sumo.buses.map((b) => (
                  <p key={`${b.route}-${b.dir}`}>
                    Route {b.route + 1} {b.dir ? "back" : "out"}: {b.minutes} min in traffic (planned {Math.round((b.dir ? m.route_min_back : m.route_min)[b.route])}), {b.delay} min lost
                  </p>
                ))}
              </div>
            )}
          </section>
        )}

        {tab === "saved" && (
          <section className="flex flex-col gap-2 print:hidden">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold">Saved scenarios ({runs.length})</h2>
              <button className={button} onClick={refreshRuns}>
                Refresh
              </button>
            </div>
            <p className="text-neutral-500">Tick two to compare side by side.</p>
            {runs.map((r) => (
              <div key={r.id} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  aria-label={`Compare ${r.id}`}
                  checked={compare.includes(r.id)}
                  onChange={(e) => setCompare(e.target.checked ? [...compare, r.id].slice(-2) : compare.filter((x) => x !== r.id))}
                />
                <span className="min-w-0 flex-1 truncate">
                  {r.name || when(r.id)} · {OPTIMIZERS[r.optimizer] ?? r.optimizer} · score {r.score.toFixed(3)}
                </span>
                {(r.training || r.trained_s > 0) && (
                  <span className="shrink-0 text-neutral-400">{r.training ? "training…" : `trained ${dur(r.trained_s)}`}</span>
                )}
                <button className="text-blue-400 hover:underline" onClick={() => load(r.id)}>
                  Load
                </button>
              </div>
            ))}
            {pair.length === 2 && (
              <table className="w-full text-left">
                <thead>
                  <tr className="text-neutral-400">
                    <th />
                    {pair.map((r) => (
                      <th key={r.id} className="font-normal">
                        {r.name || when(r.id)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(
                    [
                      ["Score (lower is better)", (r: Run) => r.score.toFixed(3)],
                      ["Riders a day", (r: Run) => (r.riders == null ? "-" : Math.round(r.riders).toLocaleString())],
                      ["People near a stop", (r: Run) => (r.covered == null ? "-" : Math.round(r.covered).toLocaleString())],
                      ["Avg trip (min)", (r: Run) => r.att.toFixed(1)],
                      ["Direct trips", (r: Run) => pct(r.d0)],
                      ["No bus within 2 changes", (r: Run) => pct(r.unserved)],
                      ["Coverage", (r: Run) => pct(r.coverage)],
                      ["Buses", (r: Run) => String(r.fleet)],
                      ["Route km", (r: Run) => r.total_km.toFixed(1)],
                      ["Routes / stops", (r: Run) => `${r.routes} / ${r.stops}`],
                      ["Designer", (r: Run) => OPTIMIZERS[r.optimizer] ?? r.optimizer],
                    ] as [string, (r: Run) => string][]
                  ).map(([label, f]) => (
                    <tr key={label} className="border-t border-neutral-800">
                      <td className="text-neutral-400">{label}</td>
                      {pair.map((r) => (
                        <td key={r.id}>{f(r)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        )}
      </aside>

      {toast && (
        <div role="status" className="fixed right-4 bottom-4 z-[2000] flex max-w-sm items-start gap-2 rounded border border-emerald-600 bg-neutral-900 p-3 text-sm shadow-lg print:hidden">
          <p className="flex-1">{toast}</p>
          <button aria-label="Close" className="text-neutral-400 hover:text-white" onClick={() => setToast(null)}>
            ✕
          </button>
        </div>
      )}

      <div className="min-h-[50vh] flex-1 print:h-[600px]">
        <MapView
          design={onMap ?? design}
          colors={COLORS}
          planner={planner}
          locked={onMap ? NONE : locked}
          showTraffic={showTraffic}
          showBuses={showBuses}
          visible={onMap ? () => true : visible}
          focus={focus}
          onView={setView}
        />
      </div>
    </main>
  );
}
