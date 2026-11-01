"""Phase 5: deep-RL route designer. A graph neural network builds a route set one decision at a time: the stop a route
starts at, the stop it heads for next (the bus takes the quickest road path there), or that the route ends. The reward
is the sim.py score of the finished set: the same score the GA and the planner see.

Training: REINFORCE with a shared baseline (Kool et al. 2019; Kwon et al. 2020, POMO): k route sets are sampled for one
city and each becomes more or less likely by how much better or worse it scored than the k's mean. Cities: random
planner map views of Kelantan, from one town to the whole state (kelantan.build, RM score), and random synthetic cities
(minutes score, like Mandl). No per-graph-size weights: a deep encoder reads the city once (road links, who travels
where), a one-layer decoder reads the design so far at every decision. One policy runs on 15 stops or 150.

Design time (active search, Bello et al. 2017; Hottung et al. 2022): the same update keeps running on the planner's own
map, so the policy learns that city while it searches. The answer is the best route set it has seen.

    python rl.py             -> self-check: "ok ..."
    python rl.py <minutes> [workers]   -> trains with 4 (or `workers`) processes, on from models/route_gnn.pt if there is
                                one (the first run builds the Kelantan training cities, ~10 min); replaces
                                models/route_gnn.pt only when better. 2 workers need about half the memory.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from torch import nn

import routes
from sim import UTURN_MIN, Problem, evaluate

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "route_gnn.pt"
CITIES = HERE / "cache" / "rl_cities.pkl"  # Kelantan training cities, rebuilt when deleted
FS, FD = 9, 12        # node features: per city (Graph.x) and per decision (Rollouts.features)
SYNTHETIC = 0.2       # share of training cities that are synthetic (minutes score, keeps Mandl in reach)
ENTROPY = 0.01        # exploration bonus per decision
CHUNK = 20_000        # nodes per batch when re-running the decisions with gradients (bounds memory)
CACHED = 30           # training cities each worker keeps prepared (Graph); more is faster, but ~1-7 MB each
LIVE_S = 10           # seconds between the live designs of the Kelantan network the website shows while the AI trains
STATE = "Kelantan network"  # the saved run the website's green button opens (web/src/app/Designer.tsx STATE_NETWORK)


class Graph:
    """One city, shared by every rollout on it: the quickest stop-to-stop paths (on[e, j] = stops a route gains heading
    from e to j), U-turns, the two propagation operators (road links, trips) and the per-city node features."""

    def __init__(self, p, n_routes, locked=()):
        n = self.n = len(p.od)
        D, self.pred = routes.quickest(p)
        self.p, self.R, self.locked = p, n_routes, [list(r) for r in locked]
        U = np.zeros((n, n, n), bool)
        for a, b, c in p.uturn:
            U[a, b, c] = True
        on, hops, first, turns = np.zeros((n, n, n), bool), np.zeros((n, n), int), np.full((n, n), -1), np.zeros((n, n))
        E, J = np.nonzero(np.isfinite(D) & ~np.eye(n, dtype=bool))
        cur, nxt, live = J.copy(), np.full(len(J), -1), np.ones(len(J), bool)
        while live.any():  # walk every quickest path back at once
            i = np.flatnonzero(live)
            e, c, x = E[i], cur[i], nxt[i]
            on[e, J[i], c] = True
            hops[e, J[i]] += 1
            prv = self.pred[e, c]
            t = x >= 0
            turns[e[t], J[i][t]] += U[prv[t], c[t], x[t]]
            end = prv == e
            first[e[end], J[i][end]] = c[end]
            nxt[i], cur[i] = c, prv
            live[i[end]] = False
        dem = p.od + p.od.T
        adj = np.isfinite(p.time)
        np.fill_diagonal(adj, False)
        A = adj + np.eye(n)
        tot = dem.sum(1) + 1e-9
        self.scale = float(p.max_route_min or D[np.isfinite(D)].max())  # minutes that make a long route
        self.per_route = float(dem.sum() / max(n_routes, 1)) + 1e-9
        xy = (p.xy - p.xy.min(0)) / (np.ptp(p.xy, 0) + 1e-9)
        walk = p.walk / 30 if p.walk is not None else np.zeros(n)  # kelantan.WALK_CAP_MIN
        city = [bool(p.headways), p.max_len / 15, (p.max_route_min or 0) / 120, n_routes / 30]
        f = lambda a: torch.from_numpy(np.asarray(a, np.float32))
        self.x = f(np.column_stack([xy, tot / tot.max(), adj.sum(1) / adj.sum(1).max(), walk, np.broadcast_to(city, (n, 4))]))
        self.A, self.T = f(A / A.sum(1, keepdims=True)), f(dem / tot[:, None])
        self.dem, self.tot = f(dem), f(tot)
        self.on, self.U, self.first, self.hops = torch.from_numpy(on), torch.from_numpy(U), torch.from_numpy(first), torch.from_numpy(hops)
        self.minutes = f(np.where(np.isfinite(D), D + UTURN_MIN * turns, np.inf))
        fits = (self.hops > 0) & (self.hops + 1 <= p.max_len)
        if p.max_route_min:
            fits &= self.minutes <= p.max_route_min
        self.can_start = fits.any(1)


class Rollouts:
    """k route sets under construction on one city, stepped together, after G's locked routes."""

    def __init__(self, G, k):
        n, given = G.n, G.locked
        self.G, self.k = G, k
        self.direct = torch.zeros(k, n, n, dtype=torch.bool)  # stop pairs one route already connects
        self.covered, self.inroute = torch.zeros(k, n, dtype=torch.bool), torch.zeros(k, n, dtype=torch.bool)
        self.end, self.prev = torch.full((k,), -1), torch.full((k,), -1)
        self.stops, self.mins = torch.zeros(k), torch.zeros(k)
        self.done = torch.full((k,), len(given))
        self.sets, self.cur = [[list(r) for r in given] for _ in range(k)], [[] for _ in range(k)]
        for r in given:
            ix = torch.tensor(r)
            self.direct[:, ix[:, None], ix] = True
            self.covered[:, ix] = True

    @property
    def alive(self):
        return self.done < self.G.R

    def mask(self):
        """(k, n + 1): stops to head for (or start at), and 'end the route'. A finished rollout may only 'end'."""
        G, n, p = self.G, self.G.n, self.G.p
        m = torch.zeros(self.k, n + 1, dtype=torch.bool)
        new = self.end < 0
        m[new, :n] = G.can_start
        r = ~new
        if r.any():
            e, pv = self.end[r], self.prev[r]
            ok = (G.hops[e] > 0) & ~(G.on[e] & self.inroute[r][:, None, :]).any(-1)  # never back through its own stops
            ok &= self.stops[r][:, None] + G.hops[e] <= p.max_len
            if p.max_route_min:
                ok &= self.mins[r][:, None] + G.minutes[e] <= p.max_route_min
            f = G.first[e]
            ok &= ~(G.U[pv.clamp(min=0)[:, None], e[:, None], f.clamp(min=0)] & (pv >= 0)[:, None])  # no U-turn at e
            m[r, :n] = ok
            m[r, n] = (self.stops[r] >= p.min_len) | ~ok.any(1)
        m[~self.alive] = False
        m[~self.alive, n] = True
        return m

    def features(self, m):
        G, n = self.G, self.G.n
        unmet = (G.dem * ~self.direct).sum(-1) / G.tot  # share of a stop's trips with no direct route yet
        gain, rel, mins, hops = (torch.zeros(self.k, n) for _ in range(4))
        r = self.end >= 0
        if r.any():
            e = self.end[r]
            on = G.on[e].float()  # (kr, n, n): stops each leg would add
            reach = (on + self.inroute[r][:, None, :].float()).clamp(max=1)
            g = ((on @ (G.dem * ~self.direct[r])) * reach).sum(-1) / G.per_route  # trips the leg would make direct
            gain[r] = g
            rel[r] = g / (g * m[r, :n]).amax(1, keepdim=True).clamp(min=1e-9)
            mins[r] = torch.where(torch.isfinite(G.minutes[e]), G.minutes[e], 0) / G.scale
            hops[r] = G.hops[e] / G.p.max_len
        isend = torch.zeros(self.k, n)
        isend[r, self.end[r]] = 1
        glob = torch.stack([self.stops / G.p.max_len, self.mins / G.scale, self.done / G.R], -1)
        return torch.cat([torch.stack([self.covered.float(), self.inroute.float(), isend, m[:, :n].float(), unmet, gain, rel,
                                       mins, hops], -1), glob[:, None].expand(-1, n, -1)], -1)

    def act(self, a):
        G, n = self.G, self.G.n
        for i in np.flatnonzero(self.alive.numpy()):
            s, e = int(a[i]), int(self.end[i])
            if e < 0:  # a new route starts here
                self.cur[i] = [s]
                self.inroute[i, s], self.end[i], self.stops[i], self.mins[i] = True, s, 1, 0
            elif s == n:  # the route ends
                self.sets[i].append(self.cur[i])
                self.covered[i, self.cur[i]], self.inroute[i] = True, False
                self.end[i], self.prev[i], self.cur[i] = -1, -1, []
                self.done[i] += 1
            else:  # head for s along the quickest path
                leg = routes.path(G.pred, e, s)[1:]
                self.cur[i] += leg
                ix = torch.tensor(self.cur[i])
                self.inroute[i, ix], self.direct[i, ix[:, None], ix] = True, True
                self.end[i], self.prev[i] = s, self.cur[i][-2]
                self.stops[i] += len(leg)
                self.mins[i] += G.minutes[e, s]


class Layer(nn.Module):
    """Message passing along road links (A) and along trips (T: who travels where), residual + layer norm."""

    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, A, T):
        return self.norm(h + self.mlp(torch.cat([h, A @ h, T @ h], -1)))


class Net(nn.Module):
    def __init__(self, d=128, depth=6):
        super().__init__()
        self.inp, self.dyn = nn.Linear(FS, d), nn.Linear(FD, d)
        self.enc = nn.ModuleList(Layer(d) for _ in range(depth))
        self.dec = Layer(d)
        self.node, self.ctx, self.out = nn.Linear(d, d), nn.Linear(2 * d, d, bias=False), nn.Linear(d, 1)  # MLP([z, ctx])
        self.stop = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def encode(self, G):
        h = self.inp(G.x)
        for layer in self.enc:
            h = layer(h, G.A, G.T)
        return h

    def forward(self, H, G, x, end):
        """H (n, d) city encoding, x (b, n, FD) decision features, end (b,) current route end (-1: none) -> (b, n + 1)."""
        z = self.dec(H + self.dyn(x), G.A, G.T)
        ze = z[torch.arange(len(end)), end.clamp(min=0)] * (end >= 0)[:, None]
        ctx = torch.cat([ze, z.mean(1)], -1)  # the route's end and the whole design
        node = self.out(torch.relu(self.node(z) + self.ctx(ctx)[:, None])).squeeze(-1)
        return torch.cat([node, self.stop(ctx)], -1)


@torch.no_grad()
def rollout(net, G, k, greedy=False, gen=None, H=None):
    """k route sets (locked routes first) and the decisions that built them, to re-run with gradients."""
    H = net.encode(G) if H is None else H
    s, trace = Rollouts(G, k), []
    while s.alive.any():
        m = s.mask()
        x = s.features(m)
        logits = net(H, G, x, s.end).masked_fill(~m, -1e9)
        a = logits.argmax(-1) if greedy else torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(1)
        trace.append((x, s.end.clone(), m, a, s.alive.clone()))
        s.act(a)
    return s.sets, trace


def learn(net, opt, G, k=16, gen=None):
    """One REINFORCE step on one city. Returns the k sampled route sets and their scores."""
    sets, trace = rollout(net, G, k, gen=gen)
    score = np.array([evaluate(G.p, rs)["score"] for rs in sets])
    c = np.log(score)
    adv = torch.from_numpy((c.mean() - c) / (c.std() + 1e-9)).float()
    opt.zero_grad()
    H = net.encode(G)
    step = max(1, CHUNK // (k * G.n))
    for i in range(0, len(trace), step):
        x, end, m, a, live = (torch.cat(t) for t in zip(*trace[i: i + step]))
        logp = torch.log_softmax(net(H, G, x, end).masked_fill(~m, -1e9), -1)
        pick = logp.gather(1, a[:, None]).squeeze(1)
        ent = -(logp.exp() * logp).sum(-1)
        w = live.float()
        loss = -((adv.repeat(len(a) // k) * pick + ENTROPY * ent) * w).sum() / k
        loss.backward(retain_graph=i + step < len(trace))
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return sets, score


def _load():
    if not MODEL.exists():
        raise FileNotFoundError(f"no trained policy at {MODEL}; run: python rl.py")
    net = Net()
    net.load_state_dict(torch.load(MODEL))
    return net.eval()


@torch.no_grad()
def _samples(net, G, k, seed=0):
    """k route sets on G, locked routes included: the first greedy, the rest sampled."""
    H = net.encode(G)
    out = rollout(net, G, 1, greedy=True, H=H)[0]
    if k > 1:
        out += rollout(net, G, k - 1, gen=torch.Generator().manual_seed(seed), H=H)[0]
    return out


def samples(p, n_routes, k=16, locked=(), seed=0):
    """k route sets from the trained policy (the first one greedy), e.g. as GA seeds. Locked routes are left out."""
    if n_routes <= len(locked):
        return [[] for _ in range(k)]
    return [rs[len(locked):] for rs in _samples(_load(), Graph(p, n_routes, locked), k, seed)]


def design(p, n_routes, locked=(), seconds=0.0, k=32, lr=1e-3, seed=0):
    """Best route set by score: of k policy samples, or (seconds > 0) of everything active search samples while it
    keeps training the policy on this city for that long."""
    if n_routes <= len(locked):
        return [list(r) for r in locked]
    rs = [list(locked) + s for s in samples(p, n_routes, k, locked, seed)]
    best = min(((evaluate(p, x)["score"], x) for x in rs), key=lambda t: t[0])
    if seconds > 0:
        torch.manual_seed(seed)
        net, G, gen = _load(), Graph(p, n_routes, locked), torch.Generator().manual_seed(seed)
        opt, end = torch.optim.Adam(net.parameters(), lr), time.time() + seconds
        while time.time() < end:
            sets, score = learn(net, opt, G, 16, gen)
            if score.min() < best[0]:
                best = (float(score.min()), sets[int(score.argmin())])
    return best[1]


def synthetic(rng, n_min=10, n_max=60):
    """Random city: Delaunay street graph thinned to Mandl-like density, gravity demand between lognormal populations."""
    n = int(rng.integers(n_min, n_max + 1))
    xy = rng.random((n, 2))
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    tri = {tuple(sorted((int(s[a]), int(s[b])))) for s in Delaunay(xy).simplices for a, b in ((0, 1), (1, 2), (0, 2))}
    med = np.median([d[e] for e in tri])
    mst = {tuple(sorted((int(a), int(b)))) for a, b in zip(*minimum_spanning_tree(d).nonzero())}  # keeps the city connected
    keep = {e for e in tri if d[e] <= 1.3 * med} | mst
    t = np.full((n, n), np.inf)
    for i, j in keep:
        t[i, j] = t[j, i] = round(1 + 30 * d[i, j])
    pop = rng.lognormal(0, 1, n) * (rng.random(n) > 0.1)
    D = routes.quickest(Problem(time=t, od=np.zeros((n, n)), xy=xy))[0]
    od = np.outer(pop, pop) * np.exp(-D / D[np.isfinite(D)].mean())
    np.fill_diagonal(od, 0)
    return Problem(time=t, od=np.round(od * 15570 / od.sum()), xy=xy, min_len=2, max_len=int(rng.integers(6, 16)))


def kelantan_cities(count, seed):
    """Random planner map views of Kelantan, a town to the whole state, with the planner settings varied:
    [(Problem, number of routes)]. Up to 60 routes, 2.5 stops per route at most: the Kelantan network's setting."""
    import kelantan
    from server import KELANTAN

    rng = np.random.default_rng(seed)
    lat, lng, pop, _ = kelantan.population()
    w = np.sqrt(pop) / np.sqrt(pop).sum()  # where people live, rural areas included
    out = []
    while len(out) < count:
        i, half = rng.choice(len(pop), p=w), float(np.exp(rng.uniform(np.log(0.03), np.log(0.6))))
        box = (max(lat[i] - half, KELANTAN[0]), max(lng[i] - half, KELANTAN[1]), min(lat[i] + half, KELANTAN[2]),
               min(lng[i] + half, KELANTAN[3]))
        heads = tuple(np.clip(np.round(np.array(kelantan.HEADWAYS) * rng.uniform(0.5, 2)), 2, 120)) if rng.random() < 0.3 \
            else kelantan.HEADWAYS
        try:
            p, _ = kelantan.build(box, int(np.exp(rng.uniform(np.log(15), np.log(150)))), headways=heads,
                                  max_route_min=float(rng.choice([90, 90, 90, 0, 60, 120])))
        except ValueError:
            continue
        if len(p.od) >= 8 and p.od.sum() > 0:
            out.append((p, int(np.clip(round(len(p.od) / rng.uniform(2.5, 8)), 2, 60))))
            print(f"city {len(out)}/{count}: {len(p.od)} stops, {out[-1][1]} routes", flush=True)
    return out


def changes(a, b):
    """What changed between two looks at a policy's designs (rl.train held_out), in words, for the planner."""
    out = []
    for k, what, f in (("d0", "direct trips", 100), ("unserved", "trips with no bus", 100), ("att", "average trip", 1),
                       ("walk", "walking per trip", 1), ("fleet", "buses", 1), ("total_km", "route km", 1)):
        x, y, unit = a[k] * f, b[k] * f, "%" if f == 100 else " min" if k in ("att", "walk") else ""
        if abs(y - x) >= max(0.01 * abs(x), 0.1):  # skip changes under 1%
            out.append(f"{what} {x:,.1f}{unit} -> {y:,.1f}{unit}")
    mix = [f"{c} {a[c]} -> {b[c]}" for c in ("shuttle", "local", "intertown") if a[c] != b[c]]
    return "; ".join(out + (["routes: " + ", ".join(mix)] if mix else [])) or "same kind of designs, slightly cheaper"


def showcase():
    """The planner's saved Kelantan network (else the whole state, 150 key stops, 60 routes): the map on which the website
    shows the AI designing live while it trains. -> (Graph, drawing data, request)"""
    import server

    d = next((x for x in (json.loads(f.read_text()) for f in sorted(server.RESULTS.glob("run-*.json"), reverse=True))
              if x["input"]["name"] == STATE), None)  # newest first, like the green button
    req = server.DesignRequest(**d["input"]) if d else server.DesignRequest(bbox=server.KELANTAN, n_stops=150, n_routes=60)
    p, net, k, lk = server._problem(req)
    return Graph(p, k, lk), net, req


def frame(net, show):
    """The policy's design of the showcase map right now, as the website draws it: the best of 8 (the first greedy, the
    rest with a fixed seed), so the lines only move when the policy has learned something."""
    import server

    G, info, req = show
    out = server._output(G.p, info, min(_samples(net, G, 8), key=lambda rs: evaluate(G.p, rs)["score"]), req, why=False)
    del out["model"]  # the trip matrix, for SUMO: not drawn
    return out


def _write(f, d):
    """All or nothing, so the engine never reads half a file. Windows refuses while the engine reads it: next time."""
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    try:
        tmp.replace(f)
    except PermissionError:
        pass


def _worker(rank, net, stop, lr, k, seed, threads, steps):
    torch.set_num_threads(threads)
    cities = pickle.loads(CITIES.read_bytes())
    rng, gen = np.random.default_rng(seed + rank), torch.Generator().manual_seed(seed + rank)
    opt, graphs, seen = torch.optim.Adam(net.parameters(), lr), {}, []
    while not stop.is_set():
        if rng.random() < SYNTHETIC:
            G = Graph(synthetic(rng), int(rng.integers(3, 9)))
        else:
            i = int(rng.integers(len(cities)))
            if i not in graphs and len(graphs) >= CACHED:
                graphs.pop(next(iter(graphs)))  # oldest out: a 16 GB laptop ran out of memory with all 300 kept
            G = graphs[i] if i in graphs else graphs.setdefault(i, Graph(*cities[i]))
        if not hasattr(G, "base"):
            G.base = evaluate(G.p, routes.baseline(G.p, G.R))["score"]
        seen.append(learn(net, opt, G, k, gen)[1].mean() / G.base)
        with steps.get_lock():
            steps.value += 1
        if rank == 0 and len(seen) % 100 == 0:
            print(f"        worker 0: sampled/baseline over its last 100 cities {np.mean(seen[-100:]):.3f}", flush=True)


def train(minutes=60.0, workers=4, lr=2e-4, k=16, seed=0, out=MODEL, start=None):
    """Train for `minutes` the computer is awake (from scratch, or from the weights file `start`): `workers` processes update one shared
    network (Hogwild, Recht et al. 2011). Saves the network to `out` whenever it beats the best so far (the file already
    there included) on held-out cities, best of 8 samples. Progress for the website goes to out.with_suffix('.json'), and
    every LIVE_S seconds the policy's design of the Kelantan network to server.RL_LIVE (the website's AI training map)."""
    import torch.multiprocessing as mp

    import server

    if not CITIES.exists():
        CITIES.write_bytes(pickle.dumps(kelantan_cities(300, seed=1)))
    torch.set_num_threads(2)
    vr = np.random.default_rng(999)
    val = [Graph(p, r) for p, r in kelantan_cities(6, seed=999)] + [Graph(synthetic(vr), int(vr.integers(3, 9))) for _ in range(3)]
    base = [evaluate(G.p, routes.baseline(G.p, G.R))["score"] for G in val]
    show, shown = showcase(), {"n": 0}

    def held_out(net):
        """(mean cost vs the baseline, what its designs look like on the Kelantan maps: trip-weighted shares, totals)."""
        best_m = [min((evaluate(G.p, rs) for rs in _samples(net, G, 8)), key=lambda m: m["score"]) for G in val]
        kel = [(G.p.od.sum(), m) for G, m in zip(val, best_m) if G.p.headways]
        w = np.array([t for t, _ in kel]) / sum(t for t, _ in kel)
        look = {k: float(np.dot(w, [m[k] for _, m in kel])) for k in ("d0", "unserved", "att", "walk")}
        look |= {k: float(sum(m[k] for _, m in kel)) for k in ("fleet", "total_km")}
        look |= {c: sum(m["category"].count(c) for _, m in kel) for c in ("shuttle", "local", "intertown")}
        return float(np.mean([m["score"] / b for m, b in zip(best_m, base)])), look

    best, seen = np.inf, None  # seen: what the best policy's designs look like
    if out.exists():  # never replace a better policy
        try:
            old = Net()
            old.load_state_dict(torch.load(out))
            best, seen = held_out(old)
        except RuntimeError:  # an older network design
            pass
    torch.manual_seed(seed)
    net = Net()
    if start:
        net.load_state_dict(torch.load(start))
    net.share_memory()
    steps, stop = mp.Value("i", 0), mp.Event()
    procs = [mp.Process(target=_worker, args=(r, net, stop, lr, k, seed, max(1, 8 // workers), steps)) for r in range(workers)]
    for x in procs:
        x.start()
    log, live, s, last = {"cities": len(pickle.loads(CITIES.read_bytes())), "history": [], "found": []}, True, 0.0, time.time()
    checked = -np.inf  # awake seconds at the last held-out check
    while live:
        live = any(x.is_alive() for x in procs)
        s, last = s + min(time.time() - last, 300), time.time()  # minutes awake: a longer gap is the laptop asleep
        if s >= 60 * minutes:
            stop.set()
        snap = Net()
        snap.load_state_dict(net.state_dict())  # the workers keep changing net: score, show and save one frozen copy
        now = frame(snap, show)
        shown.update(n=shown["n"] + 1, seconds=round(s), now=now)
        if "best" not in shown or now["metrics"]["score"] < shown["best"]["metrics"]["score"]:
            shown.update(best=now, best_at=round(s))
        _write(server.RL_LIVE, shown)
        if s - checked >= 60 or not live:
            checked = s
            v, look = held_out(snap)
            print(f"{s / 60:5.1f} min  {steps.value} updates  best of 8/baseline on held-out cities {v:.3f}  "
                  f"Kelantan network RM {now['metrics']['score']:.2f}", flush=True)
            if v < best or not log["history"]:
                log["history"].append([round(s), round(min(v, best), 4)])  # [seconds, best so far], like the GA's history
            if v < best:
                if seen:
                    log["found"].append([round(s), changes(seen, look)])
                    print("        better because its designs changed:", log["found"][-1][1], flush=True)
                best, seen = v, look
                out.parent.mkdir(exist_ok=True)
                torch.save(snap.state_dict(), out.with_suffix(".part"))
                for attempt in range(10):  # one step, so the engine never loads half a file; Windows refuses while it reads
                    try:
                        out.with_suffix(".part").replace(out)
                        break
                    except PermissionError:
                        time.sleep(0.3)
            log.update(running=live, updates=steps.value, seconds=round(s), last=round(v, 4), updated=time.time())
            _write(out.with_suffix(".json"), log)
        time.sleep(LIVE_S if live else 0)
    return best


if __name__ == "__main__":
    if len(sys.argv) > 1:  # keep improving the saved policy (lr 1e-4 worked best for that), or start one
        train(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4, lr=1e-4 if MODEL.exists() else 2e-4,
              start=MODEL if MODEL.exists() else None)
        sys.exit()
    import bench  # self-check on Mandl with an untrained network: paths, rules, one learning step, locked routes

    torch.manual_seed(0)
    p = bench.problem()
    p.max_route_min = 25
    G, net = Graph(p, 6), Net()
    for e, j in zip(*np.nonzero(np.isfinite(routes.quickest(p)[0]) & ~np.eye(G.n, dtype=bool))):
        pth = routes.path(G.pred, e, j)
        assert G.hops[e, j] == len(pth) - 1 and set(np.flatnonzero(G.on[e, j].numpy())) == set(pth[1:]) and G.first[e, j] == pth[1]
    sets, score = learn(net, torch.optim.Adam(net.parameters()), G, 8)
    for rs in sets:
        m = evaluate(p, rs)
        assert len(rs) == 6 and routes.valid(p, rs) and max(m["route_min"]) <= 25 and m["score"] in score, rs
    lock = [[0, 1, 2]]
    G = Graph(p, 3, lock)
    rs = rollout(net, G, 4, gen=torch.Generator().manual_seed(0))[0]
    assert all(r[0] == lock[0] and len(r) == 3 and routes.valid(p, r) for r in rs), rs
    assert all(np.isfinite(routes.quickest(synthetic(np.random.default_rng(s)))[0]).all() for s in range(20))  # connected
    a = {"d0": 0.2, "unserved": 0.05, "att": 60.0, "walk": 15.0, "fleet": 100.0, "total_km": 500.0, "shuttle": 2, "local": 3, "intertown": 1}
    msg = changes(a, a | {"d0": 0.25, "att": 60.3, "local": 4})  # 0.3 min is under 1%: not mentioned
    assert msg == "direct trips 20.0% -> 25.0%; routes: local 3 -> 4", msg
    print("ok", "untrained scores", np.round(np.sort(score), 2).tolist())
