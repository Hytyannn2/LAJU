"""Brain v2, Phase 0: the exam every network must pass (PLAN.txt section 9). A planner's and a rider's view of a route
set, judged at the stops a bus really serves, not by the optimisers' score:

  access    share of people who can get there door to door, on foot or by bus with at most 2 changes (walk + wait +
            ride + the waits at changes, real minutes): their nearest school in 30 min, a clinic or hospital in 45, a
            mall or market in 45, a rail station in 45, a park, stadium or beach in 60, their nearest town in 45. Per
            district too (fairness)
  coverage  people within 400 m (town) / 800 m (rural) of a stop a bus serves
  trips     the planner's own test trips (SK Kota -> KB Mall, ...)
  shape     routes over 1.5x their quickest drive; route ends where a bus can wait and the driver rest (kelantan.rests)
  fleet     buses on the road per period of the day, bus-km a day, empty runs from depots, routes an e-bus can run

Each network runs at its own headways and speeds, per period of the day (kelantan.PERIODS): BAS.MY at its official
timetable (data.gov.my GTFS: its own stops, each direction at its own times and departures), a LAJU design at what
sim.evaluate runs; each test is judged in the period its trips are made. Every boarding waits half the headway, as in
sim.py; a change at a hub of a LAJU design waits sim.TIMED_MIN (its timetable times them; BAS.MY's does not). Walking:
kelantan.WALK_MIN_PER_KM along the straight line, at most kelantan.WALK_CAP_MIN (~880 m): farther, people ride a
motorbike or get dropped off.

    python exam.py                                  -> self-check: "ok ..."
    python exam.py "BAS.MY today" "Kelantan network" -> the exam of the newest saved runs of those names, side by side
    python exam.py --basmy                          -> saves BAS.MY Kota Bharu's official timetable as the run "BAS.MY today"
"""
import csv
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import kelantan
import routes
from sim import DWELL_MIN, LAYOVER_MIN, TIMED_MIN, UTURN_MIN, electric, evaluate, quickest

HERE = Path(__file__).resolve().parent
TARGETS = HERE / "cache" / "exam_places.json"  # the exam's destinations, rebuilt from OSM when deleted
GTFS = "https://api.data.gov.my/gtfs-static/mybas-kota-bharu"  # BAS.MY Kota Bharu's official timetable (data.gov.my)
GTFS_ZIP = HERE / "cache" / "basmy_gtfs.zip"  # delete it to fetch the timetable again
TESTS = (  # destination kinds, minutes door to door, "nearest" = the one closest to home / "any" of them, the period of the
    # day it is judged in (kelantan.PERIODS: pupils go in the morning peak, outings in the evening peak): knobs
    ("school", ("school",), 30, "nearest", 0),
    ("clinic or hospital", ("clinic", "hospital"), 45, "any", 1),
    ("mall or market", ("mall", "marketplace"), 45, "any", 1),
    ("rail station", ("rail",), 45, "any", 1),
    ("park, stadium or beach", ("park", "stadium", "beach"), 60, "any", 2),
    ("town", ("town",), 45, "nearest", 1),
)
TRIP_PERIOD = 1  # the planner's test trips: in the day
TRIPS = (  # the planner's test trips (2026-10-10 / 18): (from, lat, lng), (to, lat, lng), placed by their OSM names
    (("SK Kota", 6.085193, 102.233006), ("KB Mall", 6.117549, 102.241033)),
    (("SK Kota", 6.085193, 102.233006), ("AEON Mall Kota Bharu", 6.110255, 102.230746)),
    (("Wakaf Che Yeh", 6.08436, 102.238398), ("HRPZ II", 6.124773, 102.245687)),
)
TOWN_PER_KM2, STOP_M = kelantan.TOWN_PER_KM2, kelantan.STOP_M  # town / rural, and the walk that counts as covered
MAX_CHANGES = 2
REACH_M = 1000 * kelantan.WALK_CAP_MIN / kelantan.WALK_MIN_PER_KM


def _walk(metres):
    return metres / 1000 * kelantan.WALK_MIN_PER_KM


@lru_cache
def targets():
    """{kind: (n, 2) metres}: kelantan.places() by kind (schools, clinics, malls, rail stations, parks, ...), and towns
    (OSM cities and towns, cached in TARGETS)."""
    if not TARGETS.exists():
        poly = kelantan.boundary()
        towns = kelantan._osm("points", "place IN ('city', 'town')").geometry
        TARGETS.write_text(json.dumps({"town": [[g.y, g.x] for g in towns if poly.contains(g)]}))
    out = {"town": list(kelantan._m(*np.array(json.loads(TARGETS.read_text())["town"]).reshape(-1, 2).T))}
    for x in kelantan.places():
        out.setdefault(x["kind"], []).append(kelantan._m([x["lat"]], [x["lng"]])[0])
    return {k: np.array(v).reshape(-1, 2) for k, v in out.items()}


@lru_cache
def homes():
    """(metres (n, 2), people (n,), town (n,) bool, district (n,)): where Kelantan's people live (kelantan.people)."""
    import geopandas as gpd

    lat, lng, pop, _ = kelantan.population()
    blat, blng, ppl = kelantan.people()
    P = kelantan._m(blat, blng)
    town = (pop / 0.74)[cKDTree(kelantan._m(lat, lng)).query(P)[1]] >= TOWN_PER_KM2  # H3 hexagons: 0.74 km2
    d = kelantan.districts()
    j = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy(blng, blat), crs=d.crs), d, predicate="within", how="left")
    return P, ppl, town, j[~j.index.duplicated()]["key"].fillna("other").to_numpy()


def expand(p, net, rs, express=None):
    """Route set rs (key-stop indices) at the stops a bus serves: (road node of each stop (S,), per route its stops as
    indices into those, minutes to the next stop going out (fwd) and coming back (bwd)). A link's minutes (p.time) are
    shared by its stops by road distance; an express link skips its stops and their dwell."""
    R, nodes = kelantan.roads(), net["nodes"]
    index, seqs, fwd, bwd = {}, [], [], []
    for i, r in enumerate(rs):
        seq, f, b = [nodes[r[0]]], [], []
        for k, (x, y) in enumerate(zip(r, r[1:])):
            pth, mid = net["paths"][min(x, y), max(x, y)], net["mids"][min(x, y), max(x, y)]
            if x > y:
                pth, mid = pth[::-1], mid[::-1]
            fast = bool(express and express[i][k])
            cum = np.r_[0, np.cumsum(np.hypot(*np.diff(R["P"][pth], axis=0).T))]
            at = dict(zip(pth.tolist(), cum / max(cum[-1], 1e-9)))
            share = np.diff([0.0] + ([] if fast else [at[v] for v in mid]) + [1.0])
            skip = DWELL_MIN * len(mid) if fast else 0.0
            turn = UTURN_MIN if k and (r[k - 1], x, y) in p.uturn else 0.0
            if turn:
                b[-1] += turn  # coming back, the bus turns at x before heading on to r[k - 1]
            f += list((p.time[x, y] - skip) * share)
            f[len(f) - len(share)] += turn
            b += list((p.time[y, x] - skip) * share)
            seq += ([] if fast else list(mid)) + [nodes[y]]
        seqs.append(np.array([index.setdefault(int(v), len(index)) for v in seq]))
        fwd.append(np.maximum(f, 1e-9))
        bwd.append(np.maximum(b, 1e-9))
    return np.array(list(index), int), seqs, fwd, bwd


def transit(seqs, fwd, bwd, heads, S, changes=MAX_CHANGES, hubs=()):
    """Quickest minutes between every two of S stops, at most `changes` changes, and the changes each takes: (T, C).
    Every boarding waits half its route's headway (inf: the route does not run), a change at one of the stops `hubs`
    at most TIMED_MIN; a route whose bwd is None runs one way only. The graph has a copy of every stop per number of
    buses boarded so far, so a trip that would need more changes has no path at all (sim.py instead drops a trip whose
    quickest path needs too many; here a slower path with fewer changes still counts)."""
    L, M = changes + 1, sum(len(s) for s in seqs)
    timed = np.isin(np.arange(S), list(hubs))
    rows, cols, w, base, start = [], [], [], (L + 1) * S, 0
    for s, f, b, h in zip(seqs, fwd, bwd, heads):
        for l in range(1, L + 1 if np.isfinite(h) else 1):
            node = base + (l - 1) * M + start + np.arange(len(s))
            back = b is not None
            rows += [(l - 1) * S + s, node, node[:-1]] + [node[1:]] * back  # board, alight, ride out, ride back
            cols += [node, l * S + s, node[1:]] + [node[:-1]] * back
            w += [np.where(timed[s] & (l > 1), min(TIMED_MIN, h / 2), h / 2), np.full(len(s), 1e-9), f] + [b] * back
        start += len(s)
    G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(base + L * M,) * 2).tocsr()
    T, C = np.full((S, S), np.inf), np.zeros((S, S), np.int8)
    for a in range(0, S, 256):
        D = dijkstra(G, indices=np.arange(a, min(a + 256, S)))
        lay = np.stack([D[:, l * S:(l + 1) * S] for l in range(1, L + 1)])
        T[a:a + 256], C[a:a + 256] = lay.min(0), lay.argmin(0)
    np.fill_diagonal(T, 0)
    return T, C


def via(T, t, w):
    """(S,): from every stop, the quickest way to any of stops t, plus w (the walk from t onwards)."""
    out = np.full(len(T), np.inf)
    for x, y in zip(t, w):
        np.minimum(out, T[:, x] + y, out=out)
    return out


def _pairs(A, B, metres=REACH_M):
    """Every (i, j) with A[i] within `metres` of B[j], and the walking minutes between them."""
    lists = cKDTree(B).query_ball_point(A, metres) if len(B) else [[] for _ in A]
    i = np.repeat(np.arange(len(A)), [len(x) for x in lists])
    j = np.concatenate([np.asarray(x, int) for x in lists]) if len(i) else np.zeros(0, int)
    return i, j, _walk(np.hypot(*(A[i] - B[j]).T)) if len(i) else np.zeros(0)


def exam(p, net, rs, heads, express=None, timetable=None, hubs=()):
    """The exam of route set rs (key-stop indices), route i running every heads[i][k] minutes in period k
    (kelantan.PERIODS; one number = all day; inf = no bus); changes at the key stops `hubs` are timed. With a timetable
    (BAS.MY today): the access tests ride its own stops, each direction at its official times (official()), and
    timetable[i]["minutes"] are route i's minutes out and back. -> dict (see report)."""
    R = kelantan.roads()
    hours = np.array([b - a for _, a, b in kelantan.PERIODS], float)
    heads = np.array(heads, float).reshape(len(rs), -1) * np.ones(len(hours))
    if timetable is None:
        v, seqs, fwd, bwd = expand(p, net, rs, express)
        SP, legs, timed = R["P"][v], heads, np.flatnonzero(np.isin(v, net["nodes"][list(hubs)]))
        out_back = np.array([(f.sum(), b.sum()) for f, b in zip(fwd, bwd)]).reshape(-1, 2)
    else:  # BAS.MY: each direction a one-way route through its own stops at its official minutes and departures
        SP, seqs, fwd, legs = official()
        bwd, timed, out_back = [None] * len(seqs), [], np.array([t["minutes"] for t in timetable], float)
    S = len(SP)
    nets = {}  # period -> (T, C)

    def at(k):
        if k not in nets:
            nets[k] = transit(seqs, fwd, bwd, legs[:, k], S, hubs=timed)
        return nets[k]

    P, ppl, town, district = homes()
    hi, hs, hw = _pairs(P, SP)
    tg, out = targets(), {"access": {}, "foot": {}, "districts": {}}
    passed = {}
    for name, kinds, limit, how, k in TESTS:
        T, C = at(k)
        Q = np.vstack([tg.get(k, np.zeros((0, 2))) for k in kinds])
        dq, near = cKDTree(Q).query(P)
        time = np.where(dq <= REACH_M, _walk(dq), np.inf)  # on foot all the way
        foot = time <= limit
        qi, qt, qw = _pairs(Q, SP)
        if how == "any":
            reach = via(T, qt, qw)[hs]
        else:  # their own nearest place: per place, the quickest way there from every stop
            B = np.full((len(Q), S), np.inf)
            for q in np.unique(qi):
                B[q] = via(T, qt[qi == q], qw[qi == q])
            reach = B[near[hi], hs]
        np.minimum.at(time, hi, hw + reach)
        passed[name] = time <= limit
        out["access"][name] = float(ppl[passed[name]].sum() / ppl.sum())
        out["foot"][name] = float(ppl[foot].sum() / ppl.sum())
    for d in sorted(set(district)):
        here = district == d
        out["districts"][d] = {name: float(ppl[ok & here].sum() / ppl[here].sum()) for name, ok in passed.items()}
    dist = cKDTree(SP).query(P)[0]
    ok = np.where(town, dist <= STOP_M[0], dist <= STOP_M[1])
    out["coverage"] = {k: float(ppl[ok & m].sum() / ppl[m].sum()) for k, m in (("town", town), ("rural", ~town), ("all", town | ~town))}
    out["trips"], (T, C) = {}, at(TRIP_PERIOD)
    for (a, alat, alng), (b, blat, blng) in TRIPS:
        o, d = kelantan._m([alat], [alng]), kelantan._m([blat], [blng])
        _, so, wo = _pairs(o, SP)
        _, sd, wd = _pairs(d, SP)
        t = wo[:, None] + T[np.ix_(so, sd)] + wd[None] if len(so) and len(sd) else np.full((1, 1), np.inf)
        k = np.unravel_index(t.argmin(), t.shape)
        out["trips"][f"{a} -> {b}"] = (float(t[k]), int(C[so[k[0]], sd[k[1]]])) if np.isfinite(t[k]) else (math.inf, -1)
    m = evaluate(p, rs)
    quick = quickest(p)[0]
    out["crooked"] = sum(max(o / quick[r[0], r[-1]], b / quick[r[-1], r[0]]) > kelantan.MAX_DETOUR
                         for r, (o, b) in zip(rs, out_back))
    ends = R["P"][net["nodes"][[x for r in rs for x in (r[0], r[-1])]]]
    out["rest_ends"] = int((cKDTree(kelantan.rests()).query(ends)[0].reshape(-1, 2) <= kelantan.REST_M).all(1).sum())
    out["rest_near"] = sum(bool(p.rest_end[r[0]] and p.rest_end[r[-1]]) for r in rs)  # within kelantan.REST_RUN
    cycle = out_back.sum(1) + LAYOVER_MIN + np.array([p.rest_min[r[0]] + p.rest_min[r[-1]] for r in rs])
    buses = np.where(np.isfinite(heads), np.ceil(cycle[:, None] / heads - 0.05), 0)  # headways are saved to 0.1 min
    km = np.array(m["route_km"])
    out["buses_k"] = buses.sum(0).astype(int).tolist()  # on the road in each period
    out["buses"] = max(out["buses_k"])
    out["bus_km_day"] = float((2 * km[:, None] * np.where(np.isfinite(heads), 60 * hours / heads, 0)).sum())
    out["empty_km_day"] = float(sum(b * min(p.depot_km[r[0]], p.depot_km[r[-1]]) for r, b in zip(rs, buses.max(1))))
    out["electric"] = sum(electric(k, c, hours.sum()) for k, c in zip(km, cycle))
    out["route_minutes"] = [round(float(o)) for o in out_back[:, 0]]
    out |= {"routes": len(rs), "stops": S, "hubs": len(hubs)}
    return out


def straight(p, r):
    """A route traced through landmarks, without the trips into them and back the bus does not make: a key stop
    visited twice cuts out what lies between, and U-turns go where a direct link exists (routes.unturn). Traced
    BAS.MY routes had 1-8 such U-turns each, 10 min apiece (2026-10-28)."""
    out = []
    for s in r:
        out = out[:out.index(s) + 1] if s in out else out + [s]
    return routes.unturn(p, out)


def periods(deps, ways=1):
    """Headway in each period (kelantan.PERIODS) of buses leaving at `deps` (minutes after midnight), shared by `ways`
    directions: the period's minutes / its departures each way; inf with none."""
    h = np.asarray(deps, float) / 60
    return [60 * (b - a) * ways / k if (k := int(((h >= a) & (h < b)).sum())) else math.inf for _, a, b in kelantan.PERIODS]


@lru_cache
def basmy():
    """BAS.MY Kota Bharu from its official GTFS (Konsortium E-Mutiara via data.gov.my, cached in GTFS_ZIP): per route, by
    ref (D10 ... D62), its name and, per direction, its stops in order (name, lat, lng), the minutes from each stop to
    the next and its departures (minutes after midnight). One timetable for every day: all its trips are "WEEKEND"
    (2026-10-31), each direction one stop pattern at one speed all day."""
    kelantan._download(GTFS, GTFS_ZIP)
    z = zipfile.ZipFile(GTFS_ZIP)
    rows = lambda n: csv.DictReader(io.TextIOWrapper(z.open(n), "utf-8-sig"))
    clock = lambda t: sum(int(x) * m for x, m in zip(t.split(":"), (60, 1, 1 / 60)))
    stops = {s["stop_id"]: (s["stop_name"], float(s["stop_lat"]), float(s["stop_lon"])) for s in rows("stops.txt")}
    times = defaultdict(list)
    for x in rows("stop_times.txt"):
        times[x["trip_id"]].append((int(x["stop_sequence"]), clock(x["departure_time"]), x["stop_id"]))
    out = {r["route_id"]: {"ref": r["route_short_name"], "name": r["route_long_name"], "dirs": {}} for r in rows("routes.txt")}
    for t in rows("trips.txt"):
        seq = sorted(times[t["trip_id"]])
        d = out[t["route_id"]]["dirs"].setdefault(t["direction_id"], {
            "stops": [stops[s] for _, _, s in seq], "minutes": np.diff([m for _, m, _ in seq]).tolist(), "deps": []})
        d["deps"].append(seq[0][1])
    return [r | {"dirs": [r["dirs"][k] for k in sorted(r["dirs"])]} for r in sorted(out.values(), key=lambda r: r["ref"])]


def official():
    """BAS.MY's own stops (metres (S, 2)) and, for each direction of each route, its stops (indices into them), the
    minutes to the next stop and its headway in each period: one-way routes for transit()."""
    key, seqs, fwd, heads = {}, [], [], []
    for r in basmy():
        for d in r["dirs"]:
            seqs.append(np.array([key.setdefault(s[1:], len(key)) for s in d["stops"]]))
            fwd.append(np.maximum(d["minutes"], 1e-9))
            heads.append(periods(d["deps"]))
    return kelantan._m(*np.array(list(key)).T), seqs, fwd, np.array(heads)


def onto(p, net, stops):
    """A route through `stops` ((name, lat, lng) in order) on the key stops of p: each stop's nearest key stop, joined by
    the quickest stop-graph paths, loops and U-turns cut (straight)."""
    z = cKDTree(kelantan.roads()["P"][net["nodes"]]).query(kelantan._m(*np.array([s[1:] for s in stops]).T))[1]
    D, pred = routes.quickest(p)
    seq = [int(z[0])]
    for b in z[1:]:
        if b != seq[-1] and np.isfinite(D[seq[-1], b]):
            seq += routes.path(pred, seq[-1], b)[1:]
    return straight(p, seq)


def load(name):
    """(Problem, net, routes, headways per period, express, timetable, hubs) of the newest saved run named `name`. A
    network is judged at the headways it was saved with, so a later change of the trip model or the rules does not
    change its exam; if its routes no longer match those (a stop it used is gone), at what sim.evaluate runs today."""
    import server

    d = next((x for x in (json.loads(f.read_text()) for f in sorted(server.RESULTS.glob("run-*.json"), reverse=True))
              if x["input"]["name"] == name), None)
    if d is None:
        raise SystemExit(f"no saved run named {name!r}")
    p, net, _, lk = server._problem(server.DesignRequest(**d["input"]))
    rs = lk + server._routes(d, p, net, lk)
    if "timetable" in d:  # BAS.MY today: its official timetable, every bus stopping at every stop
        t = d["timetable"]
        return p, net, rs, [[math.inf if h is None else h for h in x["periods"]] for x in t], None, t, []
    m = d["output"]["metrics"]
    if len(m["headway"]) != len(rs) or len(m["express"]) != len(rs):
        m = evaluate(p, rs, detail=True)
    heads = [[math.inf if h is None else h for h in x] for x in m["periods"]] if "periods" in m else m["headway"]
    return p, net, rs, heads, m["express"], None, m.get("hubs", [])  # runs before Phase 3: one headway all day


def save_basmy():
    """BAS.MY Kota Bharu's official timetable (basmy()) as the saved run "BAS.MY today": each route on the key stops of
    the saved "Kelantan network" nearest its stops (onto()), locked, so its riders are counted on the same key stops as a
    LAJU design's and the website draws it; the exam rides its own stops and times (official())."""
    import server

    b, runs = basmy(), [json.loads(f.read_text()) for f in sorted(server.RESULTS.glob("run-*.json"), reverse=True)]
    spots = next(x["input"]["spots"] for x in runs if x["input"]["name"] == "Kelantan network")
    base = server.DesignRequest(name="BAS.MY today", bbox=server.KELANTAN, n_routes=len(b), n_stops=150, optimizer="baseline",
                                max_route_min=0, spots=spots)
    p, net, _, _ = server._problem(base)
    R = kelantan.roads()
    at = lambda z: server.Stop(lat=float(R["lat"][net["nodes"][z]]), lng=float(R["lng"][net["nodes"][z]]))
    req = base.model_copy(update={"locked": [[at(z) for z in onto(p, net, r["dirs"][0]["stops"])] for r in b]})
    p, net, _, lk = server._problem(req)
    assert len(net["nodes"]) == len(spots) and all(len(r) >= 2 for r in lk), "the routes run on the network's key stops"
    heads = [periods(sum((d["deps"] for d in r["dirs"]), []), len(r["dirs"])) for r in b]  # each way, both directions
    server.RESULTS.mkdir(exist_ok=True)
    f = next((f for f in server.RESULTS.glob("run-*.json") if json.loads(f.read_text())["input"]["name"] == req.name),
             server.RESULTS / f"run-{datetime.now():%Y%m%d-%H%M%S-%f}.json")  # replaces the last one
    out = server._output(p, net, lk, req)
    out["metrics"] = evaluate(p, lk, detail=True, fixed=heads)  # its timetable
    server._save(f, {"input": req.model_dump(), "output": out, "timetable": [
        {"ref": r["ref"], "name": r["name"], "minutes": [sum(d["minutes"]) for d in r["dirs"]],
         "periods": [None if math.isinf(h) else round(h, 1) for h in x]} for r, x in zip(b, heads)]})
    print("saved", f.stem, "BAS.MY today:", len(lk), "routes;", round(out["metrics"]["boardings"]), "boardings a day")


def report(names, results):
    w = 26
    row = lambda label, xs: print(f"{label:<34}" + "".join(f"{x:>{w}}" for x in xs))
    row("", names)
    row("routes / stops / buses", [f"{e['routes']} / {e['stops']} / {e['buses']}" for e in results])
    row("buses: peak / day / eve / night", [" / ".join(map(str, e["buses_k"])) for e in results])
    row("bus-km a day / empty from depots", [f"{e['bus_km_day']:,.0f} / {e['empty_km_day']:,.0f}" for e in results])
    row("timed hubs / e-bus routes", [f"{e['hubs']} / {e['electric']} of {e['routes']}" for e in results])
    row("covered: town / rural / all", [" / ".join(f"{e['coverage'][k]:.0%}" for k in ("town", "rural", "all")) for e in results])
    print("can get there (on foot alone):")
    for name, _, limit, how, _ in TESTS:
        row(f"  {'nearest ' if how == 'nearest' else ''}{name} {limit} min",
            [f"{e['access'][name]:.0%} ({e['foot'][name]:.0%})" for e in results])
    row("  average", [f"{np.mean(list(e['access'].values())):.0%}" for e in results])
    print("test trips (minutes door to door, changes):")
    for t in results[0]["trips"]:
        row(f"  {t}", [("no bus" if not math.isfinite(e["trips"][t][0]) else
                        f"{e['trips'][t][0]:.0f} min, {e['trips'][t][1]} chg") for e in results])
    row("routes over 1.5x / both ends rest", [f"{e['crooked']} / {e['rest_ends']} of {e['routes']}" for e in results])
    row("  ends within a 5-min drive of one", [f"{e['rest_near']} of {e['routes']}" for e in results])
    print("by district, average of the access tests:")
    for d in sorted(results[0]["districts"]):
        row(f"  {d}", [f"{np.mean(list(e['districts'].get(d, {0: 0}).values())):.0%}" for e in results])


if __name__ == "__main__":
    if sys.argv[1:] == ["--basmy"]:
        save_basmy()
    elif len(sys.argv) > 1:
        results = []
        for n in sys.argv[1:]:
            results.append(exam(*load(n)))
            print(f"{n}: route minutes out {results[-1]['route_minutes']}", flush=True)
        report(sys.argv[1:], results)
    else:  # a line of 4 stops: route A 0-1, B 1-2, C 2-3, every 10 / 20 / 30 min, 5 / 6 / 7 minutes a hop
        seqs, f = [np.array([0, 1]), np.array([1, 2]), np.array([2, 3])], [np.array([5.0]), np.array([6.0]), np.array([7.0])]
        T, C = transit(seqs, f, f, [10, 20, 30], 4)
        assert abs(T[0, 1] - 10) < 1e-6 and C[0, 1] == 0, (T[0, 1], C[0, 1])  # wait 5 + ride 5
        assert abs(T[0, 2] - 26) < 1e-6 and C[0, 2] == 1  # + wait 10 + ride 6
        assert abs(T[0, 3] - 48) < 1e-6 and C[0, 3] == 2 and abs(T[3, 0] - 48) < 1e-6  # + wait 15 + ride 7, both ways
        assert np.isinf(transit(seqs, f, f, [10, 20, 30], 4, changes=1)[0][0, 3])  # 2 changes not allowed: no way
        assert abs(transit(seqs, f, f, [10, 20, 30], 4, hubs=[1])[0][0, 2] - (5 + 5 + TIMED_MIN + 6)) < 1e-6  # timed at 1
        assert np.isinf(transit(seqs, f, f, [10, math.inf, 30], 4)[0][0, 2])  # route B does not run then
        assert periods(390 + 60 * np.arange(13)) == [60, 60, 60, math.inf]  # 06:30 ... 18:30: none at night
        assert periods([420, 660, 900, 600, 840, 1080], ways=2) == [360, 210, 360, math.inf]  # D61: 3 a day each way
        one = transit(seqs, f, [None] * 3, [10, 20, 30], 4)[0]  # one way: 0 -> 3 only
        assert abs(one[0, 3] - 48) < 1e-6 and np.isinf(one[3, 0])
        T, C = transit(seqs + [np.array([0, 1, 2, 3])], f + [np.array([5.0, 6.0, 7.0])], f + [np.array([5.0, 6.0, 7.0])],
                       [10, 20, 30, 40], 4)
        assert abs(T[0, 3] - 38) < 1e-6 and C[0, 3] == 0  # the through route (wait 20 + 18) beats changing twice
        assert abs(T[0, 2] - 26) < 1e-6 and C[0, 2] == 1  # 26 by changing beats 31 on the through route
        assert list(via(T, [2, 3], [4.0, 1.0])) == [min(T[s, 2] + 4, T[s, 3] + 1) for s in range(4)]
        from sim import Problem

        flat = Problem(time=np.ones((5, 5)), od=np.zeros((5, 5)), xy=np.zeros((5, 2)))
        assert straight(flat, [0, 1, 2, 1, 3, 4, 3]) == [0, 1, 3], straight(flat, [0, 1, 2, 1, 3, 4, 3])  # spurs cut
        print("ok", T.round(1).tolist())
