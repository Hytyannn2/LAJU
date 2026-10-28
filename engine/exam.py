"""Brain v2, Phase 0: the exam every network must pass (PLAN.txt section 9). A planner's and a rider's view of a route
set, judged at the stops a bus really serves, not by the optimisers' score:

  access    share of people who can get there door to door, on foot or by bus with at most 2 changes (walk + wait +
            ride + the waits at changes, real minutes): their nearest school in 30 min, a clinic or hospital in 45, a
            mall or market in 45, a rail station in 45, a park, stadium or beach in 60, their nearest town in 45. Per
            district too (fairness)
  coverage  people within 400 m (town) / 800 m (rural) of a stop a bus serves
  trips     the planner's own test trips (SK Kota -> KB Mall, ...)
  shape     routes over 1.5x their quickest drive; route ends where a bus can wait and the driver rest
  fleet     buses and bus-km a day, each route at its own headway

Each network runs at its own headways and speeds: BAS.MY at its timetable (basmy.json: how often, how long one way), a
LAJU design at what sim.evaluate runs.
Every boarding waits half the headway, as in sim.py. Walking: kelantan.WALK_MIN_PER_KM along the straight line, at most
kelantan.WALK_CAP_MIN (~880 m): farther, people ride a motorbike or get dropped off.

    python exam.py                                  -> self-check: "ok ..."
    python exam.py "BAS.MY today" "Kelantan network" -> the exam of the newest saved runs of those names, side by side
    python exam.py --basmy                          -> saves BAS.MY Kelantan (basmy.json) as the run "BAS.MY today"
"""
import json
import math
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import kelantan
import routes
from sim import DWELL_MIN, LAYOVER_MIN, UTURN_MIN, evaluate, quickest

HERE = Path(__file__).resolve().parent
TARGETS = HERE / "cache" / "exam_places.json"  # the exam's destinations, rebuilt from OSM when deleted
BASMY = HERE / "basmy.json"  # BAS.MY Kelantan: each route's stops in order and how often it runs
TESTS = (  # destination kinds, minutes door to door, "nearest" = the one closest to home / "any" of them: knobs
    ("school", ("school",), 30, "nearest"),
    ("clinic or hospital", ("clinic", "hospital"), 45, "any"),
    ("mall or market", ("mall", "marketplace"), 45, "any"),
    ("rail station", ("rail",), 45, "any"),
    ("park, stadium or beach", ("park", "stadium", "beach"), 60, "any"),
    ("town", ("town",), 45, "nearest"),
)
TRIPS = (  # the planner's test trips (2026-10-10 / 18): (from, lat, lng), (to, lat, lng), placed by their OSM names
    (("SK Kota", 6.085193, 102.233006), ("KB Mall", 6.117549, 102.241033)),
    (("SK Kota", 6.085193, 102.233006), ("AEON Mall Kota Bharu", 6.110255, 102.230746)),
    (("Wakaf Che Yeh", 6.08436, 102.238398), ("HRPZ II", 6.124773, 102.245687)),
)
TOWN_PER_KM2 = 1000   # a home in a population hexagon this dense is "town"
STOP_M = (400, 800)   # town, rural: how near a stop counts as covered (planner, 2026-10-28)
REST = ("bus_station", "rail", "mall", "hospital", "university", "college", "stadium")  # a bus can wait there
REST_M = 300
MAX_CHANGES = 2
SERVICE_H = 18        # 6 am to 12 am (planner, 2026-10-28)
REACH_M = 1000 * kelantan.WALK_CAP_MIN / kelantan.WALK_MIN_PER_KM


def _walk(metres):
    return metres / 1000 * kelantan.WALK_MIN_PER_KM


@lru_cache
def targets():
    """{kind: (n, 2) metres}: schools, clinics, hospitals, malls, markets, universities, colleges, bus stations from
    kelantan.places(); rail stations, parks, stadiums, beaches and towns from OSM (cached in TARGETS)."""
    if not TARGETS.exists():
        poly, out = kelantan.boundary(), {"rail": [], "park": [], "stadium": [], "beach": [], "town": []}
        pts = kelantan._osm("points", "place IN ('city', 'town') OR other_tags LIKE '%railway%' OR other_tags LIKE '%leisure%'"
                                      " OR other_tags LIKE '%natural%'")
        for g, nm, place, other in zip(pts.geometry, pts["name"], pts["place"], pts["other_tags"]):
            t = kelantan._tags(other)
            kind = ("town" if place in ("city", "town") else "rail" if t.get("railway") in ("station", "halt", "stop")
                    else t.get("leisure") or t.get("natural"))
            if kind in out and (kind != "park" or isinstance(nm, str)) and poly.contains(g):
                out[kind].append([g.y, g.x])
        areas = kelantan._osm("multipolygons", "leisure IN ('park', 'stadium', 'sports_centre') OR natural = 'beach'")
        for g, nm, leisure, natural in zip(areas.geometry, areas["name"], areas["leisure"], areas["natural"]):
            kind = "stadium" if leisure in ("stadium", "sports_centre") else leisure or natural
            c = g.representative_point()
            if kind in out and (kind != "park" or isinstance(nm, str)) and poly.contains(c):
                out[kind].append([c.y, c.x])
        TARGETS.write_text(json.dumps(out))
    out = {k: kelantan._m(*np.array(v).reshape(-1, 2).T) for k, v in json.loads(TARGETS.read_text()).items()}
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


def transit(seqs, fwd, bwd, heads, S, changes=MAX_CHANGES):
    """Quickest minutes between every two of S stops, at most `changes` changes, and the changes each takes: (T, C).
    Every boarding waits half its route's headway. The graph has a copy of every stop per number of buses boarded so
    far, so a trip that would need more changes has no path at all (sim.py instead drops a trip whose quickest path
    needs too many; here a slower path with fewer changes still counts)."""
    L, M = changes + 1, sum(len(s) for s in seqs)
    rows, cols, w, base, start = [], [], [], (L + 1) * S, 0
    for s, f, b, h in zip(seqs, fwd, bwd, heads):
        for l in range(1, L + 1):
            node = base + (l - 1) * M + start + np.arange(len(s))
            rows += [(l - 1) * S + s, node, node[:-1], node[1:]]  # board, alight, ride out, ride back
            cols += [node, l * S + s, node[1:], node[:-1]]
            w += [np.full(len(s), h / 2), np.full(len(s), 1e-9), f, b]
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


def exam(p, net, rs, heads, express=None, minutes=None):
    """The exam of route set rs (key-stop indices), route i running every heads[i] minutes and, where minutes[i] is
    given (a real timetable), taking that long one way. -> dict (see report)."""
    R = kelantan.roads()
    v, seqs, fwd, bwd = expand(p, net, rs, express)
    for i, mm in enumerate(minutes or []):
        if mm:  # traced routes keep their roads but run at their real speed (their tracing adds U-turns: straight())
            fwd[i], bwd[i] = fwd[i] * mm / fwd[i].sum(), bwd[i] * mm / bwd[i].sum()
    SP, S = R["P"][v], len(v)
    T, C = transit(seqs, fwd, bwd, heads, S)
    P, ppl, town, district = homes()
    hi, hs, hw = _pairs(P, SP)
    tg, out = targets(), {"access": {}, "foot": {}, "districts": {}}
    passed = {}
    for name, kinds, limit, how in TESTS:
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
    out["trips"] = {}
    for (a, alat, alng), (b, blat, blng) in TRIPS:
        o, d = kelantan._m([alat], [alng]), kelantan._m([blat], [blng])
        _, so, wo = _pairs(o, SP)
        _, sd, wd = _pairs(d, SP)
        t = wo[:, None] + T[np.ix_(so, sd)] + wd[None] if len(so) and len(sd) else np.full((1, 1), np.inf)
        k = np.unravel_index(t.argmin(), t.shape)
        out["trips"][f"{a} -> {b}"] = (float(t[k]), int(C[so[k[0]], sd[k[1]]])) if np.isfinite(t[k]) else (math.inf, -1)
    m = evaluate(p, rs)
    quick = quickest(p)[0]
    out["crooked"] = sum(max(f.sum() / quick[r[0], r[-1]], b.sum() / quick[r[-1], r[0]]) > kelantan.MAX_DETOUR
                         for r, f, b in zip(rs, fwd, bwd))
    rest = cKDTree(np.vstack([tg[k] for k in REST if k in tg]))
    ends = R["P"][net["nodes"][[x for r in rs for x in (r[0], r[-1])]]]
    out["rest_ends"] = int((rest.query(ends)[0].reshape(-1, 2) <= REST_M).all(1).sum())
    out["buses"] = sum(math.ceil((f.sum() + b.sum() + LAYOVER_MIN) / h) for f, b, h in zip(fwd, bwd, heads))
    out["bus_km_day"] = sum(2 * k * SERVICE_H * 60 / h for k, h in zip(m["route_km"], heads))
    out["route_minutes"] = [round(float(f.sum())) for f in fwd]
    out |= {"routes": len(rs), "stops": S}
    return out


def straight(p, r):
    """A route traced through landmarks, without the trips into them and back the bus does not make: a key stop
    visited twice cuts out what lies between, and U-turns go where a direct link exists (routes.unturn). Traced
    BAS.MY routes had 1-8 such U-turns each, 10 min apiece (2026-10-28)."""
    out = []
    for s in r:
        out = out[:out.index(s) + 1] if s in out else out + [s]
    return routes.unturn(p, out)


def load(name):
    """(Problem, net, routes, headways, express, timetable minutes) of the newest saved run named `name`, on today's
    rules."""
    import server

    d = next((x for x in (json.loads(f.read_text()) for f in sorted(server.RESULTS.glob("run-*.json"), reverse=True))
              if x["input"]["name"] == name), None)
    if d is None:
        raise SystemExit(f"no saved run named {name!r}")
    p, net, _, lk = server._problem(server.DesignRequest(**d["input"]))
    rs = lk + server._routes(d, p, net, lk)
    if "timetable" in d:  # BAS.MY: its own headways and trip times, and every bus stops at every stop
        t = d["timetable"]
        return p, net, [straight(p, r) for r in rs], [x["headway"] for x in t], None, [x["minutes"] for x in t]
    m = evaluate(p, rs, detail=True)
    return p, net, rs, m["headway"], m["express"], None


def save_basmy():
    """BAS.MY Kelantan (basmy.json) as the saved run "BAS.MY today": every route locked through its stops, so the website
    draws it and the exam reads its timetable. The website's own numbers for it use sim.py's frequencies."""
    import server

    b = json.loads(BASMY.read_text())["routes"]
    req = server.DesignRequest(name="BAS.MY today", bbox=server.KELANTAN, n_routes=len(b), n_stops=150, optimizer="baseline",
                               max_route_min=0, locked=[[server.Stop(lat=s[1], lng=s[2], name=s[0]) for s in r["stops"]] for r in b])
    p, net, _, lk = server._problem(req)
    server.RESULTS.mkdir(exist_ok=True)
    f = next((f for f in server.RESULTS.glob("run-*.json") if json.loads(f.read_text())["input"]["name"] == req.name),
             server.RESULTS / f"run-{datetime.now():%Y%m%d-%H%M%S-%f}.json")  # replaces the last one
    out = server._output(p, net, [straight(p, r) for r in lk], req)
    server._save(f, {"input": req.model_dump(), "output": out,
                     "timetable": [{k: r[k] for k in ("ref", "headway", "minutes")} for r in b]})
    print("saved", f.stem, "BAS.MY today:", len(lk), "routes")


def report(names, results):
    w = 26
    row = lambda label, xs: print(f"{label:<34}" + "".join(f"{x:>{w}}" for x in xs))
    row("", names)
    row("routes / stops / buses", [f"{e['routes']} / {e['stops']} / {e['buses']}" for e in results])
    row("bus-km a day", [f"{e['bus_km_day']:,.0f}" for e in results])
    row("covered: town / rural / all", [" / ".join(f"{e['coverage'][k]:.0%}" for k in ("town", "rural", "all")) for e in results])
    print("can get there (on foot alone):")
    for name, _, limit, how in TESTS:
        row(f"  {'nearest ' if how == 'nearest' else ''}{name} {limit} min",
            [f"{e['access'][name]:.0%} ({e['foot'][name]:.0%})" for e in results])
    row("  average", [f"{np.mean(list(e['access'].values())):.0%}" for e in results])
    print("test trips (minutes door to door, changes):")
    for t in results[0]["trips"]:
        row(f"  {t}", [("no bus" if not math.isfinite(e["trips"][t][0]) else
                        f"{e['trips'][t][0]:.0f} min, {e['trips'][t][1]} chg") for e in results])
    row("routes over 1.5x / both ends rest", [f"{e['crooked']} / {e['rest_ends']} of {e['routes']}" for e in results])
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
        T, C = transit(seqs + [np.array([0, 1, 2, 3])], f + [np.array([5.0, 6.0, 7.0])], f + [np.array([5.0, 6.0, 7.0])],
                       [10, 20, 30, 40], 4)
        assert abs(T[0, 3] - 38) < 1e-6 and C[0, 3] == 0  # the through route (wait 20 + 18) beats changing twice
        assert abs(T[0, 2] - 26) < 1e-6 and C[0, 2] == 1  # 26 by changing beats 31 on the through route
        assert list(via(T, [2, 3], [4.0, 1.0])) == [min(T[s, 2] + 4, T[s, 3] + 1) for s in range(4)]
        from sim import Problem

        flat = Problem(time=np.ones((5, 5)), od=np.zeros((5, 5)), xy=np.zeros((5, 2)))
        assert straight(flat, [0, 1, 2, 1, 3, 4, 3]) == [0, 1, 3], straight(flat, [0, 1, 2, 1, 3, 4, 3])  # spurs cut
        print("ok", T.round(1).tolist())
