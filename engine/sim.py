"""Phase 3 transit simulation (macroscopic): put every origin-destination trip on its quickest path over a route set.

Passenger time = wait (half the headway) + in-vehicle minutes + a fixed penalty per transfer. With no waiting and a
5-minute penalty this is the standard UTRP evaluation (Mandl 1979, Mumford 2013), so Mandl numbers compare with papers.
`score` folds passenger time, operator cost and unserved demand into one cost per trip: the GA fitness and RL reward.

Bus categories (Kelantan, Problem.headways): a route's length and stops make it a shuttle, a local or an inter-town
route, which sets its bus and how often it comes at least (the planner's minimum service). Busy routes run more often:
as their busiest hour needs (peak load / bus capacity, Baaj & Mahmassani 1991) or as their riders' waiting is worth
(square-root rule, Mohring 1972: the frequency where an extra bus costs what its riders' waiting saves). Riders wait half
the headway the route really runs. So quiet routes keep the minimum service with few buses, and the optimisers learn
which categories to deploy where from the score alone.

A trip needing more than Problem.max_transfers transfers counts as unserved (riders would not make it; Kelantan: 2,
as in the UTRP literature's d_un).

Walking (Kelantan, Problem.walk): every trip end adds the walk from home to the nearest stop a bus serves. Inter-town
buses run express (key stops only) on a link where another bus still stops, or where their riders save more minutes
than the people near the skipped stops then walk.
"""
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

PEAK_SHARE = 0.1  # share of daily trips in the busiest hour
LAYOVER_MIN = 10  # turnaround time per round trip
UTURN_MIN = 10    # a bus cannot U-turn mid-road: time to find a roundabout and come back (calibration knob)
DWELL_MIN = 0.5   # time lost per intermediate stop (Problem.time includes it; express links skip it)
CROOKED_MIN = 600  # each minute a route runs beyond Problem.max_detour x its quickest drive adds 1/600 to the score:
                  # as strong as the averaged rule was on 8-route designs (1/480), now for any number of routes
CATEGORIES = (  # name, minimum service (a bus at least every ... min), passengers per bus, RM per bus-km: knobs, not data
    ("shuttle", 30, 30, 2.5),    # minibus: first/last mile between homes, offices and stations
    ("local", 30, 60, 4.0),      # 12 m city bus: town centre and its suburbs
    ("intertown", 60, 45, 3.5),  # coach, seated: town to town along a corridor
)  # minimum service 10 / 15 / 30 min until 2026-10-18: quiet routes then needed as many buses as busy ones
HEADWAYS = tuple(c[1] for c in CATEGORIES)


def category(km, stops):
    """0 shuttle (under 20 km and under 25 stops), 1 local (up to 40 km and 65 stops), 2 inter-town (longer)."""
    return 0 if km < 20 and stops < 25 else 1 if km <= 40 and stops <= 65 else 2


@dataclass
class Problem:
    """A stop graph. Buses may only run between stops whose `time` is finite."""
    time: np.ndarray              # (n, n) bus minutes between adjacent stops, inf = no direct road link
    od: np.ndarray                # (n, n) trips (per day for Kelantan)
    xy: np.ndarray                # (n, 2) coordinates (RL features)
    km: np.ndarray | None = None  # (n, n) road km between adjacent stops; None = operator cost counted in minutes
    min_len: int = 2              # stops per route
    max_len: int = 8
    transfer_min: float = 5.0     # penalty per transfer
    service_hours: float = 16.0
    vot: float = 1.0              # cost per passenger-minute
    unserved_cost: float = 60.0   # cost per trip the network cannot carry
    max_buses: int = 0            # budgets, 0 = none
    max_km: float = 0.0
    max_route_min: float = 0.0    # longest one-way trip a route may take (driver shift, reliability), 0 = none
    max_detour: float = 0.0       # a route may take this x the quickest drive between its two ends (no hooks), 0 = none
    uturn: frozenset = frozenset()  # (a, b, c): going a -> b -> c means turning back at b
    headways: tuple = ()          # per bus category: a bus at least this often; () = no waiting, no fleet (Mandl convention)
    mids: np.ndarray | None = None  # (n, n) stops along each link, between its two key stops (counted for categories)
    walk: np.ndarray | None = None  # (n,) minutes the residents around each key stop walk to it, per trip end
    walk_save: np.ndarray | None = None  # (n, n) minutes less for a's residents while a bus stops along link (a, b)
    walk_weight: float = 1.0      # a walking minute costs this many riding minutes (heat, effort); reported minutes stay real
    max_transfers: int | None = None  # a trip needing more transfers counts as unserved; None = any (Mandl convention)


def quickest(p):
    """All-pairs quickest paths on the stop graph: (minutes, predecessors)."""
    return dijkstra(csr_matrix(np.where(np.isfinite(p.time), p.time, 0)), return_predecessors=True)


def evaluate(p, routes, detail=False):
    """Metrics for a route set (lists of stop indices). detail=True adds transfers and per-segment bus loads."""
    n = len(p.od)
    routes = [list(r) for r in routes if len(r) >= 2]
    km = p.time if p.km is None else p.km
    route_km = [float(sum(km[a, b] for a, b in zip(r, r[1:]))) for r in routes]
    if p.headways:
        stops = [len(r) + (0 if p.mids is None else int(sum(p.mids[a, b] for a, b in zip(r, r[1:])))) for r in routes]
        cats = [category(k, s) for k, s in zip(route_km, stops)]
        heads = [p.headways[c] for c in cats]
    else:
        cats, heads = [], [0.0] * len(routes)
    rows, cols, w, base = [], [], [], n  # nodes: 0..n-1 = stops, then one node per (route, stop)
    for r, h in zip(routes, heads):
        board = p.transfer_min + h / 2  # every boarding pays this; the first boarding's penalty is removed below
        for k, s in enumerate(r):
            rows += [s, base + k]
            cols += [base + k, s]
            w += [board, 1e-9]  # board, alight (csgraph drops exact zeros)
            if k:
                turn = UTURN_MIN if k >= 2 and (r[k - 2], r[k - 1], s) in p.uturn else 0
                rows += [base + k - 1, base + k]
                cols += [base + k, base + k - 1]
                w += [p.time[r[k - 1], s] + turn, p.time[s, r[k - 1]] + turn]
        base += len(r)
    G = csr_matrix((w, (rows, cols)), shape=(base, base))
    D, P = dijkstra(G, indices=np.arange(n), return_predecessors=True)
    T = D[:, :n] - p.transfer_min
    reach = np.isfinite(T)
    total = float(p.od.sum())
    paths = _transfers_and_loads(p, routes, P, reach, total) if detail or p.headways or p.max_transfers is not None else None
    served = float(p.od[reach].sum())
    pax_min = float((p.od[reach] * T[reach]).sum())
    trips = np.where(reach, p.od, 0)
    ends = trips.sum(1) + trips.sum(0)  # served trips starting or ending at each key stop

    m = {
        "unserved": float(1 - served / total) if total else 0.0,
        "route_min": [float(sum(p.time[a, b] for a, b in zip(r, r[1:])) + _turns(p, r)) for r in routes],
        "route_min_back": [float(sum(p.time[b, a] for a, b in zip(r, r[1:])) + _turns(p, r)) for r in routes],
        "route_km": route_km,
        "total_km": sum(route_km),
    }
    if paths:
        m.update(paths)
        peak = [max(x) * PEAK_SHARE for x in m["loads"]]  # busiest segment, busiest hour
        m["peak_hour_load"] = [round(q) for q in peak]
        riders, boards = m.pop("riders"), m.pop("boardings")
    express = [[False] * (len(r) - 1) for r in routes]
    if p.headways and p.mids is not None:
        express = _express(p, routes, cats, riders, ends)
        for i, (r, e, q) in enumerate(zip(routes, express, riders)):
            skip = [DWELL_MIN * p.mids[a, b] if x else 0.0 for (a, b), x in zip(zip(r, r[1:]), e)]
            m["route_min"][i] -= sum(skip)
            m["route_min_back"][i] -= sum(skip)
            pax_min -= float(np.dot(q, skip))  # its riders no longer stop at the skipped stops
    walk = 0.0
    if p.walk is not None:
        stopping = np.zeros((n, n), bool)  # links where a bus stops at the stops between the key stops
        for r, e in zip(routes, express):
            for (a, b), x in zip(zip(r, r[1:]), e):
                stopping[a, b] = stopping[b, a] = stopping[a, b] or not x
        walk = float(ends @ (p.walk - (p.walk_save * stopping).sum(1)))
        pax_min += walk
        m["walk"] = walk / served if served else 0.0
    if p.headways:  # buses per hour: the planner's minimum, what the busiest hour needs, or what waiting riders are worth
        per_h = [max(60 / h, q / CATEGORIES[c][2], math.sqrt(30 * b * p.vot / (2 * max(k, 1e-9) * p.service_hours * CATEGORIES[c][3])))
                 for h, q, c, b, k in zip(heads, peak, cats, boards, route_km)]
        pax_min += sum(b * (60 / f - h) / 2 for b, f, h in zip(boards, per_h, heads))  # the routing assumed the minimum service
    m["att"] = pax_min / served if served else 0.0
    cost = p.vot * (pax_min + (p.walk_weight - 1) * walk) + float((p.unserved_cost * p.od)[~reach].sum())
    if p.headways:
        m["category"] = [CATEGORIES[c][0] for c in cats]
        m["express"] = express  # per route, per link
        m["headway"] = [round(60 / f, 1) for f in per_h]
        m["buses"] = [math.ceil((t + b + LAYOVER_MIN) * f / 60) for t, b, f in zip(m["route_min"], m["route_min_back"], per_h)]
        m["fleet"] = sum(m["buses"])
        # ponytail: all day at the frequency of the busiest hour; a peak/off-peak timetable would cut off-peak bus-km
        day = [2 * k * p.service_hours * f for k, f in zip(route_km, per_h)]
        m["bus_km_day"] = sum(day)
        cost += sum(CATEGORIES[c][3] * x for c, x in zip(cats, day))
    s = cost / total if total else 0.0
    if p.max_buses and m.get("fleet", 0) > p.max_buses:
        s *= 1 + (m["fleet"] - p.max_buses) / p.max_buses
    if p.max_km and m["total_km"] > p.max_km:
        s *= 1 + (m["total_km"] - p.max_km) / p.max_km
    if p.max_route_min:
        s *= 1 + sum(max(0.0, t - p.max_route_min) for t in m["route_min"] + m["route_min_back"]) / p.max_route_min
    if p.max_detour:  # crooked routes: every minute beyond max_detour x the quickest drive between the route's two ends
        if "_quick" not in p.__dict__:  # once per problem: its stop graph never changes
            p._quick = quickest(p)[0]
        over = sum(max(0.0, t - p.max_detour * p._quick[r[0], r[-1]]) + max(0.0, b - p.max_detour * p._quick[r[-1], r[0]])
                   for r, t, b in zip(routes, m["route_min"], m["route_min_back"]))
        s *= 1 + over / CROOKED_MIN  # per crooked minute, however many routes (2026-10-18; averaged over the routes before,
        # which made a crooked route nearly free in a 60-route network). The first rule (1/60 per minute, summed) outweighed
        # RM 50 per unserved trip, and the GA and RL dropped riders to avoid it (2026-10-10)
    m["score"] = s
    return m


def _express(p, routes, cats, riders, ends):
    """Per inter-town route and link: run express (key stops only) when another bus still stops at the stops between,
    or when the minutes its riders save there beat the extra minutes the people near those stops then walk. Greedy, in
    route order. ponytail: decided after the assignment, so riders who would switch to the faster bus are not counted."""
    stopping = Counter(frozenset(x) for r in routes for x in zip(r, r[1:]))
    out = []
    for r, c, q in zip(routes, cats, riders):
        row = []
        for (a, b), ride in zip(zip(r, r[1:]), q):
            dwell = DWELL_MIN * p.mids[a, b]
            lost = 0.0 if p.walk_save is None else (ends[a] * p.walk_save[a, b] + ends[b] * p.walk_save[b, a]) * p.walk_weight
            x = bool(c == 2 and dwell > 0 and (stopping[frozenset((a, b))] > 1 or ride * dwell > lost))
            stopping[frozenset((a, b))] -= x
            row.append(x)
        out.append(row)
    return out


def _turns(p, r):
    return UTURN_MIN * sum(t in p.uturn for t in zip(r, r[1:], r[2:]))


def _walk(P, o, d, flow, n):
    """Walk the quickest paths o -> d back at once: each one's boardings, and its flow on each route edge v->v+1 (fwd) /
    v+1->v (bwd), stored at v, and onto each route node it boards at (on)."""
    cur, boards = d.copy(), np.zeros(len(d), int)
    fwd, bwd, on = np.zeros(P.shape[1]), np.zeros(P.shape[1]), np.zeros(P.shape[1])
    live = cur != o
    while live.any():
        i = np.flatnonzero(live)
        prv, c = P[o[i], cur[i]], cur[i]
        b = (prv < n) & (c >= n)
        boards[i] += b
        np.add.at(on, c[b], flow[i][b])
        ride = (prv >= n) & (c >= n)
        up, down = ride & (c == prv + 1), ride & (c == prv - 1)
        np.add.at(fwd, prv[up], flow[i][up])
        np.add.at(bwd, c[down], flow[i][down])
        cur[i] = prv
        live[i] = prv != o[i]
    return boards, fwd, bwd, on


def _transfers_and_loads(p, routes, P, reach, total):
    """Walk every used shortest path back (vectorised over all OD pairs): count boardings, add up segment loads. Trips
    needing more than p.max_transfers transfers are taken off `reach` (in place): riders would not make them."""
    n = len(p.od)
    o, d = np.nonzero((p.od > 0) & reach)
    boards, fwd, bwd, on = _walk(P, o, d, p.od[o, d], n)
    if p.max_transfers is not None and (far := boards > p.max_transfers + 1).any():
        reach[o[far], d[far]] = False
        _, f2, b2, o2 = _walk(P, o[far], d[far], -p.od[o[far], d[far]], n)  # their loads come off again
        fwd, bwd, on = fwd + f2, bwd + b2, on + o2
        o, d, boards = o[~far], d[~far], boards[~far]
    flow = p.od[o, d]
    shares = [float(flow[boards == k + 1].sum() / total) if total else 0.0 for k in range(3)]
    loads, riders, boarding, base = [], [], [], n
    for r in routes:
        seg = slice(base, base + len(r) - 1)
        loads.append(np.maximum(fwd[seg], bwd[seg]).round(1).tolist())
        riders.append(fwd[seg] + bwd[seg])  # both directions, per day
        boarding.append(float(on[base: base + len(r)].sum()))  # boardings per day
        base += len(r)
    return {"d0": shares[0], "d1": shares[1], "d2": shares[2], "dun": 1 - sum(shares), "loads": loads, "riders": riders,
            "boardings": boarding}


if __name__ == "__main__":
    inf = np.inf
    line = np.full((4, 4), inf)
    for a in range(3):
        line[a, a + 1] = line[a + 1, a] = 2.0
    od = np.ones((4, 4)) - np.eye(4)
    p = Problem(time=line, od=od, xy=np.zeros((4, 2)))

    m = evaluate(p, [[0, 1, 2, 3]], detail=True)
    assert abs(m["att"] - 2 * (3 * 1 + 2 * 2 + 1 * 3) * 2 / 12) < 1e-6, m["att"]  # mean |i-j| * 2 min
    assert m["d0"] == 1 and m["unserved"] == 0 and m["route_min"] == [6.0]
    assert m["loads"] == [[3.0, 4.0, 3.0]], m["loads"]  # middle segment carries 4 trips each way

    m = evaluate(p, [[0, 1], [1, 2, 3]], detail=True)
    assert abs(m["d1"] - 4 / 12) < 1e-9, m["d1"]  # 0<->2 and 0<->3 change at stop 1
    assert abs(m["att"] - (2 * (2 + 2 + 4 + 2) + 2 * (4 + 5) + 2 * (6 + 5)) / 12) < 1e-6, m["att"]

    m = evaluate(p, [[0, 1, 2]])
    assert abs(m["unserved"] - 6 / 12) < 1e-9 and m["score"] > evaluate(p, [[0, 1, 2, 3]])["score"]
    assert evaluate(p, [])["unserved"] == 1

    capped = Problem(time=line, od=od, xy=p.xy, max_route_min=4)
    assert evaluate(capped, [[0, 1, 2]])["score"] == evaluate(p, [[0, 1, 2]])["score"]  # 4 min: within the cap
    assert evaluate(capped, [[0, 1, 2, 3]])["score"] == 2 * evaluate(p, [[0, 1, 2, 3]])["score"]  # 2 min over, both ways

    ring = np.full((4, 4), inf)
    for a in range(4):
        ring[a, (a + 1) % 4] = ring[(a + 1) % 4, a] = 2.0
    plain, bent = Problem(time=ring, od=od, xy=p.xy), Problem(time=ring, od=od, xy=p.xy, max_detour=1.5)
    assert evaluate(bent, [[0, 1, 2]])["score"] == evaluate(plain, [[0, 1, 2]])["score"]  # 4 min: as quick as it gets
    assert abs(evaluate(bent, [[0, 1, 2, 3]])["score"] - 1.01 * evaluate(plain, [[0, 1, 2, 3]])["score"]) < 1e-9  # 6 min
    # the long way round where 0-3 is 2 min: 3 min over the 1.5x allowance, both ways; a straight route next to it changes
    # nothing (it did while the penalty was averaged over the routes)
    two = [[0, 1, 2, 3], [0, 1]]
    assert abs(evaluate(bent, two)["score"] - 1.01 * evaluate(plain, two)["score"]) < 1e-9

    hops = [[0, 1], [1, 2], [2, 3]]  # 0 <-> 3 needs two transfers
    assert evaluate(p, hops)["unserved"] == 0 and evaluate(Problem(time=line, od=od, xy=p.xy, max_transfers=2), hops)["unserved"] == 0
    m = evaluate(Problem(time=line, od=od, xy=p.xy, max_transfers=1), hops, detail=True)
    assert abs(m["unserved"] - 2 / 12) < 1e-9 and m["loads"][1] == [3.0], m  # 0 <-> 3 off the middle bus too

    turned = Problem(time=line, od=od, xy=p.xy, uturn=frozenset({(0, 1, 2)}))
    assert evaluate(turned, [[0, 1, 2, 3]])["route_min"] == [6.0 + UTURN_MIN]
    assert evaluate(turned, [[0, 1, 2, 3]])["att"] > evaluate(p, [[0, 1, 2, 3]])["att"]

    km8 = np.where(np.isfinite(line), 8.0, inf)  # 8 km links
    cat = Problem(time=line, od=od, xy=p.xy, km=km8, headways=HEADWAYS)
    assert evaluate(cat, [[0, 1, 2]])["category"] == ["shuttle"]  # 16 km, 3 stops
    assert evaluate(cat, [[0, 1, 2, 3]])["category"] == ["local"]  # 24 km
    cat.mids = np.full((4, 4), 30)  # 30 stops between key stops
    assert evaluate(cat, [[0, 1, 2]])["category"] == ["local"]  # 16 km but 63 stops
    assert evaluate(cat, [[0, 1, 2, 3]])["category"] == ["intertown"]  # 94 stops
    busy = Problem(time=line, od=od * 3000, xy=p.xy, km=km8, headways=HEADWAYS, vot=0.01)
    m = evaluate(busy, [[0, 1, 2]])  # 6000 trips/day on each segment: 600 in the peak hour, 20 minibuses an hour
    assert m["headway"] == [3.0] and m["buses"] == [math.ceil((4 + 4 + LAYOVER_MIN) * 20 / 60)], m
    busy.vot = 1.0  # waiting 18,000 boardings a day worth RM 1 a minute: the square-root rule runs more buses still
    f = math.sqrt(30 * 18000 * 1.0 / (2 * 16 * 16 * 2.5))
    assert evaluate(busy, [[0, 1, 2]])["headway"] == [round(60 / f, 1)] and f > 20
    busy.vot = 0.01
    busy.max_buses = 3  # 6 buses needed: twice the budget doubles the score
    assert abs(evaluate(busy, [[0, 1, 2]])["score"] - 2 * m["score"]) < 1e-9

    dwelt = np.where(np.isfinite(line), 2 + DWELL_MIN * 30, inf)  # as in Kelantan: link time includes the 30 stops' dwell
    walk = np.where(np.isfinite(line), 50.0, 0)  # a resident walks 120 min to the key stop, 50 less per stopping link
    exp = Problem(time=dwelt, od=od, xy=p.xy, km=km8, headways=HEADWAYS, mids=np.full((4, 4), 30),
                  walk=np.full(4, 120.0), walk_save=walk)
    m = evaluate(exp, [[0, 1, 2, 3]])  # inter-town (94 stops): few riders, many walkers -> stops everywhere
    assert m["category"] == ["intertown"] and m["express"] == [[False, False, False]], m
    assert m["walk"] == 6 * (70 + 20 + 20 + 70) / 12  # trip ends per stop x minutes walked there, per trip
    exp.walk_weight = 2.0  # the same minutes reported, but each walking minute now costs two riding minutes
    m2 = evaluate(exp, [[0, 1, 2, 3]])
    assert m2["att"] == m["att"] and abs(m2["score"] - m["score"] - m["walk"]) < 1e-9, (m2["score"], m["score"])
    exp.walk_weight = 1.0
    exp.od = od.copy()
    exp.od[0, 3] = exp.od[3, 0] = 101
    assert evaluate(exp, [[0, 1, 2, 3]])["express"] == [[False, True, False]]  # 200 riders 0<->3 beat the walkers at 1, 2
    exp.od = od
    m = evaluate(exp, [[0, 1, 2, 3], [1, 2]])  # a local bus (32 stops) on the middle link still stops there
    assert m["category"] == ["intertown", "local"] and m["express"] == [[False, True, False], [False]], m
    assert m["route_min"][0] == 3 * (2 + DWELL_MIN * 30) - DWELL_MIN * 30  # skips the dwell on the middle link only
    print("ok", {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
