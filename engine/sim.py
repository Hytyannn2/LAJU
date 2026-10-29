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

Score v3 by period of the day (brain v2 Phase 3, Problem.period_h): each route's buses per period follow its riders then,
at clock-face headways (CLOCK) so buses meet; a change at a hub waits TIMED_MIN; a route that does not end where a bus
can wait drives empty to the nearest such place and back (Problem.rest_min).
"""
import bisect
import heapq
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
# Score v3 (brain v2 Phase 2, 2026-10-29), when a Problem has `drive`: riders choose, and the score counts them.
WAIT_WEIGHT = 2.0   # a waiting minute feels like 2 riding minutes (heat, not knowing when the bus comes; studies 1.5-2.5)
THETA = 0.015       # logit: each felt minute the bus loses to the motorbike divides its odds by e^0.015. Halving the headway
                    # of an hourly bus then brings ~55% more riders, of a half-hourly one ~25%: the 0.4 (short run) to 0.7
                    # (long run) frequency elasticity transport studies report
LIFT_MIN = 120      # people without a vehicle: a lift, the walk or not going feels this much worse than driving would, so
                    # they take the same bus ~6x as often as people with one (e^(0.015 x 120); at 30 it was 1.6x)
SCORE_SCALE = 1000  # score = SCORE_SCALE / (riders + cover_weight x people near a served stop): lower is better
# Brain v2 Phase 3 (2026-10-29): buses by period of the day, clock-face headways, timed transfers, route ends.
CLOCK = (5, 10, 15, 20, 30, 60, 120)  # the headways a route runs (min): each divides an hour or two, so buses can meet
TIMED_MIN = 5.0   # a change at a hub waits this long, not half the headway: its buses arrive and leave together
HUB_ROUTES = 3    # a stop where this many routes meet is a hub; at a bus or rail station (Problem.hubs) two are enough
PEAK_HOUR = 1.3   # a period's busiest hour carries 1.3x its average hour: no bus may be over-full in it
REST_END = 0.1   # each route end too far from a rest place (Problem.rest_end) multiplies the score by 1.1: routes end only
                 # where a bus can wait (planner, 2026-10-28). The empty run there alone (rest_min) left 4 of 16 GA routes
                 # 15-26 min there and back from one (2026-10-29)
E_RANGE_KM = 250  # an electric bus on one charge (12 m e-buses: 200-300 km)
E_CHARGE_KM_MIN = 2.0  # range a minute of charging at a layover adds (~120 kW at 1 kWh/km)
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
    drive: np.ndarray | None = None       # (n, n) felt minutes by motorbike or car, door to door: set = score v3, od = all trips
    od_captive: np.ndarray | None = None  # (n, n) the part of od made by people without a vehicle
    asc: float = 0.0                      # felt minutes the bus loses to a motorbike before any minute counts (calibrated)
    budget: float = 0.0                   # buses the routes share (score v3)
    cover_cells: object = None            # sparse (cells, n + links): the key stops and links (cover_links) that cover each
    cover_pop: np.ndarray | None = None   # group of homes (a cell: homes with the same stops near), and its people
    cover_links: np.ndarray | None = None  # (links, 2) key-stop pairs; a link covers while a bus stops on it
    cover_weight: float = 0.0             # a person near a served stop counts as this many daily riders
    period_h: tuple = ()                  # hours of each period of the day (score v3); () = one period, service_hours long
    od_k: np.ndarray | None = None        # (periods, n, n) od by period (od is their sum)
    captive_k: np.ndarray | None = None   # (periods, n, n) od_captive by period
    hubs: np.ndarray | None = None        # (n,) bool: at a bus or rail station, where two routes meeting time their changes
    rest_min: np.ndarray | None = None    # (n,) minutes from each stop to the nearest place a route may end, and back
    rest_end: np.ndarray | None = None    # (n,) bool: near enough such a place for a route to end here
    depot_km: np.ndarray | None = None    # (n,) km from the nearest depot to each stop, and back


def quickest(p):
    """All-pairs quickest paths on the stop graph: (minutes, predecessors)."""
    return dijkstra(csr_matrix(np.where(np.isfinite(p.time), p.time, 0)), return_predecessors=True)


def _clock(h):
    """The longest clock-face headway (CLOCK) not over h minutes (the shortest one if h is shorter still)."""
    return CLOCK[max(0, bisect.bisect_right(CLOCK, h + 1e-9) - 1)]


def electric(km, cycle, hours):
    """Can an electric bus run a route of `km` one way all day (`hours` of round trips of `cycle` minutes, back to back)
    on its range (E_RANGE_KM) plus what it charges in each layover (LAYOVER_MIN)?"""
    return hours * 60 / cycle * (2 * km - E_CHARGE_KM_MIN * LAYOVER_MIN) <= E_RANGE_KM


def _assign(p, routes, heads, wait=1.0, hubs=()):
    """Quickest paths over the routes: every boarding waits `wait` x half its route's headway and pays the transfer
    penalty, taken off again for the first boarding; a change at one of the `hubs` waits at most TIMED_MIN (timed
    transfers). A route whose headway is infinite does not run. -> (minutes between stops (n, n), predecessors over the
    nodes: stops 0..n-1, one per (route, stop) from n, then each hub's change node, where its buses let riders off)"""
    n = len(p.od)
    hub = {s: n + sum(len(r) for r in routes) + i for i, s in enumerate(hubs)}
    rows, cols, w, base = [], [], [], n
    for r, h in zip(routes, heads):
        board = p.transfer_min + wait * h / 2  # every boarding pays this; the first boarding's penalty is removed below
        for k, s in enumerate(r if np.isfinite(h) else ()):
            rows += [s, base + k]
            cols += [base + k, hub.get(s, s)]
            w += [board, 1e-9]  # board, alight (csgraph drops exact zeros)
            if s in hub:  # board after a change here
                rows.append(hub[s])
                cols.append(base + k)
                w.append(p.transfer_min + wait * min(TIMED_MIN, h / 2))
            if k:
                turn = UTURN_MIN if k >= 2 and (r[k - 2], r[k - 1], s) in p.uturn else 0
                rows += [base + k - 1, base + k]
                cols += [base + k, base + k - 1]
                w += [p.time[r[k - 1], s] + turn, p.time[s, r[k - 1]] + turn]
        base += len(r)
    rows += list(hub.values())  # off the bus at a hub: arrived there
    cols += list(hub)
    w += [1e-9] * len(hub)
    G = csr_matrix((w, (rows, cols)), shape=(base + len(hub),) * 2)
    D, P = dijkstra(G, indices=np.arange(n), return_predecessors=True)
    return D[:, :n] - p.transfer_min, P


def evaluate(p, routes, detail=False, fixed=None):
    """Metrics for a route set (lists of stop indices). detail=True adds transfers and per-segment bus loads.
    fixed: each route's own headway (a real timetable) instead of the frequencies score v3 sets: one for the whole day
    or one per period (Problem.period_h; inf = no bus then)."""
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
    if p.drive is not None:
        return _v3(p, routes, cats, heads, route_km, fixed)
    T, P = _assign(p, routes, heads)
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
    m["score"] = s * _shape(p, routes, m)
    return m


def _shape(p, routes, m):
    """The score's multiplier for route km over budget, routes over the time cap, crooked minutes and route ends away
    from a rest place."""
    x = 1.0
    if p.rest_end is not None:
        x *= 1 + REST_END * sum((not p.rest_end[r[0]]) + (not p.rest_end[r[-1]]) for r in routes)
    if p.max_km and m["total_km"] > p.max_km:
        x *= 1 + (m["total_km"] - p.max_km) / p.max_km
    if p.max_route_min:
        x *= 1 + sum(max(0.0, t - p.max_route_min) for t in m["route_min"] + m["route_min_back"]) / p.max_route_min
    if p.max_detour:  # crooked routes: every minute beyond max_detour x the quickest drive between the route's two ends
        if "_quick" not in p.__dict__:  # once per problem: its stop graph never changes
            p._quick = quickest(p)[0]
        over = sum(max(0.0, t - p.max_detour * p._quick[r[0], r[-1]]) + max(0.0, b - p.max_detour * p._quick[r[-1], r[0]])
                   for r, t, b in zip(routes, m["route_min"], m["route_min_back"]))
        x *= 1 + over / CROOKED_MIN  # per crooked minute, however many routes (2026-10-18; averaged over the routes before,
        # which made a crooked route nearly free in a 60-route network). The first rule (1/60 per minute, summed) outweighed
        # RM 50 per unserved trip, and the GA and RL dropped riders to avoid it (2026-10-10)
    return x


def _choose(p, T, reach, wz):
    """The chance that a trip between key stops takes the bus, for people with a motorbike or car and for people without
    one: the logit (THETA) of its felt minutes (T, plus both walks x walk_weight, plus asc) against driving (p.drive),
    or against a lift (LIFT_MIN). -> (own, lift), each (n, n)"""
    gap = np.where(reach, T + p.walk_weight * (wz[:, None] + wz[None]) + p.asc - p.drive, np.inf)
    return tuple(1 / (1 + np.exp(np.minimum(THETA * g, 700))) for g in (gap, gap - LIFT_MIN))


def _share(budget, cycle, rate, floor):
    """Clock-face headways (CLOCK) per route and period of the day for a budget of buses. Each route starts at the
    longest one within its floor (floor (routes, periods): the longest headway allowed, the minimum service or less so
    that no bus is over-full); then the step to the next shorter headway that saves most waiting per extra bus goes first
    (rate (routes, periods): boardings an hour, x the minutes of half-headway saved: the square-root rule, Newell 1971,
    in whole buses and clock-face steps), while the period's buses fit the budget. One price of a bus all day: a period
    that never runs short of buses buys only the steps worth as much as the last one the budget allowed in a period that
    did, so quiet hours run fewer buses ("jimat", the planner, 2026-10-28). cycle (routes,): minutes a round trip takes.
    -> (headways (routes, periods), buses (routes, periods), buses the floors need beyond the budget). ponytail: with buses
    to spare all day every step that saves any waiting runs; a cost per bus-hour would keep the night lean then."""
    nr, nk = rate.shape
    start = np.maximum(np.searchsorted(CLOCK, floor + 1e-9, side="right") - 1, 0)
    need = lambda r, i: max(1, math.ceil(cycle[r] / CLOCK[i] - 1e-9))

    def run(k, price):
        i, heap = start[:, k].copy(), []
        b = [need(r, i[r]) for r in range(nr)]
        used, last, short = sum(b), 0.0, False

        def push(r):
            if i[r] > 0:
                db = need(r, i[r] - 1) - b[r]
                heapq.heappush(heap, (-rate[r, k] * (CLOCK[i[r]] - CLOCK[i[r] - 1]) / 2 / max(db, 1e-9), r, db))

        for r in range(nr):
            push(r)
        while heap:
            v, r, db = heapq.heappop(heap)
            if -v <= 0 or -v < price:
                break
            if used + db > budget:
                short = True  # this step does not fit: the route keeps its headway
                continue
            i[r], b[r], used = i[r] - 1, b[r] + db, used + db
            last = -v if db else last  # a step that needs no extra bus sets no price
            push(r)
        return i, b, used, last if short else 0.0

    out = [run(k, 0.0) for k in range(nk)]
    price = max((x[3] for x in out), default=0.0)
    out = [x if x[3] == price else run(k, price) for k, x in enumerate(out)]
    floors = max((sum(need(r, start[r, k]) for r in range(nr)) for k in range(nk)), default=0)
    return (np.array([[CLOCK[j] for j in x[0]] for x in out], float).reshape(nk, nr).T,
            np.array([x[1] for x in out], int).reshape(nk, nr).T, max(0, floors - int(budget)))


def _v3(p, routes, cats, floor_h, route_km, fixed=None):
    """Score v3 (brain v2 Phase 2, 2026-10-29; by period of the day since Phase 3): riders choose, and the routes share a
    budget of buses. Every trip compares the bus (waits x WAIT_WEIGHT, walks x walk_weight, transfer_min per change, asc)
    with the motorbike or car, or a lift for people without one (_choose); a trip needing more than max_transfers changes
    takes no bus. Two rounds: every period's trips on the paths at the minimum service, which set the express links and
    each route's riders and loads per period; then each period at the clock-face headways the budget buys (_share), or at
    the fixed timetable. A change at a hub waits at most TIMED_MIN (a fixed timetable does not time them); a route end
    away from a rest place costs the empty run there and back (p.rest_min) every round trip. Score = SCORE_SCALE / (riders
    + cover_weight x people near a served stop), times the shape penalties (_shape) and the buses the floors need beyond
    the budget. ponytail: riders the second round's frequencies win do not change the shares again; the express pattern
    is decided in the first round; a hub times every change there, where a real timetable can time a route at about two."""
    n, nr = len(p.od), len(routes)
    hours = np.array(p.period_h or (p.service_hours,), float)
    od_k = p.od[None] if p.od_k is None else p.od_k
    cap_k = p.captive_k if p.captive_k is not None else None if p.od_captive is None else p.od_captive[None]
    route_min = np.array([sum(p.time[a, b] for a, b in zip(r, r[1:])) + _turns(p, r) for r in routes], float)
    route_back = np.array([sum(p.time[b, a] for a, b in zip(r, r[1:])) + _turns(p, r) for r in routes], float)
    dead = np.array([0.0 if p.rest_min is None else p.rest_min[r[0]] + p.rest_min[r[-1]] for r in routes])
    meet = Counter(s for r in routes for s in r)
    hubs = [] if fixed is not None else sorted(int(s) for s, c in meet.items()
                                               if c >= HUB_ROUTES or c >= 2 and p.hubs is not None and p.hubs[s])
    express = [[False] * (len(r) - 1) for r in routes]

    paths_at = {}  # headways -> quickest paths: round 2 often runs the minimum service of round 1

    def ride(heads, ks):
        """Periods ks at the same headways: (T, reach, walk per trip end, bus trips (len(ks), n, n), paths, trips that
        chose the bus before the limit on changes, links where a bus stops between the key stops)."""
        if tuple(heads) not in paths_at:
            paths_at[tuple(heads)] = _assign(p, routes, heads, WAIT_WEIGHT, hubs)
        T, P = paths_at[tuple(heads)]
        stopping = np.zeros((n, n), bool)
        for r, e in zip(routes, express):
            for (a, b), x in zip(zip(r, r[1:]), e):
                stopping[a, b] = stopping[b, a] = stopping[a, b] or not x
        wz = np.zeros(n) if p.walk is None else p.walk - (p.walk_save * stopping).sum(1)  # minutes per trip end
        reach = np.isfinite(T)
        own, lift = _choose(p, T, reach, wz)
        cap = 0.0 if cap_k is None else cap_k[ks]
        R = (od_k[ks] - cap) * own + cap * lift
        chose = float(R.sum())
        paths = _transfers_and_loads(p, routes, P, reach, chose or 1.0, R)  # drops trips over max_transfers
        return T, reach, wz, np.where(reach, R, 0.0), paths, chose, stopping

    if fixed is None:  # round 1: the minimum service
        _, _, _, R, paths, _, _ = ride([_clock(h) for h in floor_h], range(len(hours)))
        if p.mids is not None:
            express = _express(p, routes, cats, paths["riders"], R.sum((0, 1)) + R.sum((0, 2)))
            skip = np.array([sum(DWELL_MIN * p.mids[a, b] for (a, b), x in zip(zip(r, r[1:]), e) if x)
                             for r, e in zip(routes, express)])
            route_min, route_back = route_min - skip, route_back - skip
    cycle = route_min + route_back + LAYOVER_MIN + dead  # minutes a round trip takes a bus
    if fixed is None:
        peak = np.array([x.max(0) for x in paths["loads"]]).reshape(nr, len(hours)) / hours * PEAK_HOUR  # busiest hour
        seats = np.array([CATEGORIES[c][2] for c in cats], float)[:, None]
        floor = np.minimum(np.array(floor_h, float)[:, None], 60 * seats / np.maximum(peak, 1e-9))
        heads, buses, over = _share(p.budget, cycle, np.array(paths["boardings"]).reshape(nr, len(hours)) / hours, floor)
    else:
        heads = np.broadcast_to(np.array(fixed, float).reshape(nr, -1), (nr, len(hours)))
        buses, over = np.where(np.isfinite(heads), np.maximum(1, np.ceil(cycle[:, None] / heads - 1e-9)), 0).astype(int), 0

    groups = {}  # round 2: each period at its headways; periods with the same ones share their paths
    for k in range(len(hours)):
        groups.setdefault(tuple(heads[:, k]), []).append(k)
    board, loads = np.zeros((nr, len(hours))), [np.zeros((len(r) - 1, len(hours))) for r in routes]
    riders = chose = felt = walked = served = 0.0
    by_changes = np.zeros(3)
    for hk, ks in groups.items():
        T, reach, wz, R, paths, c, stopping = ride(list(hk), ks)
        riders, chose = riders + float(R.sum()), chose + c
        by_changes += c * np.array([paths["d0"], paths["d1"], paths["d2"]])
        board[:, ks] = np.array(paths["boardings"]).reshape(nr, len(ks))
        for x, y in zip(loads, paths["loads"]):
            x[:, ks] = y
        felt += float((R * np.where(reach, T + wz[:, None] + wz[None], 0.0)).sum())
        walked += float((R * (wz[:, None] + wz[None])).sum())
        served += float(sum(od_k[k][reach].sum() for k in ks))
    on = np.zeros(n, bool)
    on[[s for r in routes for s in r]] = True
    covered = 0.0 if p.cover_pop is None else float(p.cover_pop @ (p.cover_cells @ np.r_[
        on, stopping[p.cover_links[:, 0], p.cover_links[:, 1]]] > 0))  # people with any served stop near
    f = np.where(np.isfinite(heads), 60 / heads, 0.0)  # buses an hour each way
    most, fleet_k = buses.max(1, initial=0), buses.sum(0)
    m = {"riders": riders, "boardings": float(board.sum()), "covered": covered, "market": float(p.od.sum()),
         "unserved": float(1 - served / p.od.sum()) if p.od.sum() else 0.0,
         "route_min": route_min.tolist(), "route_min_back": route_back.tolist(), "route_km": route_km,
         "total_km": sum(route_km), "att": felt / riders if riders else 0.0, "walk": walked / riders if riders else 0.0,
         **{f"d{k}": float(by_changes[k] / chose) if chose else 0.0 for k in range(3)},
         "dun": float(1 - by_changes.sum() / chose) if chose else 0.0, "loads": [x.sum(1).round(1).tolist() for x in loads],
         "peak_hour_load": [round(float((x / hours).max() * PEAK_HOUR)) for x in loads],
         "category": [CATEGORIES[c][0] for c in cats], "express": express,
         "periods": [[round(float(h), 1) if np.isfinite(h) else None for h in row] for row in heads],  # per route, period
         "headway": [round(float(row[np.isfinite(row)].min()), 1) for row in heads],  # its busiest period's
         "buses": [int(x) for x in most], "fleet_k": [int(x) for x in fleet_k], "fleet": int(fleet_k.max(initial=0)),
         "budget": p.budget, "hubs": hubs, "bus_km_day": float((2 * np.array(route_km, float)[:, None] * hours * f).sum()),
         "empty_km_day": 0.0 if p.depot_km is None else float(sum(b * min(p.depot_km[r[0]], p.depot_km[r[-1]])
                                                                  for r, b in zip(routes, most))),  # from depots and back
         "electric": [bool(electric(k, c, hours.sum())) for k, c in zip(route_km, cycle)]}
    s = SCORE_SCALE / max(riders + p.cover_weight * covered, 1e-9)
    if over:
        s *= 1 + over / p.budget
    m["score"] = s * _shape(p, routes, m)
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


def _walk(P, o, d, n, m):
    """Walk the quickest paths o -> d back at once. -> (each path's boardings, and for the route edges v->v+1 (fwd) and
    v+1->v (bwd), stored at v, and the route nodes boarded at (on): (nodes, paths) of every use). Route nodes are
    n .. n + m - 1 (_assign): below them the stops, above them the hubs' change nodes."""
    cur, boards = d.copy(), np.zeros(len(d), int)
    ev = [([np.zeros(0, int)], [np.zeros(0, int)]) for _ in range(3)]  # (nodes, paths) of fwd, bwd, on
    live = cur != o
    while live.any():
        i = np.flatnonzero(live)
        prv, c = P[o[i], cur[i]], cur[i]
        rp, rc = (prv >= n) & (prv < n + m), (c >= n) & (c < n + m)
        up, down, b = rp & rc & (c == prv + 1), rp & rc & (c == prv - 1), rc & ~rp  # b: onto a bus from a stop / change node
        boards[i] += b
        for (at, of), x, v in zip(ev, (up, down, b), (prv, c, c)):
            at.append(v[x])
            of.append(i[x])
        cur[i] = prv
        live[i] = prv != o[i]
    return boards, *((np.concatenate(at), np.concatenate(of)) for at, of in ev)


def _put(use, flow, size):
    """The flow ((paths,) or (paths, periods)) of the paths onto the nodes they use (_walk): (size,) or (size, periods)."""
    at, of = use
    return (np.bincount(at, flow[of], minlength=size) if flow.ndim == 1 else
            np.stack([np.bincount(at, flow[of, k], minlength=size) for k in range(flow.shape[1])], 1))


def _transfers_and_loads(p, routes, P, reach, total, trips=None):
    """Walk every used shortest path back (vectorised over all OD pairs): count boardings, add up segment loads of
    `trips` (default p.od; (periods, n, n): loads and boardings per period, all on the same paths). Trips needing more
    than p.max_transfers transfers are taken off `reach` (in place): riders would not make them."""
    n, m, trips = len(p.od), sum(len(r) for r in routes), p.od if trips is None else trips
    o, d = np.nonzero((trips > 0).reshape(-1, n, n).any(0) & reach)
    flow = trips[..., o, d].T  # (pairs,) or (pairs, periods)
    boards, fwd, bwd, on = _walk(P, o, d, n, m)
    if p.max_transfers is not None and (far := boards > p.max_transfers + 1).any():
        reach[o[far], d[far]] = False
        flow = flow * ~far.reshape((-1,) + (1,) * (flow.ndim - 1))  # nobody rides them
    fwd, bwd, on = (_put(x, flow, P.shape[1]) for x in (fwd, bwd, on))
    shares = [float(flow[boards == k + 1].sum() / total) if total else 0.0 for k in range(3)]
    loads, riders, boarding, base = [], [], [], n
    for r in routes:
        seg = slice(base, base + len(r) - 1)
        ld = np.maximum(fwd[seg], bwd[seg])
        loads.append(ld.round(1).tolist() if ld.ndim == 1 else ld)
        riders.append((fwd[seg] + bwd[seg]).reshape(len(r) - 1, -1).sum(1))  # both directions, per day
        boarding.append(float(on[base: base + len(r)].sum()) if on.ndim == 1 else on[base: base + len(r)].sum(0))  # a day
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
    # score v3: riders choose the bus over a 10-min drive; more buses, more riders; covered people count
    cells = csr_matrix(np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]))  # homes near stop 0 / 3 / link 1-2
    v3 = Problem(time=line, od=od * 100, xy=p.xy, km=km8, headways=HEADWAYS, drive=np.full((4, 4), 10.0), budget=4,
                 cover_cells=cells, cover_pop=np.array([100.0, 100, 50]), cover_links=np.array([[1, 2]]), cover_weight=0.1)
    a = evaluate(v3, [[0, 1, 2, 3]])
    v3.budget = 40
    b = evaluate(v3, [[0, 1, 2, 3]])
    assert b["riders"] > a["riders"] > 0 and b["score"] < a["score"] and b["headway"][0] < a["headway"][0], (a, b)
    assert a["covered"] == 250 and evaluate(v3, [[1, 2]])["covered"] == 50, a["covered"]
    assert abs(a["score"] - SCORE_SCALE / (a["riders"] + 0.1 * 250)) < 1e-9
    v3.od_captive = v3.od / 2  # half the trips by people without a vehicle: the bus wins more of them
    assert evaluate(v3, [[0, 1, 2, 3]])["riders"] > b["riders"]
    assert evaluate(v3, [[0, 1, 2, 3]], fixed=[60])["headway"] == [60]  # a real timetable keeps its headway
    # Phase 3. Clock-face: a 22-min round trip runs every 10 min on 3 buses; a 4th bus would not shorten it, a 5th would
    assert (a["fleet"], a["headway"], b["headway"]) == (3, [10], [5]), (a["fleet"], a["headway"], b["headway"])
    T, _ = _assign(p, [[0, 1], [1, 2]], [60, 60])
    Th, _ = _assign(p, [[0, 1], [1, 2]], [60, 60], hubs=[1])  # a timed change at stop 1 waits TIMED_MIN, not 30 min
    assert abs(T[0, 2] - (30 + 2 + 5 + 30 + 2)) < 1e-6 and abs(Th[0, 2] - (30 + 2 + 5 + TIMED_MIN + 2)) < 1e-6, Th[0, 2]
    assert abs(Th[0, 1] - 32) < 1e-6 and abs(Th[1, 2] - 32) < 1e-6  # ending or starting at the hub: no change, no timing
    day = Problem(time=line, od=od * 100, xy=p.xy, km=km8, headways=HEADWAYS, drive=np.full((4, 4), 10.0), budget=4,
                  period_h=(3, 15), od_k=np.stack([od * 90, od * 10]))  # 90% of the trips in a 3-hour peak
    m = evaluate(day, [[0, 1, 2, 3]])  # the peak runs out of buses; the quiet hours buy none worth as much ("jimat")
    assert m["periods"] == [[10, 30]] and m["fleet_k"] == [3, 1] and m["fleet"] == 3, m["periods"]
    assert m["bus_km_day"] == 2 * 24 * (3 * 6 + 15 * 2) and m["electric"] == [False]  # 2,304 bus-km on 3 buses
    day.rest_min = np.array([0, 0, 0, 20.0])  # stop 3 is 10 min from a place to wait: 20 min more per round trip
    assert evaluate(day, [[0, 1, 2, 3]])["periods"] == [[15, 30]]
    day.rest_min, day.hubs = None, np.array([False, True, False, False])  # a bus station at stop 1
    two = evaluate(day, [[0, 1], [1, 2, 3]])
    same = [[h or np.inf for h in row] for row in two["periods"]]  # the same headways, changes not timed
    assert two["hubs"] == [1] and two["riders"] > evaluate(day, [[0, 1], [1, 2, 3]], fixed=same)["riders"]
    m = evaluate(day, [[0, 1, 2, 3]], fixed=[[60, np.inf]])  # a timetable with no bus after the peak
    assert m["periods"] == [[60, None]] and m["fleet_k"] == [1, 0] and m["hubs"] == []
    assert m["riders"] < evaluate(day, [[0, 1, 2, 3]], fixed=[[60, 60]])["riders"]
    assert electric(20, 100, 18) and not electric(60, 180, 18)
    one = evaluate(day, [[0, 1]])["score"]
    day.rest_end = np.array([True, False, True, True])  # no place to wait at stop 1: a route ending there costs 10%
    assert abs(evaluate(day, [[0, 1]])["score"] - (1 + REST_END) * one) < 1e-9
    print("ok", {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, "v3 riders", round(a["riders"]), round(b["riders"]))
