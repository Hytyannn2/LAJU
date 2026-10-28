"""Phase 4/5 benchmarks: baseline vs GA vs deep RL vs RL+GA on Mandl's Swiss network and on Kota Bharu.

Mandl's Swiss network (15 stops, 21 links, 15,570 trips) is the standard transit-route-design benchmark.

Data: C.E. Mandl, Applied Network Optimization (1979), as distributed in the OR-Library / Mumford (2013) benchmark sets.
Setting used by recent papers (Mumford 2013; Holliday & Dudek 2024): 6 routes of 2-8 stops, 5-min transfer penalty,
no waiting time. Best published passenger cost (ATT) for that setting: 10.18 min (Ahmed, Mumford & Kheiri 2019).

    python bench.py [ga_seconds]   -> Mandl table (with published route sets), then a Kota Bharu table
"""
import sys
import time

import numpy as np

import routes
from sim import Problem, evaluate

LINKS = "1-2:8 2-3:2 2-4:3 2-5:6 3-6:3 4-5:4 4-6:4 4-12:10 6-8:2 6-15:3 7-10:7 7-15:2 8-10:8 8-15:2 9-15:8 10-11:5 10-13:10 10-14:8 11-12:10 11-13:5 13-14:2"
DEMAND = """
0 400 200 60 80 150 75 75 30 160 30 25 35 0 0
400 0 50 120 20 180 90 90 15 130 20 10 10 5 0
200 50 0 40 60 180 90 90 15 45 20 10 10 5 0
60 120 40 0 50 100 50 50 15 240 40 25 10 5 0
80 20 60 50 0 50 25 25 10 120 20 15 5 0 0
150 180 180 100 50 0 100 100 30 880 60 15 15 10 0
75 90 90 50 25 100 0 50 15 440 35 10 10 5 0
75 90 90 50 25 100 50 0 15 440 35 10 10 5 0
30 15 15 15 10 30 15 15 0 140 20 5 0 0 0
160 130 45 240 120 880 440 440 140 0 600 250 500 200 0
30 20 20 40 20 60 35 35 20 600 0 75 95 15 0
25 10 10 25 15 15 10 10 5 250 75 0 70 0 0
35 10 10 10 5 15 10 10 0 500 95 70 0 45 0
0 5 5 5 0 10 5 5 0 200 15 0 45 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
"""
# Node layout drawn by Arbex & Cunha (2015); only used as RL input features.
XY = [(-46.449, -25.875), (-46.350, -25.974), (-46.217, -25.977), (-46.349, -26.084), (-46.507, -26.084),
      (-46.218, -26.086), (-45.884, -26.218), (-46.100, -26.219), (-45.837, -26.085), (-45.978, -26.376),
      (-46.045, -26.461), (-46.386, -26.332), (-45.936, -26.504), (-45.855, -26.439), (-45.987, -26.085)]
# Published 6-route solutions (1-based stops), as collected in the Mandl benchmark set of Arbex & Cunha.
PUBLISHED = {
    "Mumford (2013) best passenger": "1-2-3-6-15-7-10-11 12-11-13-14-10-7-15-9 1-2-5-4-6-8-10-11 1-2-3-6-8-10-13-11 1-2-4-12-11-10-14-13 1-2-5-4-6-8-15-7",
    "Kechagiopoulos (2014)": "1-2-3-6-8-10-11-13 9-15-7-10-11-13 14-10-7-15-6-3-2-1 10-11-12-4-6-8-15-7 1-2-5-4-6-8-10-11 14-13-11-12-4-2-1",
    "Buba & Lee (2018)": "5-4-6-3-2-1 12-11-13-14-10-7-15-9 5-4-6-8-10-11-12 1-2-4-12-11-10-14-13 1-2-3-6-8-10-13-11 1-2-3-6-15-7-10-8",
}


def problem():
    t = np.full((15, 15), np.inf)
    for link in LINKS.split():
        ab, m = link.split(":")
        a, b = (int(x) - 1 for x in ab.split("-"))
        t[a, b] = t[b, a] = float(m)
    od = np.array([row.split() for row in DEMAND.strip().splitlines()], float)
    return Problem(time=t, od=od, xy=np.array(XY), min_len=2, max_len=8, transfer_min=5.0)


def parse(s):
    return [[int(x) - 1 for x in r.split("-")] for r in s.split()]


def row(name, p, rs, secs=None):
    m = evaluate(p, rs, detail=True)
    pct = " ".join(f"{100 * m[k]:6.2f}" for k in ("d0", "d1", "d2", "dun"))
    ok = "" if routes.valid(p, rs) else "  (breaks the stops-per-route rule)"
    took = f"{secs:6.0f}s" if secs is not None else "       "
    print(f"{name:34s} {m['att']:6.2f} {pct} {sum(m['route_min']):6.0f} {m['score']:7.2f} {took}{ok}", flush=True)
    return m


def compare(p, n_routes, secs):
    """Our four designers on one problem; RL needs a trained policy (python rl.py)."""
    row("LAJU baseline", p, routes.baseline(p, n_routes))
    t = time.time()
    row("LAJU GA", p, routes.ga(p, n_routes, seconds=secs), time.time() - t)
    try:
        import rl

        t = time.time()
        row("LAJU RL (GNN, best of 32)", p, rl.design(p, n_routes), time.time() - t)
        t = time.time()
        row("LAJU RL, learning this city", p, rl.design(p, n_routes, seconds=secs), time.time() - t)
        t = time.time()
        row("LAJU RL+GA", p, routes.ga(p, n_routes, seconds=secs, seeds=rl.samples(p, n_routes, 8)), time.time() - t)
    except FileNotFoundError as e:
        print("RL skipped:", e)


if __name__ == "__main__":
    p = problem()
    assert p.od.sum() == 15570 and (p.od == p.od.T).all() and np.isfinite(p.time).sum() == 42
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    head = f"{'ATT':>6s} {'d0%':>6s} {'d1%':>6s} {'d2%':>6s} {'dun%':>6s} {'TRT':>6s} {'score':>7s}"
    print(f"{'Mandl, 6 routes (min)':34s} {head}")
    for name, s in PUBLISHED.items():
        row(name, p, parse(s))
    compare(p, 6, secs)

    import kelantan

    p, net = kelantan.build((6.0, 102.15, 6.2, 102.35), n_stops=40)
    print(f"\n{'Kota Bharu, 8 routes, 40 key stops':34s} {head}  (score = RM per trip)")
    compare(p, 8, secs)
