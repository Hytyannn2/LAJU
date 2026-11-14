"""Calibration (plan.md T-A2). The rider model has one fitted number, BUS_ASC: the felt minutes the bus loses to a
motorbike before any minute counts. It is fitted so that BAS.MY's 16 routes at their official timetable carry the
boardings BAS.MY reports. Any change to the stop graph or the trip model moves it, so it is refitted, not remembered:
road stops (2026-11-08) left the 2026-10-31 value giving 1,936 boardings a day instead of 2,800.

In the logit's tail the bus's share is ~exp(-THETA x asc) x the rest (plan.md 6.3): asc scales every network's riders
alike, so it sets the level of ridership and never which design is better.

    python calib.py   -> refits on today's graph, writes cache/calibration.json (kelantan.build uses it), checks it:
                         "ok asc=... boardings=..."
"""
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

import sim

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "cache" / "calibration.json"  # read by kelantan.build; delete it to fall back on kelantan.BUS_ASC
TARGET = 2800.0  # BAS.MY Kota Bharu's boardings a day: 84,000 a month (Q4 2025)


def fit(p, rs, heads, target=TARGET, lo=100.0, hi=600.0, steps=25):
    """asc for which route set rs at its fixed headways (a timetable) carries `target` boardings a day on problem p:
    bisection (boardings fall as asc rises). p.asc is left as it was. -> (asc, its metrics)"""
    was = p.asc
    try:
        def board(a):
            p.asc = a
            return sim.evaluate(p, rs, fixed=heads)["boardings"]

        top, bottom = board(lo), board(hi)
        print(f"boardings at asc {lo:g}: {top:,.0f}; at {hi:g}: {bottom:,.0f}", flush=True)
        if not bottom <= target <= top:
            raise ValueError(f"{target:,.0f} boardings lie outside asc {lo:g}-{hi:g}: widen the bracket")
        for _ in range(steps):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if board(mid) > target else (lo, mid)
        a = (lo + hi) / 2
        p.asc = a
        return a, sim.evaluate(p, rs, fixed=heads)
    finally:
        p.asc = was


def asc(target=TARGET):
    """Refit BUS_ASC on today's graph from the saved run "BAS.MY today" (python exam.py --basmy), and write it with
    what BAS.MY's timetable then gives (riders, vehicle-hours and bus-km a day: the equal-resources budgets, plan.md
    T-A3) to CALIBRATION. -> asc"""
    import exam
    import server

    p, net, rs, heads, _, _, _ = exam.load("BAS.MY today")
    a, m = fit(p, rs, heads, target)
    d = {"asc": a, "target": target, "boardings": m["boardings"], "riders": m["riders"],
         "vh_day": m.get("vh_day", float(np.dot(m["fleet_k"], p.period_h))), "bus_km_day": m["bus_km_day"],
         "key_stops": len(p.od), "score_version": sim.SCORE_VERSION, "date": date.today().isoformat()}
    server._save(CALIBRATION, d)
    return a


if __name__ == "__main__":
    t0 = time.time()
    a = asc()
    d = json.loads(CALIBRATION.read_text())
    assert abs(d["boardings"] / TARGET - 1) < 0.02, d  # within 2% of what BAS.MY reports
    print(f"ok asc={a:.2f} boardings={d['boardings']:,.0f} riders={d['riders']:,.0f} vehicle-hours={d['vh_day']:,.0f} "
          f"bus-km={d['bus_km_day']:,.0f} key stops={d['key_stops']} ({time.time() - t0:.0f} s)")
