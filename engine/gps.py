"""BAS.MY Kota Bharu's buses, live (PLAN.txt section 9, "Eyes and experience"): every 30 s, the position each bus reports
(data.gov.my GTFS Realtime, updated every 30 s) goes into cache/gps/<date>.csv, one row per new report: real speeds by
hour and road, delays against the timetable, buses really in service. Standard library only, so it runs for weeks in
its own window; a sleeping laptop or a dropped connection leaves gaps. The feed carries only each bus's trip id, position,
bearing and time: route, direction, speed, stop and status are LAJU's estimates (--fill); occupancy nothing tells.

    python gps.py          -> logs until STOP_HOUR or its window is closed (restarting appends; a second copy exits)
    Start GPS logger.bat   -> the same in its own minimized window (the LAJU folder): by double-click, and every day at
                              06:00 by the Windows task "LAJU GPS logger"
    python gps.py --fill   -> cache/gps/filled/<date>.csv: every day's log with the empty columns worked out
    python gps.py --test   -> self-check: "ok ..."
"""
import csv
import datetime
import math
import socket
import struct
import sys
import time
import urllib.request
from pathlib import Path

FEED = "https://api.data.gov.my/gtfs-realtime/vehicle-position/mybas-kota-bharu"
OUT = Path(__file__).resolve().parent / "cache" / "gps"
EVERY_S = 30
STOP_HOUR = 23  # no bus reported after 22:03 on Sunday 2026-11-01; check against weekdays
LOCK_PORT = 47613  # one logger at a time: the one holding this local port
COLUMNS = ("bus", "plate", "trip", "route", "direction", "lat", "lng", "bearing", "speed_ms", "stop", "status", "occupancy",
           "time")
NEAR_M, SLOW_MS, GAP_S = 50, 3, 120  # --fill: standing at a stop = within 50 m, under 3 m/s; no speed over a 2-min gap


def _varint(b, i):
    x = s = 0
    while True:
        c = b[i]
        i, x, s = i + 1, x | (c & 0x7F) << s, s + 7
        if c < 0x80:
            return x, i


def _each(b):
    """(field number, value) of every field of one protobuf message, in order (wire format)."""
    i = 0
    while i < len(b):
        k, i = _varint(b, i)
        t = k & 7
        if t == 0:
            v, i = _varint(b, i)
        elif t == 1:
            v, i = b[i:i + 8], i + 8
        elif t == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif t == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"protobuf wire type {t}")
        yield k >> 3, v


def buses(feed):
    """Rows (COLUMNS) of a GTFS Realtime FeedMessage (bytes), one per vehicle position: entity (field 2) -> vehicle (4) ->
    trip (1: trip_id 1, route_id 5, direction_id 6), position (2: lat 1, lng 2, bearing 3, speed 5), stop_id (7),
    current_status (4), timestamp (5), occupancy_status (9), vehicle (8: id 1, license_plate 3)."""
    for f, e in _each(feed):
        if f != 2 or 4 not in (e := dict(_each(e))):
            continue
        v = dict(_each(e[4]))
        trip, pos, car = (dict(_each(v.get(k, b""))) for k in (1, 2, 8))
        text = lambda m, k: m.get(k, b"").decode()
        num = lambda k: round(struct.unpack("<f", pos[k])[0], 6) if k in pos else ""
        yield (text(car, 1), text(car, 3), text(trip, 1), text(trip, 5), trip.get(6, ""), num(1), num(2), num(3), num(5),
               text(v, 7), v.get(4, ""), v.get(9, ""),
               datetime.datetime.fromtimestamp(v[5]).isoformat(timespec="seconds") if 5 in v else "")


def _xy(lat, lng):  # metres on a flat Kelantan, as kelantan._m (not imported: the logger is standard library only)
    return lng * 110_800, lat * 110_570


def _off(p, a, b):
    """Metres from point p to the segment a-b (all _xy)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    f = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy or 1)))
    return math.dist(p, (a[0] + f * dx, a[1] + f * dy))


def timetable():
    """Trip id prefix ("249_1" of trip "249_1_WE_22") -> (route code, direction, its stops in order as (name, _xy)), from
    BAS.MY's official timetable as basmy_map.py reads it. The prefix, not the whole id: the timetable lists weekend trips
    only."""
    import basmy_map
    return {w["trip_prefix"]: (r["code"], w["direction"], [(n, _xy(a, b)) for n, a, b in w["stops"]])
            for r in basmy_map.routes() for w in r["ways"]}


def fill(rows, ways):
    """The columns BAS.MY leaves empty, worked out for rows (dicts of COLUMNS, each bus's in time order; ways: timetable()).
    route, direction: from the trip id. speed_ms: the straight distance from the bus's previous row over the time between
    (none over GAP_S). stop: the stop of its route it is at or heading to (its nearest stop-to-stop segment). status:
    STOPPED_AT (within NEAR_M of the stop, slower than SLOW_MS) or IN_TRANSIT_TO. No trip (not in service): only speed."""
    last = {}
    for r in rows:
        t, p = datetime.datetime.fromisoformat(r["time"]), _xy(float(r["lat"]), float(r["lng"]))
        q, last[r["bus"]] = last.get(r["bus"]), (t, p)
        s = (t - q[0]).total_seconds() if q else 0
        r["speed_ms"] = v = round(math.dist(p, q[1]) / s, 1) if 0 < s <= GAP_S else ""
        if way := ways.get("_".join(r["trip"].split("_")[:2])):
            r["route"], r["direction"], stops = way
            # ponytail: nearest segment only; where a route runs one road twice (into a town and back out) the stop can
            # flip between the two passes (0.4% of steps on 2026-11-01). Follow each trip's progress if that matters.
            k = min(range(len(stops) - 1), key=lambda k: _off(p, stops[k][1], stops[k + 1][1]))
            at = min(stops[k:k + 2], key=lambda x: math.dist(p, x[1]))
            r["status"], r["stop"] = (("STOPPED_AT", at[0]) if (v == "" or v < SLOW_MS) and math.dist(p, at[1]) <= NEAR_M
                                      else ("IN_TRANSIT_TO", stops[k + 1][0]))
        yield r


def until(now):
    """When a logger started at `now` stops for the night: the next STOP_HOUR."""
    stop = now.replace(hour=STOP_HOUR, minute=0, second=0, microsecond=0)
    return stop if now < stop else stop + datetime.timedelta(days=1)


def main():
    lock = socket.socket()
    try:
        lock.bind(("127.0.0.1", LOCK_PORT))
    except OSError:
        print("The GPS logger is already running (another window).")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    seen = {}  # bus -> time of its last report written
    stop = until(datetime.datetime.now())
    while datetime.datetime.now() < stop:
        t0 = time.time()
        try:
            with urllib.request.urlopen(FEED, timeout=20) as r:
                rows = [x for x in buses(r.read()) if seen.get(x[0]) != x[-1]]
            day = OUT / f"{datetime.date.today()}.csv"
            new = not day.exists()
            with open(day, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(COLUMNS)
                w.writerows(rows)
            seen.update((x[0], x[-1]) for x in rows)
            print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {len(rows)} new positions", flush=True)
        except Exception as err:  # no connection, a feed hiccup, OneDrive holding the file: try again next round
            print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  skipped: {err}", flush=True)
        time.sleep(max(1.0, EVERY_S - (time.time() - t0)))


if __name__ == "__main__":
    if sys.argv[1:] == ["--fill"]:
        (OUT / "filled").mkdir(exist_ok=True)
        tt = timetable()
        for day in sorted(OUT.glob("*.csv")):
            done = OUT / "filled" / day.name
            if done.exists() and done.stat().st_mtime > day.stat().st_mtime:
                continue  # filled since its last row (delete cache/gps/filled to redo all)
            with open(day, newline="") as f:
                rows = list(fill(csv.DictReader(f), tt))
            with open(done, "w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel reads it right
                out = csv.DictWriter(f, COLUMNS)
                out.writeheader()
                out.writerows(rows)
            print(f"{day.name}: {len(rows)} rows, {sum(bool(r['route']) for r in rows)} on a route ->", done)
        sys.exit()
    if sys.argv[1:] != ["--test"]:
        main()
        sys.exit()

    def vb(x):  # varint bytes
        out = b""
        while True:
            out += bytes([x & 0x7F | (0x80 if x > 0x7F else 0)])
            x >>= 7
            if not x:
                return out

    def msg(*kv):  # (field, value): int = varint, float = 32-bit, bytes = a message or a string
        return b"".join(bytes([k << 3]) + vb(v) if isinstance(v, int) else bytes([k << 3 | 5]) + struct.pack("<f", v)
                        if isinstance(v, float) else bytes([k << 3 | 2]) + vb(len(v)) + v for k, v in kv)

    car = lambda plate, lat, ts: msg((1, msg((1, b"249_1_WE_24"), (5, b"30425"), (6, 1))), (2, msg((1, lat), (2, 102.208))),
                                     (5, ts), (8, msg((1, plate), (3, plate))))
    feed = msg((1, msg((1, b"2.0"), (3, 1793278862))), (2, msg((1, b"a"), (4, car(b"DFG6155", 6.07903, 1793278859)))),
               (2, msg((1, b"b"), (4, car(b"DEJ7355", 6.12784, 1793278840)))))
    rows = list(buses(feed))
    assert [x[0] for x in rows] == ["DFG6155", "DEJ7355"] and rows[0][2:5] == ("249_1_WE_24", "30425", 1), rows
    assert abs(rows[0][5] - 6.07903) < 1e-5 and abs(rows[1][6] - 102.208) < 1e-4 and rows[0][-1].startswith("2026-10-")

    stops = [(n, _xy(6.0, 102.0 + i / 100)) for i, n in enumerate("ABC")]  # 1.1 km apart, west to east
    row = lambda bus, trip, lng, s: dict.fromkeys(COLUMNS, "") | {"bus": bus, "trip": trip, "lat": "6.0", "lng": str(lng),
                                                                  "time": f"2026-11-01T07:{s // 60:02d}:{s % 60:02d}"}
    log = [row("X", "249_1_WE_22", *x) for x in ((102.0, 0), (102.0015, 30), (102.0098, 60), (102.0098, 90),
                                                 (102.0102, 120), (102.012, 400))] + [row("Y", "", 102.0, 0)]
    got = [(r["route"], r["direction"], r["speed_ms"], r["status"], r["stop"]) for r in fill(log, {"249_1": ("D32", "1", stops)})]
    assert got == [("D32", "1", "", "STOPPED_AT", "A"), ("D32", "1", 5.5, "IN_TRANSIT_TO", "B"),  # waits at A, leaves
                   ("D32", "1", 30.7, "IN_TRANSIT_TO", "B"), ("D32", "1", 0.0, "STOPPED_AT", "B"),  # fast to B, stands
                   ("D32", "1", 1.5, "STOPPED_AT", "B"), ("D32", "1", "", "IN_TRANSIT_TO", "C"),  # creeps; after a gap
                   ("", "", "", "", "")], got  # no trip

    day = datetime.datetime(2026, 11, 1)
    assert until(day.replace(hour=6)) == day.replace(hour=STOP_HOUR)
    assert until(day.replace(hour=STOP_HOUR, minute=30)) == day.replace(day=2, hour=STOP_HOUR)  # started late: tomorrow's
    print("ok", rows[0])
