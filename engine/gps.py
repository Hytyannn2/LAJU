"""BAS.MY Kota Bharu's buses, live (PLAN.txt section 9, "Eyes and experience"): every 30 s, the position each bus reports
(data.gov.my GTFS Realtime, updated every 30 s) goes into cache/gps/<date>.csv, one row per new report: real speeds by
hour and road, delays against the timetable, buses really in service. Standard library only, so it runs for weeks in
its own window; a sleeping laptop or a dropped connection leaves gaps.

    python gps.py          -> logs until its window is closed (restarting appends)
    python gps.py --test   -> self-check: "ok ..."
"""
import csv
import datetime
import struct
import sys
import time
import urllib.request
from pathlib import Path

FEED = "https://api.data.gov.my/gtfs-realtime/vehicle-position/mybas-kota-bharu"
OUT = Path(__file__).resolve().parent / "cache" / "gps"
EVERY_S = 30
COLUMNS = ("bus", "plate", "trip", "route", "direction", "lat", "lng", "bearing", "speed_ms", "stop", "status", "occupancy",
           "time")


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seen = {}  # bus -> time of its last report written
    while True:
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
    if sys.argv[1:] != ["--test"]:
        main()

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
    print("ok", rows[0])
