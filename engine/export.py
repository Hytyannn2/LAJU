"""Exports of a saved design: GeoJSON (opens in QGIS) and GTFS (the standard transit feed; buses are route_type 3).

GTFS trips are frequency-based: one template trip per route direction, repeated every headway from 06:00, or in each
period of the day at that period's headway (out["periods"], metrics["periods"]; none where the route does not run).
Stop times are spread along each route in proportion to road distance.
"""
import csv
import io
import zipfile
from datetime import date, timedelta

import numpy as np

START_S = 6 * 3600


def geojson(out):
    m = out["metrics"]
    feats = [{"type": "Feature",
              "geometry": {"type": "LineString", "coordinates": [[lng, lat] for lat, lng in r["path"]]},
              "properties": {"route": i + 1, "stops": len(r["stops"]), "km": round(m["route_km"][i], 2),
                             "minutes": round(m["route_min"][i], 1), "headway_min": m["headway"][i],
                             "category": m.get("category", [""] * len(out["routes"]))[i],
                             "peak_hour_load": m["peak_hour_load"][i]}
                            | ({k: out["why"][i][k] for k in ("role", "boardings", "lost_riders", "lost_covered")}
                               if "why" in out else {})}
             for i, r in enumerate(out["routes"])]
    feats += [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [s["lng"], s["lat"]]},
               "properties": {k: s[k] for k in ("id", "name", "road", "key")} | {"routes": [i + 1 for i in s["routes"]]}}
              for s in out["stops"]]
    return {"type": "FeatureCollection", "features": feats}


def _offsets(r, stops, minutes):
    """Seconds from the first stop to each stop, by share of road distance along the path."""
    path = np.array(r["path"])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(path * [1, np.cos(np.radians(path[0, 0]))], axis=0).T))]
    out, k = [], 0
    for sid in r["stops"]:
        s = stops[sid]
        k += int(np.argmin(np.hypot(path[k:, 0] - s["lat"], path[k:, 1] - s["lng"])))
        out.append(seg[k] / seg[-1] * minutes * 60 if seg[-1] else 0)
    return out


def gtfs(out, run_id, today=None):
    today = today or date.today()
    stops, m = {s["id"]: s for s in out["stops"]}, out["metrics"]
    files = {
        "agency.txt": [["agency_id", "agency_name", "agency_url", "agency_timezone"],
                       ["laju", "LAJU scenario " + run_id, "https://example.invalid/laju", "Asia/Kuala_Lumpur"]],
        "calendar.txt": [["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                          "start_date", "end_date"],
                         ["daily"] + ["1"] * 7 + [f"{today:%Y%m%d}", f"{today + timedelta(days=365):%Y%m%d}"]],
        "stops.txt": [["stop_id", "stop_name", "stop_lat", "stop_lon"]]
                     + [[s["id"], s["name"], s["lat"], s["lng"]] for s in out["stops"]],
        "routes.txt": [["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"]],
        "trips.txt": [["route_id", "service_id", "trip_id", "direction_id", "shape_id"]],
        "stop_times.txt": [["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]],
        "frequencies.txt": [["trip_id", "start_time", "end_time", "headway_secs", "exact_times"]],
        "shapes.txt": [["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]],
    }
    hms = lambda s: f"{round(s) // 3600:02d}:{round(s) % 3600 // 60:02d}:{round(s) % 60:02d}"
    for i, r in enumerate(out["routes"]):
        rid = f"R{i + 1}"
        first, last = stops[r["stops"][0]]["name"], stops[r["stops"][-1]]["name"]
        files["routes.txt"].append([rid, "laju", str(i + 1), f"{first} - {last}", "3"])
        fwd = _offsets(r, stops, m["route_min"][i])
        for d, (seq, secs, path) in enumerate([(r["stops"], fwd, r["path"]),
                                              (r["stops"][::-1], [fwd[-1] - t for t in fwd[::-1]], r["path"][::-1])]):
            tid = f"{rid}_{d}"
            files["trips.txt"].append([rid, "daily", tid, str(d), tid])
            files["stop_times.txt"] += [[tid, hms(START_S + t), hms(START_S + t), sid, str(k + 1)]
                                        for k, (sid, t) in enumerate(zip(seq, secs))]
            spans = ([(a * 3600, b * 3600, h) for (_, a, b), h in zip(out["periods"], m["periods"][i]) if h]
                     if "periods" in m else [(START_S, START_S + out.get("service_hours", 16) * 3600, m["headway"][i])])
            files["frequencies.txt"] += [[tid, hms(a), hms(b), str(int(h * 60)), "0"] for a, b, h in spans]
            files["shapes.txt"] += [[tid, lat, lng, str(k + 1)] for k, (lat, lng) in enumerate(path)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, rows in files.items():
            text = io.StringIO()
            csv.writer(text, lineterminator="\n").writerows(rows)
            z.writestr(name, text.getvalue())
    return buf.getvalue()


if __name__ == "__main__":
    out = {
        "metrics": {"route_km": [2.0], "route_min": [10.0], "peak_hour_load": [40], "headway": [15]},
        "stops": [{"id": "a", "name": "A", "lat": 6.10, "lng": 102.20, "road": "primary road", "key": True, "routes": [0]},
                  {"id": "b", "name": "B", "lat": 6.11, "lng": 102.20, "road": "primary road", "key": False, "routes": [0]},
                  {"id": "c", "name": "C", "lat": 6.12, "lng": 102.20, "road": "primary road", "key": True, "routes": [0]}],
        "routes": [{"stops": ["a", "b", "c"], "path": [[6.10, 102.20], [6.105, 102.20], [6.11, 102.20], [6.12, 102.20]]}],
    }
    g = geojson(out)
    assert len(g["features"]) == 4 and g["features"][0]["geometry"]["coordinates"][0] == [102.20, 6.10]
    z = zipfile.ZipFile(io.BytesIO(gtfs(out, "test", date(2026, 10, 8))))
    st = list(csv.DictReader(io.TextIOWrapper(z.open("stop_times.txt"))))
    assert [x["arrival_time"] for x in st[:3]] == ["06:00:00", "06:05:00", "06:10:00"], st[:3]
    assert [x["stop_id"] for x in st[3:]] == ["c", "b", "a"] and st[4]["arrival_time"] == "06:05:00"
    assert "R1,laju,1,A - C,3" in z.read("routes.txt").decode()
    out["periods"], out["metrics"]["periods"] = [["peak", 6, 9], ["day", 9, 16], ["night", 19, 24]], [[10, 30, None]]
    fq = list(csv.DictReader(io.TextIOWrapper(zipfile.ZipFile(io.BytesIO(gtfs(out, "test"))).open("frequencies.txt"))))
    assert [(x["start_time"], x["end_time"], x["headway_secs"]) for x in fq[:2]] == [("06:00:00", "09:00:00", "600"),
                                                                                     ("09:00:00", "16:00:00", "1800")], fq
    print("ok", z.namelist())
