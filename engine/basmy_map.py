"""BAS.MY Kota Bharu's routes as LAJU reads them from the official GTFS (cache/basmy_gtfs.zip, see exam.GTFS), drawn on
one web page so a planner can check them against the real routes: each route's road (the official file has none: its
shapes join the stops in a straight line, and direction 0 has no shape), traced from a real bus's GPS (cache/gps, see
gps.py); its stops in order per direction, its trips and first / last bus.

    python basmy_map.py   -> ../basmy_routes.html (open it in a browser; needs internet for the map tiles)
"""
import csv
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "basmy_routes.html"
GPS = HERE / "cache" / "gps"
REACH_M = 300  # a GPS dot this near passes a stop (30 s at 60 km/h is 500 m between dots)
JITTER_M = 30  # dots closer than this to the last one: the bus standing
SNAP_M = 60  # a GPS dot may belong to any road this near
OFF_MAP_M = 500  # a GPS dot further than this from every road node is on a road the map lacks
SPIKE_M = 200  # back at a road node after less than this: a dot put on the wrong road, not a real trip and back


def read(z, name):
    return list(csv.DictReader(io.TextIOWrapper(z.open(name), "utf-8-sig")))


def routes():
    z = zipfile.ZipFile(HERE / "cache" / "basmy_gtfs.zip")
    stops = {s["stop_id"]: s for s in read(z, "stops.txt")}
    shapes = defaultdict(list)
    for s in read(z, "shapes.txt"):
        shapes[s["shape_id"]].append((int(s["shape_pt_sequence"]), float(s["shape_pt_lat"]), float(s["shape_pt_lon"])))
    times = defaultdict(list)
    for s in read(z, "stop_times.txt"):
        times[s["trip_id"]].append((int(s["stop_sequence"]), s["departure_time"], s["stop_id"]))
    trips = defaultdict(list)  # (route, direction) -> trips
    for t in read(z, "trips.txt"):
        trips[t["route_id"], t["direction_id"]].append(t)
    out = []
    for r in sorted(read(z, "routes.txt"), key=lambda r: r["route_short_name"]):
        ways = []
        for d in ("0", "1"):
            ts = trips.get((r["route_id"], d))
            if not ts:
                continue
            seqs = {t["trip_id"]: sorted(times[t["trip_id"]]) for t in ts}
            longest = max(seqs.values(), key=len)  # the stop list of its longest trip
            first = sorted(s[0][1] for s in seqs.values() if s)
            seq = [[stops[x]["stop_name"], float(stops[x]["stop_lat"]), float(stops[x]["stop_lon"])] for _, _, x in longest]
            shape = [[a, b] for _, a, b in sorted(shapes[ts[0]["shape_id"]])] or [x[1:] for x in seq]  # some have none
            ways.append({"direction": d, "shape": shape, "stops": seq,
                         "trip_prefix": "_".join(ts[0]["trip_id"].split("_")[:2]),  # "249_1" of trip "249_1_WE_22"
                         "trips": len(ts), "first": first[0][:5] if first else "", "last": first[-1][:5] if first else "",
                         "services": sorted({t["service_id"] for t in ts})})
        out.append({"code": r["route_short_name"], "name": r["route_long_name"], "ways": ways})
    return out


def _unspike(line):
    """Points `line` (metres) without its short trips up a road and straight back (under SPIKE_M in all): GPS dots put on
    the wrong road. Longer ones the bus really drove (into a town and back out)."""
    out = [p for p, q in zip(line, [None] + line) if p != q]
    i = 1
    while i < len(out) - 1:
        k = 0
        while i - k > 0 and i + k + 1 < len(out) and out[i - k - 1] == out[i + k + 1]:
            k += 1
        if k and sum(math.dist(a, b) for a, b in zip(out[i - k:i + k], out[i - k + 1:i + k + 1])) < SPIKE_M:
            del out[i - k + 1:i + k + 1]
            i = max(1, i - k)
        else:
            i += 1
    return out


def _trace(xy, K, P, near):
    """The way a bus at GPS dots xy (metres, in time order) drove, as points (metres), along the roads K (km, both ways: the
    dots show which way it went), nodes at P. Each dot goes to the node within SNAP_M whose road from the last node best
    fits the distance between the dots (so not the other carriageway or a side road). A dot with no such road, on a road
    the map lacks (it ends at Kelantan's border), is drawn as it is, straight from the last, until a dot is within SNAP_M
    of a node again. (points, dots drawn as they are)"""
    import numpy as np
    from scipy.sparse.csgraph import dijkstra

    line, at, last, raw = [], None, xy[0], 0
    for k, p in enumerate(xy):
        gone = math.dist(p, last)
        if k and gone < JITTER_M:
            continue
        off, nodes = near.query(p, k=8)
        hop = None
        if at is None:
            if off[0] <= SNAP_M:
                hop = [int(nodes[0])]
        elif off[0] <= OFF_MAP_M:
            d, pred = dijkstra(K, indices=at, return_predecessors=True, limit=(1.5 * gone + 300) / 1000)
            cost = np.abs(d[nodes] * 1000 - gone) + off
            cost[off > max(SNAP_M, off[0])] = np.inf
            if np.isfinite(cost.min()):
                hop = [int(nodes[cost.argmin()])]
                while hop[-1] != at:
                    hop.append(int(pred[hop[-1]]))
                hop.reverse()
        if hop:
            line += [tuple(P[x]) for x in hop]
            at = hop[-1]
        else:
            line.append(tuple(p))
            at, raw = None, raw + 1
        last = p
    return _unspike(line), raw


def drive(data):
    """Adds to each way of routes() "road" (lat, lng along the roads) and "seen" (which bus, when, how much): of all its
    logged trips (cache/gps), the one passing both its end stops and most of the others, cut from where it is nearest its first stop to where it
    is nearest its last (before and after, the same trip id drives from the depot or waits)."""
    import numpy as np
    import shapely
    from scipy.spatial import cKDTree

    import kelantan

    R = kelantan.roads()
    near, K = cKDTree(R["P"]), R["K"].maximum(R["K"].T)
    logged = defaultdict(lambda: defaultdict(list))
    for f in sorted(GPS.glob("*.csv")):
        with open(f, newline="") as h:
            for r in csv.DictReader(h):
                if r["trip"]:
                    logged["_".join(r["trip"].split("_")[:2])][f.stem, r["bus"], r["trip"]].append(r)
    for way in (w for r in data for w in r["ways"]):
        stops, best = kelantan._m(*np.array([s[1:] for s in way["stops"]]).T), (False, 0, 0, None)
        for key, rows in logged[way["trip_prefix"]].items():
            if len(rows) > 9:
                xy = kelantan._m([float(x["lat"]) for x in rows], [float(x["lng"]) for x in rows])
                gap = np.diff(np.array([x["time"] for x in rows], "datetime64[s]").astype(int)).max()
                hit = cKDTree(xy).query(stops)[0] <= REACH_M
                best = max(best, (bool(hit[0] and hit[-1]), float(hit.mean()), -int(gap), key), key=lambda x: x[:3])
        if best[3]:
            rows = logged[way["trip_prefix"]][best[3]]
            xy = kelantan._m([float(x["lat"]) for x in rows], [float(x["lng"]) for x in rows])
            mid = len(xy) // 2
            a = np.hypot(*(xy[:mid] - stops[0]).T)
            i, j = mid - 1 - int(a[::-1].argmin()), mid + int(np.hypot(*(xy[mid:] - stops[-1]).T).argmin())
            line, raw = _trace(xy[i:j + 1], K, R["P"], near)
            drawn = shapely.LineString(line)
            lng, lat = (np.array(line) / kelantan._m(1, 1)).T
            way["road"] = np.round(shapely.get_coordinates(shapely.LineString(np.column_stack([lat, lng])).simplify(2e-5)), 5).tolist()
            way["seen"] = {"day": best[3][0], "bus": best[3][1], "from": rows[i]["time"][11:16], "to": rows[j]["time"][11:16],
                           "cover": round(best[1] * 100), "dots": j + 1 - i, "raw": raw, "km": round(drawn.length / 1000, 1),
                           "follow": round(float(np.percentile(shapely.distance(shapely.points(xy[i:j + 1]), drawn), 95)))}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BAS.MY routes check</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{margin:0;display:flex;height:100vh;font:14px system-ui,sans-serif;background:#111;color:#eee}
#side{width:360px;overflow:auto;padding:10px;box-sizing:border-box}#map{flex:1}
.r{padding:6px;border-bottom:1px solid #333;cursor:pointer}.r:hover,.r.on{background:#223}
.c{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
ol{margin:4px 0 8px;padding-left:22px;font-size:12px;color:#bbb}small{color:#999}
@media(max-width:700px){body{flex-direction:column}#side{width:100%;height:45vh}}</style></head><body>
<div id="side"><b>BAS.MY Kota Bharu: 16 routes, as LAJU reads them</b><br><small>Stops and times: the official timetable (data.gov.my GTFS).
Coloured lines: the road a real bus drove (its GPS, on the road map). Click a route: its stops in order (numbered dots), its timetable, and BAS.MY's own line in thin grey (its stops joined straight: the official file has no roads). Click again to show all.</small><div id="list"></div></div>
<div id="map"></div><script>
const R = __DATA__;
const COL = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#46f0f0","#f032e6","#bcf60c","#fabebe","#008080","#e6beff","#9a6324","#fffac8","#800000","#aaffc3"];
const map = L.map("map").setView([6.0, 102.2], 10);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {attribution: "© OpenStreetMap"}).addTo(map);
const layers = R.map((r, i) => {
  const g = L.layerGroup();
  r.ways.forEach((w, k) => {
    L.polyline(w.road || w.shape, {color: COL[i % 16], weight: 4, opacity: .85, dashArray: k ? "6 6" : null}).bindTooltip(`${r.code} ${r.name}`).addTo(g);
  });
  return g.addTo(map);
});
const dots = L.layerGroup().addTo(map);
let pick = -1;
function show(i) {
  pick = pick === i ? -1 : i;
  dots.clearLayers();
  layers.forEach((g, k) => (pick < 0 || k === pick ? g.addTo(map) : g.remove()));
  document.querySelectorAll(".r").forEach((e, k) => e.classList.toggle("on", k === pick));
  if (pick < 0) return;
  R[pick].ways.forEach((w) => L.polyline(w.shape, {color: "#777", weight: 1.5}).addTo(dots));
  R[pick].ways.forEach((w, k) => w.stops.forEach(([n, a, b], j) =>
    L.circleMarker([a, b], {radius: 5, color: k ? "#fff" : "#000", fillColor: COL[pick % 16], fillOpacity: 1, weight: 1})
      .bindTooltip(`${k ? "back" : "out"} #${j + 1}: ${n}`).addTo(dots)));
  map.fitBounds(L.featureGroup(layers[pick].getLayers()).getBounds(), {padding: [20, 20]});
}
document.getElementById("list").innerHTML = R.map((r, i) => `<div class="r" onclick="show(${i})">
  <span class="c" style="background:${COL[i % 16]}"></span><b>${r.code}</b> ${r.name}
  ${r.ways.map((w) => `<br><small>${w.direction === "1" ? "back (dashed)" : "out"}: ${w.stops.length} stops, ${w.trips} trips, first ${w.first}, last ${w.last} (${w.services.join(", ")})
  <br>${w.seen ? `road: bus ${w.seen.bus}, ${w.seen.day} ${w.seen.from}-${w.seen.to}, passes ${w.seen.cover}% of its stops, ${w.seen.km} km` : "no bus logged yet: stops joined straight"}</small>
  <ol>${w.stops.map((s) => `<li>${s[0]}</li>`).join("")}</ol>`).join("")}</div>`).join("");
</script></body></html>"""

if __name__ == "__main__":
    a, b, d = (0, 0), (100, 0), (200, 0)
    assert _unspike([a, b, (100, 50), b, d]) == [a, b, d]  # up a side road and back: a wrong dot
    assert _unspike([a, b, (100, 500), b, d]) == [a, b, (100, 500), b, d]  # far and back: really driven
    far = [(0, 0), (90, 0), (180, 0), (270, 0), (180, 0), (90, 0), (90, 50)]
    assert _unspike(far) == far  # small steps, but far in all
    data = routes()
    drive(data)
    assert len(data) == 16 and all(r["ways"] and all(w["shape"] and w["stops"] for w in r["ways"]) for r in data), data
    assert all(w["seen"]["follow"] < 150 for r in data for w in r["ways"] if "seen" in w), "a line leaves its bus"
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(data)), "utf-8")
    print("ok", OUT)
    for r in data:
        print(r["code"], [(w["direction"], w["seen"], len(w["road"])) if "road" in w else (w["direction"], "no bus") for w in r["ways"]])
