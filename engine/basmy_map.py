"""BAS.MY Kota Bharu's routes as LAJU reads them from the official GTFS (cache/basmy_gtfs.zip, see exam.GTFS), drawn on
one web page so a planner can check them against the real routes: each route's line (GTFS shapes: stop to stop in a
straight line, not the road; where a direction has none, its stops joined), its stops in order
per direction, its trips and first / last bus.

    python basmy_map.py   -> ../basmy_routes.html (open it in a browser; needs internet for the map tiles)
"""
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "basmy_routes.html"


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
                         "trips": len(ts), "first": first[0][:5] if first else "", "last": first[-1][:5] if first else "",
                         "services": sorted({t["service_id"] for t in ts})})
        out.append({"code": r["route_short_name"], "name": r["route_long_name"], "ways": ways})
    return out


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
<div id="side"><b>BAS.MY Kota Bharu: 16 routes, as LAJU reads them</b><br><small>From the official timetable (data.gov.my GTFS).
Lines join the stops straight (the official file has no road paths). Click a route: its line, its stops in order (numbered dots) and its timetable. Click again to show all.</small><div id="list"></div></div>
<div id="map"></div><script>
const R = __DATA__;
const COL = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#46f0f0","#f032e6","#bcf60c","#fabebe","#008080","#e6beff","#9a6324","#fffac8","#800000","#aaffc3"];
const map = L.map("map").setView([6.0, 102.2], 10);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {attribution: "© OpenStreetMap"}).addTo(map);
const layers = R.map((r, i) => {
  const g = L.layerGroup();
  r.ways.forEach((w, k) => {
    L.polyline(w.shape, {color: COL[i % 16], weight: 4, opacity: .85, dashArray: k ? "6 6" : null}).bindTooltip(`${r.code} ${r.name}`).addTo(g);
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
  R[pick].ways.forEach((w, k) => w.stops.forEach(([n, a, b], j) =>
    L.circleMarker([a, b], {radius: 5, color: k ? "#fff" : "#000", fillColor: COL[pick % 16], fillOpacity: 1, weight: 1})
      .bindTooltip(`${k ? "back" : "out"} #${j + 1}: ${n}`).addTo(dots)));
  map.fitBounds(L.featureGroup(layers[pick].getLayers()).getBounds(), {padding: [20, 20]});
}
document.getElementById("list").innerHTML = R.map((r, i) => `<div class="r" onclick="show(${i})">
  <span class="c" style="background:${COL[i % 16]}"></span><b>${r.code}</b> ${r.name}
  ${r.ways.map((w) => `<br><small>${w.direction === "1" ? "back (dashed)" : "out"}: ${w.stops.length} stops, ${w.trips} trips, first ${w.first}, last ${w.last} (${w.services.join(", ")})</small>
  <ol>${w.stops.map((s) => `<li>${s[0]}</li>`).join("")}</ol>`).join("")}</div>`).join("");
</script></body></html>"""

if __name__ == "__main__":
    data = routes()
    assert len(data) == 16 and all(r["ways"] and all(w["shape"] and w["stops"] for w in r["ways"]) for r in data), data
    OUT.write_text(PAGE.replace("__DATA__", json.dumps(data)), "utf-8")
    print("ok", OUT, [f"{r['code']}: {'/'.join(str(len(w['stops'])) for w in r['ways'])} stops" for r in data])
