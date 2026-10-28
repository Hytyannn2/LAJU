"""Phase 3 traffic simulation with Eclipse SUMO (C++ microsimulator, run from Python).

simulate() turns the same OSM road graph LAJU routes on into a SUMO network for the area a design covers (left-hand
traffic), adds background cars between key stops from the gravity trip matrix and the designed bus lines (kerb-lane
stops, one bus per headway each way), runs the peak hour, and returns road congestion for the map heatmap plus
simulated bus trip times. Batch runs only need the sumo binary and its output files, so no TraCI loop.

    python traffic.py   -> simulates a small Kota Bharu design (needs the kelantan.py cache)
"""
import hashlib
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import sumo
import sumolib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

import kelantan
from sim import PEAK_SHARE

BIN = Path(sumo.__file__).parent / "bin"
CARS_PER_TRIP = 2.0  # background cars per bus-able trip in the trip matrix: uncalibrated, scale it in the UI
DWELL_S = 20
LANES = {"motorway": 2, "trunk": 2, "primary": 2}  # per direction; OSM lane counts are not kept, other roads get 1


NETCONVERT = ["--proj.utm", "--lefthand", "--geometry.remove", "--roundabouts.guess", "--ramps.guess", "--junctions.join",
              "--tls.guess", "--tls.join", "--no-turnarounds.except-deadend", "--output.original-names", "--no-warnings"]


def _net(bbox):
    """SUMO network for bbox (south, west, north, east), cached by area and netconvert options."""
    folder = kelantan.CACHE / "sumo"
    folder.mkdir(parents=True, exist_ok=True)
    net = folder / f"{hashlib.sha1(repr((np.round(bbox, 3).tolist(), NETCONVERT)).encode()).hexdigest()[:12]}.net.xml"
    if not net.exists():
        R = kelantan.roads()
        s, w, n, e = bbox
        inside = (R["lat"] >= s) & (R["lat"] <= n) & (R["lng"] >= w) & (R["lng"] <= e)
        k = np.flatnonzero(inside[R["u"]] & inside[R["v"]])
        # Clipping cuts off one-way stubs; keep the part every car can drive in and out of.
        lab = connected_components(R["G"][inside][:, inside], connection="strong")[1]
        part = np.zeros(len(inside), bool)
        part[np.flatnonzero(inside)[lab == np.bincount(lab).argmax()]] = True
        k = k[part[R["u"][k]] & part[R["v"][k]]]
        nodes = np.unique(np.r_[R["u"][k], R["v"][k]])
        (folder / "plain.nod.xml").write_text("<nodes>" + "".join(
            f'<node id="n{i}" x="{R["lng"][i]:.7f}" y="{R["lat"][i]:.7f}"/>' for i in nodes) + "</nodes>")
        edges = []
        for j in k:
            c = str(R["cls"][j]).replace("_link", "")
            speed = R["km"][j] / R["minutes"][j] * 60 / 3.6
            prio = {"motorway": 13, "trunk": 12, "primary": 11, "secondary": 10}.get(c, 9)
            edges.append(f'<edge id="e{j}" from="n{R["u"][j]}" to="n{R["v"][j]}" speed="{speed:.1f}" '
                         f'priority="{prio}" numLanes="{LANES.get(c, 1)}"/>')
        (folder / "plain.edg.xml").write_text("<edges>" + "".join(edges) + "</edges>")
        subprocess.run([BIN / "netconvert", "--node-files", "plain.nod.xml", "--edge-files", "plain.edg.xml",
                        "-o", net.name] + NETCONVERT, cwd=folder, check=True, capture_output=True)
    return net


def _drive(R, a, b):
    """Road graph nodes of the quickest legal path a -> b: what the plan's travel times assume."""
    pred = dijkstra(R["G"], indices=a, return_predecessors=True, limit=60)[1]
    p = [b]
    while p[-1] != a and p[-1] >= 0:
        p.append(int(pred[p[-1]]))
    return p[::-1] if p[-1] == a else None


def simulate(out, scale=1.0, seconds=3600, seed=42):
    """out = a saved design ('output' of a run). scale multiplies the background traffic."""
    pts = np.array([pt for r in out["routes"] for pt in r["path"]])
    (s, w), (n, e) = pts.min(0) - 0.01, pts.max(0) + 0.01
    net_file = _net((s, w, n, e))
    net = sumolib.net.readNet(str(net_file))
    folder = net_file.parent / f"run-{hashlib.sha1(repr((out['routes'], scale)).encode()).hexdigest()[:12]}"
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    stops = {x["id"]: x for x in out["stops"]}
    heads = out["metrics"]["headway"]
    R = kelantan.roads()
    J = csr_matrix((np.arange(len(R["u"])) + 1, (R["u"], R["v"])), shape=R["G"].shape)  # (u, v) -> edge index + 1

    def sumo_edge(j):
        hits = net.getEdgesByOrigID(f"e{j}")  # empty if netconvert folded it into a junction
        return next(iter(hits)) if hits else None

    add, veh, lanes = ['<additional>', f'<edgeData id="peak" file="edges.xml" begin="0" end="{seconds}"/>'], [], {}
    for i, r in enumerate(out["routes"]):
        for d, seq in enumerate((r["stops"], r["stops"][::-1])):
            # Drive the planned roads key stop to key stop (quickest legal path, as the plan assumes), mapped onto
            # SUMO edges; then stop at every planned stop that lies on that path, in order.
            keys = [int(x[1:]) for x in seq if stops[x]["key"]]
            walk = [keys[0]]
            for a, b in zip(keys, keys[1:]):
                leg = _drive(R, a, b)
                if leg is None:
                    break
                walk += leg[1:]
            js = np.asarray(J[walk[:-1], walk[1:]]).ravel() - 1
            route = []
            for j in js:
                edge = sumo_edge(j)
                if edge is None or (route and route[-1] is edge):
                    continue
                if route and edge not in route[-1].getOutgoing():
                    gap = net.getShortestPath(route[-1], edge, vClass="bus")[0]
                    route += (gap or [None])[1:-1]
                route.append(edge)
            if len(route) < 2 or None in route:
                continue
            calls, at = [], 0
            for x in seq:
                v = int(x[1:])
                if v not in walk[at:]:
                    continue  # this stop is on the other street of a one-way pair
                at = walk.index(v, at)
                edge = sumo_edge(js[at - 1] if at else js[0])
                lane = edge.getLane(0) if edge else None
                if lane is None or lane.getLength() < 16:
                    continue
                pos = lane.getClosestLanePosAndDist(net.convertLonLat2XY(stops[x]["lng"], stops[x]["lat"]))[0]
                a = int(min(max(0, pos - 16), lane.getLength() - 16))
                near = [sid for b, sid in lanes.get(lane.getID(), []) if abs(b - a) < 16]
                sid = near[0] if near else f"{lane.getID()}@{a}"
                if not near:
                    lanes.setdefault(lane.getID(), []).append((a, sid))
                    add.append(f'<busStop id="{sid}" lane="{lane.getID()}" startPos="{a}" endPos="{a + 16}"/>')
                if not calls or calls[-1] != sid:
                    calls.append(sid)
            if len(calls) >= 2:
                body = "".join(f'<stop busStop="{sid}" duration="{DWELL_S}"/>' for sid in calls)
                edges = " ".join(e.getID() for e in route)
                veh.append((0, f'<flow id="bus{i}_{d}" type="bus" begin="0" end="{seconds}" period="{heads[i] * 60}">'
                               f'<route edges="{edges}"/>{body}</flow>'))
    add.append("</additional>")

    zones = out["model"]
    zxy = [net.convertLonLat2XY(b, a) for a, b in zip(zones["lat"], zones["lng"])]
    near = [[ed.getID() for ed, _ in net.getNeighboringEdges(x, y, 1500) if ed.allows("passenger")] for x, y in zxy]
    od = np.array(zones["od"]) * PEAK_SHARE * CARS_PER_TRIP * scale * seconds / 3600
    for (a, b), lam in np.ndenumerate(od):
        if near[a] and near[b] and lam > 0:
            for t in rng.uniform(0, seconds, rng.poisson(lam)):
                veh.append((t, f'<trip id="car{a}_{b}_{t:.0f}_{len(veh)}" type="car" depart="{t:.1f}" '
                               f'from="{rng.choice(near[a])}" to="{rng.choice(near[b])}"/>'))
    veh.sort(key=lambda v: v[0])
    (folder / "stops.add.xml").write_text("\n".join(add))
    (folder / "demand.rou.xml").write_text("\n".join(
        ['<routes>', '<vType id="bus" vClass="bus" length="12" maxSpeed="22"/>', '<vType id="car" vClass="passenger"/>']
        + [v for _, v in veh] + ["</routes>"]))
    subprocess.run([BIN / "sumo", "-n", net_file, "-r", "demand.rou.xml", "-a", "stops.add.xml", "--begin", "0",
                    "--end", str(2 * seconds), "--tripinfo-output", "trips.xml", "--statistic-output", "stats.xml",
                    "--time-to-teleport", "300", "--ignore-route-errors", "--no-step-log", "--no-warnings",
                    "--seed", str(seed)], cwd=folder, check=True, capture_output=True)

    congestion = []
    for ed in ET.parse(folder / "edges.xml").getroot().iter("edge"):
        if ed.get("speed") and not ed.get("id").startswith(":"):
            edge = net.getEdge(ed.get("id"))
            c = 1 - float(ed.get("speed")) / edge.getSpeed()
            line = [net.convertXY2LonLat(x, y)[::-1] for x, y in edge.getShape()]
            congestion.append({"path": np.round(line, 5).tolist(), "c": round(min(max(c, 0), 1), 2)})
    buses = {}
    for t in ET.parse(folder / "trips.xml").getroot().iter("tripinfo"):
        if t.get("id").startswith("bus"):
            buses.setdefault(t.get("id").split(".")[0], []).append((float(t.get("duration")), float(t.get("timeLoss"))))
    stats = ET.parse(folder / "stats.xml").getroot()
    return {
        "congestion": congestion,
        "buses": [{"route": int(k[3:].split("_")[0]), "dir": int(k.split("_")[1]), "trips": len(v),
                   "minutes": round(np.mean([x for x, _ in v]) / 60, 1), "delay": round(np.mean([y for _, y in v]) / 60, 1)}
                  for k, v in sorted(buses.items())],
        "cars": int(stats.find("vehicles").get("loaded")) - sum(len(v) for v in buses.values()),
        "teleports": int(stats.find("teleports").get("total")),
        "scale": scale,
    }


if __name__ == "__main__":
    import kelantan
    import routes
    from sim import evaluate

    p, net = kelantan.build((6.08, 102.2, 6.16, 102.3), n_stops=12)
    rs = routes.baseline(p, 3)
    out = kelantan.describe(p, net, rs) | {"metrics": evaluate(p, rs, detail=True),
                                           "model": {"lat": p.xy[:, 1].tolist(), "lng": p.xy[:, 0].tolist(), "od": p.od.tolist()}}
    res = simulate(out, scale=0.5, seconds=900)
    assert res["congestion"] and all(0 <= c["c"] <= 1 for c in res["congestion"])
    assert res["buses"], "buses completed trips"
    m = evaluate(p, rs)
    print("ok", len(res["congestion"]), "edges,", res["cars"], "cars,", res["teleports"], "teleports")
    for b in res["buses"]:
        planned = (m["route_min_back"] if b["dir"] else m["route_min"])[b["route"]]
        print(f"route {b['route']} dir {b['dir']}: {b['trips']} trips, {b['minutes']} min (planned {planned:.0f}), delay {b['delay']} min")
