"""Kelantan data and AI stop placement (Phase 2). Buses only: the network is the OSM roads a bus can use, never rail.

Downloaded once into engine/cache/ (delete a file there to refresh it):
  OpenStreetMap  Geofabrik Malaysia extract (ODbL), read with GDAL: Kelantan boundary and districts, roads from
                 expressway to tertiary with their road class, and hospitals, clinics, schools, universities, markets,
                 malls and bus stations
  population     Kontur 400 m H3 hexagons (CC BY 4.0) scaled to DOSM district estimates (OpenDOSM, CC BY 4.0)
  buildings      Microsoft Global ML Building Footprints (ODbL): each hexagon's people placed on the roofs seen from space
  district data  public hospital beds (MOH, data.gov.my) and government-school pupils (KPM, OpenDOSM), CC BY 4.0: how
                 big each hospital and school is, instead of one guess for all
build(): the AI picks key stops (weighted k-means over people and places, moved onto the road spot where a bus may stop
and the people around it walk least),
links neighbouring key stops by road, adds intermediate stops along those roads where people live, and estimates
who travels where with a gravity model. The result is a sim.Problem plus what describe() needs to draw it.

    python kelantan.py   -> downloads everything (first run: several minutes), then self-checks a Kota Bharu design
"""
import csv
import gzip
import io
import json
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

import routes
from sim import DWELL_MIN, HEADWAYS, Problem, evaluate

CACHE = Path(__file__).resolve().parent / "cache"
BOX = (101.2, 4.4, 102.8, 6.4)  # lng/lat box around Kelantan

# OSM road class -> (typical bus speed km/h, may a bus stop here to drop off / pick up?).
# Expressways and ramps: buses drive on them but never stop. Small roads (residential, unclassified, service, track)
# are left out entirely: too narrow for a 12 m bus. Calibration knob: add "unclassified": (30, True) for rural feeders.
ROADS = {
    "motorway": (80, False), "motorway_link": (40, False),
    "trunk": (60, True), "trunk_link": (35, False),
    "primary": (45, True), "primary_link": (30, False),
    "secondary": (40, True), "secondary_link": (30, False),
    "tertiary": (35, True), "tertiary_link": (25, False),
}
JUNCTION_M = 30    # stops stay this far from junctions, roundabouts, bridges, tunnels and ramps
WALK_M = 500       # people within this distance of a stop count as covered
WALK_MIN_PER_KM = 17  # walking 4.5 km/h on streets about 1.3x longer than the straight line
WALK_CAP_MIN = 15  # farther (~900 m), people ride a motorbike or get dropped at the stop: in Kelantan's heat few walk
                   # longer to a bus (planner, 2026-10-09; was 30). Access counts as this
WALK_WEIGHT = 2.0  # a walking minute costs two riding minutes (heat, effort; transport studies use 1.5-2.5)
SPACING_M = ((2000, 400), (300, 800), (0, 1500))  # (people within PEOPLE_NEAR_M >=, metres since last stop)
MIN_PEOPLE = 50    # no intermediate stop where (almost) nobody lives
TRIP_RATE = 0.05   # bus trips per resident per day: potential demand, not today's ridership
BETA = 0.05        # gravity model: trips fall off as exp(-BETA * minutes)
VOT_RM_H = 10.0    # value of passenger time per hour (bus costs per category: sim.CATEGORIES)
UNSERVED_RM = 100.0  # per trip left unserved: high on purpose, serving the area is the goal (a soft constraint). At 50
                     # the score tied a state network leaving 20.3% of trips without a bus with one leaving 8.5% (2026-10-18)
MAX_DETOUR = 1.5   # a route may take 1.5x the quickest drive between its two ends; each minute beyond costs (sim.evaluate):
                   # hooks into side roads and zigzags confuse riders and planners (planner, 2026-10-10)
TRANSFER_MIN = 10  # a change of bus feels like 10 riding minutes on top of the wait (studies: 10-15; Mandl's benchmark
                   # uses 5). At 5 the AI spread routes thin and sent riders to busy places by changing (2026-10-10)
MAX_TRANSFERS = 2  # a trip needing more changes counts as unserved: riders would not make it (planner, 2026-10-18; it was
                   # served, and 12-21% of the state network's trips needed 3+ changes)
PLACES = {  # daily trips attracted per place: planning assumptions, replace with surveys
    "university": 4000, "college": 800, "hospital": 2500, "clinic": 200, "school": 400,
    "marketplace": 1500, "mall": 2500, "supermarket": 400, "bus_station": 1500,
}
# Real sizes where official data exists: a hospital attracts trips by its district's public beds (MOH), a school by its
# district's pupils (KPM), shared by the district's hospitals / schools in OSM. The rates keep the old totals
# (26 hospitals x 2,500, 427 schools x 400 trips/day): the data decides which places are big, not how many ride a bus.
TRIPS_PER_BED = 22.0
TRIPS_PER_PUPIL = 0.57
# Shops and markets attract trips by the footprint of the building OSM maps them in (planner, 2026-10-18: AEON Mall,
# the state's biggest, counted as much as a small bazaar). The rates keep each kind's old total over the places OSM
# sizes; one without a mapped building keeps PLACES. ponytail: footprint, not floor area (building:levels is on 1 in 6
# retail buildings in OSM); multiply by levels once OSM has them.
TRIPS_PER_M2 = {"mall": 0.24, "supermarket": 0.13, "marketplace": 0.75}
ALIASES = {"pasir putih": "pasir puteh", "lojing": "gua musang"}  # KPM spelling; OSM's Lojing district is in Gua Musang
PEOPLE_NEAR_M = 485  # "people nearby" for intermediate stops: a circle as big as one population hexagon (H3, 0.74 km2)
OSM = "https://download.geofabrik.de/asia/malaysia-singapore-brunei-latest.osm.pbf"
KONTUR = "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/kontur_population_MY_20231101.gpkg.gz"
DOSM = "https://storage.dosm.gov.my/population/population_district.csv"
BEDS = "https://storage.data.gov.my/healthcare/hospital_beds.csv"  # MOH via data.gov.my, CC BY 4.0
PUPILS = "https://storage.data.gov.my/education/enrolment_school_district.csv"  # KPM via OpenDOSM, CC BY 4.0
BUILDINGS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"  # Microsoft, ODbL


def _m(lat, lng):
    """Metres on a flat Kelantan (equirectangular at 5.5 N), good to <1% across the state."""
    return np.column_stack([np.asarray(lng, float) * 110_800, np.asarray(lat, float) * 110_570])


def _download(url, f, unzip=False):
    if not f.exists():
        CACHE.mkdir(exist_ok=True)
        part = f.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(part, "wb") as dst:
            shutil.copyfileobj(gzip.open(r) if unzip else r, dst)
        part.replace(f)
    return f


def _osm(layer, where):
    """One layer of the OpenStreetMap extract (GDAL OSM driver: tags without a column sit in other_tags)."""
    return gpd.read_file(_download(OSM, CACHE / "malaysia.osm.pbf"), layer=layer, where=where, bbox=BOX)


def _tags(other):
    return dict(re.findall(r'"([^"]+)"=>"([^"]*)"', other)) if isinstance(other, str) else {}


@lru_cache
def boundary():
    f = CACHE / "kelantan.geojson"
    if not f.exists():
        b = _osm("multipolygons", "boundary = 'administrative' AND admin_level = '4' AND name = 'Kelantan'")
        f.write_text(shapely.to_geojson(b.geometry.iloc[0]))
    return shapely.from_geojson(f.read_text())


@lru_cache
def roads():
    f = CACHE / "roads.npz"
    if not f.exists():
        g = _osm("lines", "highway IN (%s)" % ",".join(f"'{k}'" for k in ROADS))
        g = g[g.intersects(boundary())].reset_index(drop=True)
        tags = [_tags(t) for t in g["other_tags"]]
        allowed = np.array([t.get("access") not in ("no", "private") or "yes" in (t.get("bus"), t.get("psv")) for t in tags])
        g, tags = g[allowed].reset_index(drop=True), [t for t, a in zip(tags, allowed) if a]
        hw = g["highway"].to_numpy().astype(str)
        # Graph of OSM nodes: ways that share a node share its exact coordinates.
        xy = [np.asarray(line.coords)[:, :2] for line in g.geometry]
        key, node = np.unique(np.vstack(xy).round(7), axis=0, return_inverse=True)
        node, way = node.ravel(), np.repeat(np.arange(len(xy)), [len(c) for c in xy])
        a = np.flatnonzero(way[:-1] == way[1:])
        u, v, w = node[a], node[a + 1], way[a]
        oneway = np.array([t.get("oneway", "") for t in tags])
        roundabout = np.array([t.get("junction") in ("roundabout", "circular") for t in tags])
        fwd_only = np.isin(oneway, ["yes", "true", "1"]) | roundabout | (np.isin(hw, ["motorway", "motorway_link"]) & (oneway != "no"))
        f_, b_ = oneway[w] != "-1", ~fwd_only[w]
        u, v, w = np.r_[u[f_], v[b_]], np.r_[v[f_], u[b_]], np.r_[w[f_], w[b_]]
        # Keep the largest strongly connected part: a bus must be able to get back.
        N = len(key)
        lab = connected_components(csr_matrix((np.ones(len(u)), (u, v)), shape=(N, N)), connection="strong")[1]
        big = lab == np.bincount(lab).argmax()
        keep = big[u] & big[v] & (u != v)
        new = np.cumsum(big) - 1
        u, v, w = new[u[keep]], new[v[keep]], w[keep]
        lng, lat = key[big, 0], key[big, 1]
        P = _m(lat, lng)
        km = np.hypot(*(P[v] - P[u]).T) / 1000
        cls = hw[w]
        minutes = km / np.array([ROADS[c][0] for c in cls]) * 60
        ok = np.array([ROADS[h][1] and not r and t.get("bridge", "no") == "no" and t.get("tunnel", "no") == "no"
                       for h, r, t in zip(hw, roundabout, tags)])[w]
        name = np.array([nm if isinstance(nm, str) else t.get("ref", "") for nm, t in zip(g["name"], tags)])[w]
        o = np.lexsort((minutes, v, u))  # parallel ways between two nodes: keep the quickest
        o = o[np.r_[True, (np.diff(u[o]) != 0) | (np.diff(v[o]) != 0)]]
        u, v, cls, km, minutes, ok, name = u[o], v[o], cls[o], km[o], minutes[o], ok[o], name[o]
        n = len(lat)
        # A bus may stop at a node in the middle of a stop-able road, away from anything that is not.
        good = np.ones(n, bool)
        np.logical_and.at(good, u, ok)
        np.logical_and.at(good, v, ok)
        pairs = np.unique(np.sort(np.column_stack([u, v]), 1), axis=0)
        good &= np.bincount(pairs.ravel(), minlength=n) == 2
        stop_ok = good & (cKDTree(P[~good]).query(P)[0] >= JUNCTION_M)
        node_cls, node_name = np.empty(n, cls.dtype), np.empty(n, name.dtype)
        node_cls[u], node_name[u] = cls, name
        np.savez_compressed(f, lat=lat, lng=lng, u=u, v=v, cls=cls, km=km, minutes=minutes, stop_ok=stop_ok,
                            node_cls=node_cls, node_name=node_name)
    R = dict(np.load(f))
    N = len(R["lat"])
    R["G"] = csr_matrix((R["minutes"], (R["u"], R["v"])), shape=(N, N))
    R["K"] = csr_matrix((R["km"], (R["u"], R["v"])), shape=(N, N))
    R["P"] = _m(R["lat"], R["lng"])
    return R


def _district(name):
    """'<Jawi> / Bachok', 'Jajahan Kecil Lojing' -> 'bachok', 'lojing'."""
    words = str(name).split("/")[-1].lower().split()
    return " ".join(x for x in words if x not in ("jajahan", "daerah", "kecil", "district"))


def _dosm(lat, lng, pop):
    """Scale hexagons so each district matches DOSM's latest estimate, then the state total (districts OSM lacks)."""
    with urllib.request.urlopen(DOSM) as r:
        rows = [x for x in csv.DictReader(io.TextIOWrapper(r, "utf-8")) if x["state"] == "Kelantan"
                and x["sex"] == "both" and x["age"] == "overall" and x["ethnicity"] == "overall"]
    year = max(x["date"] for x in rows)
    dosm = {_district(x["district"]): float(x["population"]) * 1000 for x in rows if x["date"] == year}
    d = _osm("multipolygons", "boundary = 'administrative' AND admin_level = '6'")
    d = d.assign(key=d["name"].map(_district))
    d = d[d["key"].isin(dosm)]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lng, lat), crs=d.crs)
    j = gpd.sjoin(pts, d[["key", "geometry"]], predicate="within", how="left")
    key = j[~j.index.duplicated()]["key"].fillna("").to_numpy()
    out = pop.copy()
    for k in set(key) - {""}:
        out[key == k] *= dosm[k] / pop[key == k].sum()
    return out * sum(dosm.values()) / out.sum(), year[:4]


@lru_cache
def population():
    """(lat, lng, people) per 400 m hexagon inside Kelantan, and the DOSM year it is scaled to."""
    f = CACHE / "population.npz"
    if not f.exists():
        gpkg = _download(KONTUR, CACHE / "kontur_MY.gpkg", unzip=True)
        poly = boundary()
        hexes = gpd.read_file(gpkg, bbox=tuple(gpd.GeoSeries([poly], crs=4326).to_crs(3857).total_bounds))
        c = hexes.geometry.centroid.to_crs(4326)
        inside = c.within(poly).to_numpy()
        lat, lng = c.y.to_numpy()[inside], c.x.to_numpy()[inside]
        pop, year = _dosm(lat, lng, hexes["population"].to_numpy()[inside])
        np.savez_compressed(f, lat=lat, lng=lng, pop=pop, year=year)
    d = np.load(f)
    return d["lat"], d["lng"], d["pop"], str(d["year"])


@lru_cache
def places():
    """OSM places that attract trips, each with the trips a day it attracts ("trips"): PLACES, for hospitals and
    schools their share of their district's real beds / pupils, for shops and markets their footprint ("m2")."""
    f = CACHE / "places.json"
    if not f.exists():
        pts = _osm("points", "other_tags LIKE '%amenity%' OR other_tags LIKE '%shop%'")
        areas = _osm("multipolygons", "amenity IS NOT NULL OR shop IS NOT NULL OR building IN ('retail', 'commercial', 'supermarket')")
        tags = [_tags(t) for t in pts["other_tags"]]
        found = [(p, nm, t.get("amenity"), t.get("shop")) for p, nm, t in zip(pts.geometry, pts["name"], tags)]
        found += zip(areas.geometry.representative_point(), areas["name"], areas["amenity"], areas["shop"])
        poly, rows = boundary(), []
        for p, nm, a, s in found:
            kind = a if a in PLACES else s
            if kind in PLACES and poly.contains(p):
                rows.append({"lat": p.y, "lng": p.x, "kind": kind, "name": nm if isinstance(nm, str) else ""})
        # Footprint of the shop, market or commercial building each place sits in, the smallest if several (a shop mapped
        # inside a mall gets its own); 0 where OSM maps none. Buildings only: many markets are mapped as open lots
        # (Pasar Awam Bukit Bunga's 15,000 m2 would have drawn 3x Pasar Besar Siti Khadijah).
        b = areas["building"]
        shops = areas[b.notna() & (areas["shop"].isin(TRIPS_PER_M2) | areas["amenity"].isin(TRIPS_PER_M2)
                                   | b.isin(["retail", "commercial", "supermarket"]))].geometry.to_numpy()
        m2 = shapely.area(shapely.transform(shops, lambda c: _m(c[:, 1], c[:, 0])))
        size = np.full(len(rows), np.inf)
        i, j = shapely.STRtree(shops).query(shapely.points([(x["lng"], x["lat"]) for x in rows]), predicate="within")
        np.minimum.at(size, i, m2[j])
        for x, s in zip(rows, size):
            x["m2"] = round(float(s)) if np.isfinite(s) else 0
        f.write_text(json.dumps(rows))
    rows = json.loads(f.read_text())
    d = districts()
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x["lng"] for x in rows], [x["lat"] for x in rows]), crs=d.crs)
    j = gpd.sjoin(pts, d, predicate="within", how="left")
    key = j[~j.index.duplicated()]["key"].fillna("").to_numpy()
    num, count = district_numbers(), Counter(zip(key, (x["kind"] for x in rows)))
    for k, x in zip(key, rows):
        x["trips"] = float(PLACES[x["kind"]])
        if x["kind"] == "hospital":  # no public beds in its district: a private hospital or a clinic tagged as one
            x["trips"] = TRIPS_PER_BED * num["beds"][k] / count[k, "hospital"] if num["beds"].get(k) else PLACES["clinic"]
        elif x["kind"] == "school" and num["pupils"].get(k):
            x["trips"] = TRIPS_PER_PUPIL * num["pupils"][k] / count[k, "school"]
        elif x["kind"] in TRIPS_PER_M2 and x["m2"]:
            x["trips"] = TRIPS_PER_M2[x["kind"]] * x["m2"]
    return rows


@lru_cache
def districts():
    """Kelantan's districts (OSM admin level 6) with the key that matches KPM / MOH data."""
    f = CACHE / "districts.geojson"
    if not f.exists():
        d = _osm("multipolygons", "boundary = 'administrative' AND admin_level = '6'")
        f.write_text(d[d.representative_point().within(boundary())][["name", "geometry"]].to_json())
    d = gpd.read_file(f)
    return gpd.GeoDataFrame({"key": [ALIASES.get(_district(x), _district(x)) for x in d["name"]]}, geometry=d.geometry)


@lru_cache
def district_numbers():
    """{"beds": {district: public hospital beds}, "pupils": {district: government-school pupils}}, Kelantan, latest year."""
    f = CACHE / "district_numbers.json"
    if not f.exists():
        def latest(url, value, keep):
            with urllib.request.urlopen(url) as r:
                rows = [x for x in csv.DictReader(io.TextIOWrapper(r, "utf-8"))
                        if x["state"] == "Kelantan" and x["district"] != "All Districts" and keep(x)]
            year, out = max(x["date"] for x in rows), defaultdict(float)
            for x in rows:
                if x["date"] == year:
                    k = _district(x["district"])
                    out[ALIASES.get(k, k)] += float(x[value])
            return dict(out), year[:4]

        beds, beds_year = latest(BEDS, "beds", lambda x: x["type"] in ("hospital_moh", "hospital_non_moh"))
        pupils, pupils_year = latest(PUPILS, "students", lambda x: x["sex"] == "both")
        f.write_text(json.dumps({"beds": beds, "beds_year": beds_year, "pupils": pupils, "pupils_year": pupils_year}))
    return json.loads(f.read_text())


def _quadkey_box(qk):
    """lat/lng box (south, west, north, east) of a Bing Maps quadkey tile."""
    x = y = 0
    for ch in qk:
        x, y = 2 * x + (int(ch) & 1), 2 * y + (int(ch) >> 1)
    n = 2 ** len(qk)
    lat = lambda t: float(np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * t / n)))))
    return lat(y + 1), x / n * 360 - 180, lat(y), (x + 1) / n * 360 - 180


@lru_cache
def buildings():
    """(lat, lng, m2) of every building in Kelantan as seen from space (Microsoft Global ML Building Footprints, ODbL):
    downloaded once (~55 MB), kept as centroids and roof areas."""
    f = CACHE / "buildings.npz"
    if not f.exists():
        with urllib.request.urlopen(BUILDINGS) as r:
            links = [x.split(",") for x in r.read().decode().splitlines() if x.startswith("Malaysia,")]
        lat, lng, m2 = [], [], []
        for _, qk, url, *_ in links:
            s, w, n, e = _quadkey_box(qk)
            if s < BOX[3] and n > BOX[1] and w < BOX[2] and e > BOX[0]:
                with urllib.request.urlopen(url) as r:
                    for line in gzip.open(r):
                        ring = np.asarray(json.loads(line)["geometry"]["coordinates"][0])[:-1]
                        x, y = (ring[:, 0] - ring[0, 0]) * 110_800, (ring[:, 1] - ring[0, 1]) * 110_570
                        m2.append(abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2)
                        lng.append(ring[:, 0].mean())
                        lat.append(ring[:, 1].mean())
        lat, lng, m2 = np.array(lat), np.array(lng), np.array(m2)
        inside = shapely.contains_xy(boundary(), lng, lat)
        np.savez_compressed(f, lat=lat[inside], lng=lng[inside], m2=m2[inside])
    d = np.load(f)
    return d["lat"], d["lng"], d["m2"]


@lru_cache
def people():
    """(lat, lng, people) where people live: each population hexagon's people shared by the buildings in it, by roof
    area, so homes sit where the satellite saw them. A hexagon with people but no building seen keeps its centre."""
    lat, lng, pop, _ = population()
    blat, blng, m2 = buildings()
    d, h = cKDTree(_m(lat, lng)).query(_m(blat, blng))
    ok = d < 600  # in a populated hexagon (corners ~460 m from the centre); other roofs are farms, factories, empty
    roof = np.bincount(h[ok], weights=m2[ok], minlength=len(pop))
    alone = roof == 0
    return (np.r_[blat[ok], lat[alone]], np.r_[blng[ok], lng[alone]],
            np.r_[pop[h[ok]] * m2[ok] / roof[h[ok]], pop[alone]])


@lru_cache
def _people_tree():
    lat, lng, ppl = people()
    return cKDTree(_m(lat, lng)), ppl


def _walk_min(metres):
    return np.minimum(metres / 1000 * WALK_MIN_PER_KM, WALK_CAP_MIN)


def _kmeans(X, w, k, fixed, iters=30, seed=0):
    """Weighted k-means. `fixed` centres (planner stops) never move but still claim the points around them."""
    rng = np.random.default_rng(seed)
    k = min(k, int((w > 0).sum()))
    C = np.vstack([fixed, X[rng.choice(len(X), k, replace=False, p=w / w.sum())]])
    for _ in range(iters):
        lab = cKDTree(C).query(X)[1]
        sw = np.bincount(lab, weights=w, minlength=len(C))
        move = sw > 0
        move[: len(fixed)] = False
        for dim in (0, 1):
            C[move, dim] = np.bincount(lab, weights=w * X[:, dim], minlength=len(C))[move] / sw[move]
    return C


def build(bbox, n_stops=60, forced=(), locked=(), headways=HEADWAYS, max_buses=0, max_km=0.0, max_route_min=0.0, spots=None):
    """Stop graph for bbox = (south, west, north, east).

    forced: planner stops [{"lat", "lng", "name", "demand"}], kept but moved to the nearest place a bus may stop.
    locked: planner routes, each a list of such stops; returned as stop-index routes in net["locked"].
    spots: road nodes for the AI's key stops (a design whose stops were moved, improve_stops); None = place them.
    """
    R, (lat, lng, pop, year) = roads(), population()
    s, w, n, e = bbox
    inb = (lat >= s) & (lat <= n) & (lng >= w) & (lng <= e)
    hP, hpop = _m(lat[inb], lng[inb]), pop[inb]
    blat, blng, bppl = people()  # the same people at their buildings: walking, coverage and stop spots use these
    inb = (blat >= s) & (blat <= n) & (blng >= w) & (blng <= e)
    bP, bpop = _m(blat[inb], blng[inb]), bppl[inb]
    if hpop.sum() <= 0 or bpop.sum() <= 0:
        raise ValueError("nobody lives in this area")
    pl = [x for x in places() if s <= x["lat"] <= n and w <= x["lng"] <= e]
    plP = _m([x["lat"] for x in pl], [x["lng"] for x in pl])
    plw = np.array([x["trips"] for x in pl], float)
    planner = list(forced) + [x for r in locked for x in r]
    # Walkers: residents at their homes and each place's visitors at its door, in residents' trip ends (a resident starts
    # TRIP_RATE trips a day and ends TRIP_RATE * qbar of them; a place ends plw * qbar, qbar = trips per unit attraction).
    qbar = TRIP_RATE * bpop.sum() / (TRIP_RATE * bpop.sum() + plw.sum())
    wP, wpop = np.vstack([bP, np.reshape(plP, (-1, 2))]), np.r_[bpop, plw * qbar / (TRIP_RATE * (1 + qbar))]

    # 1. Key stops: where the trips start and end. A planner's stop goes to the nearest road spot a bus may stop on; an
    # AI stop goes to the spot (of the 40 nearest within 1 km) where the people around it walk least, judged from their
    # homes and the doors of the places they go to, the way an engineer would pick it on site.
    C = _m([x["lat"] for x in planner], [x["lng"] for x in planner]).reshape(-1, 2)  # planner stops: fixed centres
    if spots is None:
        C = _kmeans(np.vstack([hP, plP]), np.r_[hpop, plw / TRIP_RATE], n_stops, C)
    cand = np.flatnonzero(R["stop_ok"])
    tree = cKDTree(R["P"][cand])
    snap, j = tree.query(C) if len(C) else (np.zeros(0), np.zeros(0, int))
    if spots is None:
        near = cKDTree(C).query(wP)[1]
        for c in range(len(planner), len(C)):
            dd, oo = tree.query(C[c], k=40, distance_upper_bound=1000)
            opts, mine = oo[np.isfinite(dd)], near == c
            if len(opts) > 1 and mine.any():
                walked = _walk_min(np.linalg.norm(wP[mine][:, None] - R["P"][cand[opts]][None], axis=2))
                j[c] = opts[int(np.argmin(wpop[mine] @ walked))]
    pick = np.r_[cand[j[: len(planner)]], cand[j[len(planner):]] if spots is None else np.asarray(spots, int)]
    first = np.sort(np.unique(pick, return_index=True)[1])
    nodes = pick[first]
    K, N = len(nodes), len(R["lat"])
    zone_of = np.full(N, -1)
    zone_of[nodes] = np.arange(K)
    planner_zone = zone_of[pick[: len(planner)]]

    # 2. Neighbouring key stops (their road catchments touch) and the road between them.
    _, _, src = dijkstra(R["G"], indices=nodes, min_only=True, return_predecessors=True, limit=60)
    cell = np.where(src >= 0, zone_of[np.maximum(src, 0)], -1)
    cu, cv = cell[R["u"]], cell[R["v"]]
    ok = (cu >= 0) & (cv >= 0) & (cu != cv)
    nb = defaultdict(set)
    for a, b in np.unique(np.sort(np.column_stack([cu[ok], cv[ok]]), 1), axis=0):
        nb[a].add(b)
        nb[b].add(a)
    T, paths = np.full((K, K), np.inf), {}
    for a in range(K):
        dist, pred = dijkstra(R["G"], indices=nodes[a], return_predecessors=True, limit=120)
        T[a] = dist[nodes]
        # Neighbours and neighbours-of-neighbours: a key stop on a side road must not force a detour into it.
        for b in {c for x in nb[a] | {a} for c in nb[x]}:
            if b > a and np.isfinite(dist[nodes[b]]):
                pth = [nodes[b]]
                while pth[-1] != nodes[a]:
                    pth.append(pred[pth[-1]])
                if not (zone_of[pth[1:-1]] >= 0).any():  # passing another key stop: a-x-b is the real link
                    paths[a, b] = np.array(pth[::-1])

    # 3. Intermediate stops along each link: on stop-able road, spaced by how many people live there.
    ptree, ppl = _people_tree()
    reg, mids, km = set(nodes.tolist()), {}, np.full((K, K), np.inf)
    for (a, b), pth in sorted(paths.items()):
        cum = np.r_[0, np.cumsum(np.asarray(R["K"][pth[:-1], pth[1:]]).ravel())] * 1000
        km[a, b] = km[b, a] = cum[-1] / 1000
        nearby = np.array([ppl[ix].sum() for ix in ptree.query_ball_point(R["P"][pth], PEOPLE_NEAR_M)])
        last, mids[a, b] = 0.0, []
        for k in range(1, len(pth) - 1):
            v, gap = int(pth[k]), next(g for p_, g in SPACING_M if nearby[k] >= p_)
            if v in reg or (R["stop_ok"][v] and nearby[k] >= MIN_PEOPLE and cum[k] - last >= gap and cum[-1] - cum[k] >= gap / 2):
                reg.add(v)
                mids[a, b].append(v)
                last = cum[k]
    time, n_mid = np.full((K, K), np.inf), np.zeros((K, K), int)
    for (a, b), m in mids.items():
        time[a, b], time[b, a] = T[a, b] + DWELL_MIN * len(m), T[b, a] + DWELL_MIN * len(m)
        n_mid[a, b] = n_mid[b, a] = len(m)
    # Key stops are mid-block: leaving towards c by the same road node it arrived on from a is a U-turn at b.
    exits = defaultdict(list)  # key stop -> (neighbour, first road node towards it)
    for (a, b), pth in paths.items():
        exits[a].append((b, pth[1]))
        exits[b].append((a, pth[-2]))
    uturn = frozenset((a, b, c) for b, ex in exits.items() for a, x in ex for c, y in ex if a != c and x == y)

    # 4. Trips: residents produce TRIP_RATE trips a day, spread over destinations by attraction / travel time.
    zP = R["P"][nodes]
    zone_m, zone = cKDTree(zP).query(bP)
    zpop = np.bincount(zone, weights=bpop, minlength=K)

    # 5. Walking: a trip end walks between the key stop and the home or the place's door (walkers wP, wpop), or the
    # nearest stop on a link that ends there once a bus stops on that link. walk[z] = minutes per trip end in zone z;
    # walk_save[z, x] = minutes less while link (z, x) has a stopping bus. ponytail: each walker counts only its nearest
    # link stop, so serving a second nearby link saves nothing more; a per-walker walk to the nearest served stop would
    # cost a KD-tree query per evaluation. Before 2026-10-10 only homes walked: a stop 1 km from a hospital cost nothing.
    wm, wz = cKDTree(zP).query(wP)
    walk = np.bincount(wz, weights=wpop * _walk_min(wm), minlength=K)
    walk_save, cand = np.zeros((K, K)), defaultdict(list)  # zone -> [(other end, stop)] on the links ending there
    for (a, b), ms in mids.items():
        for v in ms:
            cand[a].append((b, v))
            cand[b].append((a, v))
    for z, c in cand.items():
        h = np.flatnonzero(wz == z)
        if len(h):
            d, j = cKDTree(R["P"][[v for _, v in c]]).query(wP[h])
            np.add.at(walk_save[z], [c[i][0] for i in j], wpop[h] * np.maximum(_walk_min(wm[h]) - _walk_min(d), 0))
    res = np.maximum(np.bincount(wz, weights=wpop, minlength=K), 1e-9)
    walk, walk_save = walk / res, walk_save / res[:, None]
    np.add.at(zpop, planner_zone[: len(forced)], [float(x.get("demand", 0)) for x in forced])
    attr = TRIP_RATE * zpop
    if pl:
        np.add.at(attr, cKDTree(zP).query(plP)[1], plw)
    f = np.where(np.isfinite(T), np.exp(-BETA * np.where(np.isfinite(T), T, 0)), 0)
    np.fill_diagonal(f, 0)
    share = attr[None] * f
    den = share.sum(1, keepdims=True)
    od = TRIP_RATE * zpop[:, None] * np.divide(share, den, out=np.zeros_like(share), where=den > 0)

    p = Problem(time=time, od=od, xy=np.column_stack([R["lng"][nodes], R["lat"][nodes]]), km=km, min_len=2, max_len=15,
                headways=tuple(headways), mids=n_mid, walk=walk, walk_save=walk_save, walk_weight=WALK_WEIGHT, vot=VOT_RM_H / 60, unserved_cost=UNSERVED_RM,
                transfer_min=TRANSFER_MIN, max_transfers=MAX_TRANSFERS,
                max_buses=max_buses, max_km=max_km, max_route_min=max_route_min, max_detour=MAX_DETOUR, uturn=uturn)

    # Planner routes: consecutive locked stops joined by the quickest stop-graph path.
    _, pred = routes.quickest(p)
    lk, i = [], len(forced)
    for r in locked:
        z = planner_zone[i: i + len(r)].tolist()
        i += len(r)
        seq = [z[0]]
        for a, b in zip(z, z[1:]):
            if a != b and pred[a, b] >= 0:
                seq += routes.path(pred, a, b)[1:]
        lk.append(seq)

    named = [x for x in pl if x["name"]]
    near = cKDTree(_m([x["lat"] for x in named], [x["lng"] for x in named])).query(zP)[1] if named else None
    names = []
    for z, v in enumerate(nodes):
        given = [x.get("name") for x, pz in zip(planner, planner_zone) if pz == z and x.get("name")]
        close = near is not None and np.hypot(*(zP[z] - _m([named[near[z]]["lat"]], [named[near[z]]["lng"]])[0])) < 300
        names.append(given[0] if given else named[near[z]]["name"] if close else str(R["node_name"][v]) or f"Stop {z + 1}")
    net = {"nodes": nodes, "spots": [int(v) for v in pick[len(planner):]], "paths": paths, "mids": mids, "names": names,
           "locked": lk, "hP": bP, "hpop": bpop, "wP": wP, "wpop": wpop,
           "planner_snap_m": snap[: len(planner)].round().tolist(), "people": float(bpop.sum()), "dosm_year": year,
           "trips": float(od.sum())}
    return p, net


def describe(p, net, rs, express=None):
    """Map-ready routes (road geometry + stop sequence) and stops (name, road class, key or intermediate).
    express: sim metrics["express"], links where the route skips the stops between key stops."""
    R, nodes = roads(), net["nodes"]
    stops, out = {}, []

    def stop(v, z=None):
        if v not in stops:
            stops[v] = {"id": f"s{v}", "lat": round(float(R["lat"][v]), 6), "lng": round(float(R["lng"][v]), 6),
                        "name": net["names"][z] if z is not None else str(R["node_name"][v]) or "Unnamed road",
                        "road": f"{R['node_cls'][v]} road", "key": z is not None, "routes": []}
        return stops[v]

    for i, r in enumerate(rs):
        seq, line = [stop(int(nodes[r[0]]), r[0])], []
        for k, (a, b) in enumerate(zip(r, r[1:])):
            pth, mid = net["paths"][min(a, b), max(a, b)], net["mids"][min(a, b), max(a, b)]
            if a > b:
                pth, mid = pth[::-1], mid[::-1]
            line += np.column_stack([R["lat"][pth], R["lng"][pth]]).round(6).tolist()[1 if line else 0:]
            seq += [stop(int(v)) for v in mid if not (express and express[i][k])] + [stop(int(nodes[b]), b)]
        for x in seq:
            if i not in x["routes"]:
                x["routes"].append(i)
        out.append({"stops": [x["id"] for x in seq], "path": line})
    served = np.array([[s["lat"], s["lng"]] for s in stops.values()]).reshape(-1, 2)
    near = cKDTree(_m(served[:, 0], served[:, 1])).query(net["hP"])[0] if len(served) else np.inf
    covered = float(net["hpop"][near <= WALK_M].sum() / net["people"])
    return {"routes": out, "stops": list(stops.values()), "coverage": covered}


def _repair(p, rs):
    """Routes kept running after a key stop moved: a link that no longer exists becomes the quickest path round it; a
    route cut where it would come back to one of its own stops."""
    D, pred = routes.quickest(p)
    out = []
    for r in rs:
        seq = [r[0]]
        for b in r[1:]:  # from the last stop kept: a stop the move cut off is skipped, not joined to the next one
            a = seq[-1]
            if np.isfinite(p.time[a, b]):
                seq.append(b)
            elif np.isfinite(D[a, b]):
                seq += routes.path(pred, a, b)[1:]
        keep = []
        for x in seq:
            if x in keep:
                break
            keep.append(x)
        keep = routes.trim(p, routes.unturn(p, keep))
        if len(keep) >= 2:
            out.append(keep)
    return out


def improve_stops(make, spots, rs, n_routes, stop=lambda: False, seconds=10.0, tries=3):
    """An engineer revising a design on site: move one AI key stop onto another road within 1 km (the best spot for its
    people's walk on each of the `tries` nearest such roads), re-plan the routes around it, keep the move if the whole
    network gets cheaper: a main road may be worth a longer walk, a side road the other way round. A pass tries every
    AI stop, the one whose people walk most first; passes repeat until one keeps nothing (or stop() is true).
    make(spots) -> (p, net) builds the design for those spots; rs are its routes without the planner's locked ones.
    Yields (score, spots, routes, p, net, kept) after every try. ponytail: a full rebuild per try (~3 s for the whole
    state); an incremental rebuild of the moved stop's links would be faster."""
    import rl  # here: rl imports this module

    R = roads()
    cand = np.flatnonzero(R["stop_ok"])
    tree = cKDTree(R["P"][cand])
    spots, rs = [int(v) for v in spots], [list(r) for r in rs]
    p, net = make(spots)
    best = evaluate(p, net["locked"] + rs)["score"]

    def catchments():  # which key stop each home or place door walks to, and how long its people walk
        zm, zone = cKDTree(R["P"][net["nodes"]]).query(net["wP"])
        return zone, np.bincount(zone, weights=net["wpop"] * _walk_min(zm), minlength=len(net["nodes"]))

    while True:
        moved = False
        zone, burden = catchments()
        pos = {int(v): z for z, v in enumerate(net["nodes"])}
        for v in sorted((v for v in spots if v in pos), key=lambda v: -burden[pos[v]]):  # each AI stop once per pass
            if v not in spots:
                continue
            i, z = spots.index(v), int(np.flatnonzero(net["nodes"] == v)[0])
            mine, used = zone == z, set(int(x) for x in net["nodes"])
            dd, oo = tree.query(R["P"][v], k=40, distance_upper_bound=1000)
            opts = [int(x) for x in cand[oo[np.isfinite(dd)]] if int(x) not in used]
            if not opts or not mine.any():
                continue
            walked = net["wpop"][mine] @ _walk_min(np.linalg.norm(net["wP"][mine][:, None] - R["P"][opts][None], axis=2))
            road, picks = lambda x: (str(R["node_name"][x]), str(R["node_cls"][x])), {}
            for x in np.argsort(walked):  # the stop already has the best spot on its own road: try the other roads
                if road(opts[x]) != road(v):
                    picks.setdefault(road(opts[x]), opts[x])
            for c in list(picks.values())[:tries]:
                if stop():
                    return
                trial = spots[:i] + [c] + spots[i + 1:]
                p2, net2 = make(trial)
                if not np.array_equal(net2["nodes"], np.where(net["nodes"] == v, c, net["nodes"])):
                    continue  # the move merged stops: their numbers no longer match the routes
                rs2 = _repair(p2, rs)
                s2 = evaluate(p2, net2["locked"] + rs2)["score"]
                if best <= s2 < 1.02 * best:  # close: re-plan the routes around the moved stop before judging
                    try:
                        seeds = [rs2] + rl.samples(p2, n_routes, 4, net2["locked"])
                    except FileNotFoundError:  # no trained policy
                        seeds = [rs2]
                    rs2 = routes.ga(p2, n_routes, net2["locked"], seconds, seeds=seeds)[len(net2["locked"]):]
                    s2 = evaluate(p2, net2["locked"] + rs2)["score"]
                kept = s2 < best
                if kept:
                    best, spots, rs, p, net, moved = s2, trial, rs2, p2, net2, True
                    zone, burden = catchments()  # homes now walk to the moved stop
                yield best, spots, rs, p, net, kept
                if kept:
                    break
        if not moved:
            return


if __name__ == "__main__":
    import time as clock

    cut = np.full((3, 3), np.inf)
    cut[1, 2] = cut[2, 1] = 2.0  # stop 0 cut off from 1 and 2 (a moved stop split the road network)
    assert _repair(Problem(time=cut, od=np.ones((3, 3)), xy=np.zeros((3, 2))), [[0, 1, 2], [1, 2, 0]]) == [[1, 2]]

    t = clock.time()
    R, (lat, lng, pop, year) = roads(), population()
    print(f"roads: {len(R['lat'])} nodes, {len(R['u'])} edges, {R['stop_ok'].sum()} stop-able; "
          f"people: {pop.sum():,.0f} (DOSM {year}); places: {len(places())}; {clock.time() - t:.0f}s")
    assert 1.5e6 < pop.sum() < 2.5e6, "Kelantan has ~1.9 million people"
    assert not R["stop_ok"][np.isin(R["node_cls"], ["motorway", "motorway_link"])].any(), "never stop on an expressway"
    blat, blng, bppl = people()
    print(f"buildings: {len(buildings()[0]):,} seen from space; people placed on {len(bppl):,} points")
    assert abs(bppl.sum() - pop.sum()) < 1, "placing people on buildings keeps every person"
    hosp = sorted(x["trips"] for x in places() if x["kind"] == "hospital")
    assert hosp[-1] > 5 * hosp[0], "hospitals differ by their district's real beds"
    size = {x["name"]: x["trips"] for x in places() if x["kind"] == "mall"}
    assert size["AEON Mall Kota Bharu"] > 5 * size["Bazar Tuan Padang"], "malls differ by their footprint"

    forced = [{"lat": 6.1254, "lng": 102.2386, "name": "Kota Bharu Centre", "demand": 5000}]
    p, net = build((6.0, 102.15, 6.2, 102.35), n_stops=30, forced=forced, locked=[forced + [{"lat": 6.099, "lng": 102.283}]])
    assert net["names"][0] == "Kota Bharu Centre" and net["planner_snap_m"][0] < 500
    assert net["locked"][0][0] == 0 and len(net["locked"][0]) >= 2 and routes.valid(p, net["locked"])
    rs = routes.ga(p, 6, locked=net["locked"], seconds=5)
    d = describe(p, net, rs)
    on_road = {f"s{v}" for v in np.flatnonzero(R["stop_ok"])}
    assert all(s["id"] in on_road for s in d["stops"]), "every stop is on a road where buses may stop"
    m = evaluate(p, rs, detail=True)
    print(f"Kota Bharu: {len(p.od)} key stops, {len(d['stops'])} stops, {net['trips']:,.0f} trips/day, "
          f"coverage {d['coverage']:.0%}, ATT {m['att']:.1f} min, fleet {m['fleet']}, unserved {m['unserved']:.1%}")
