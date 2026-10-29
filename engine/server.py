import json
import re
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import shapely
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

import export
import kelantan
import routes
from sim import HEADWAYS, evaluate

RESULTS = Path(__file__).resolve().parent.parent / "results"
RL_PROGRESS = Path(__file__).resolve().parent / "models" / "route_gnn.json"  # written by rl.train while the AI trains
RL_LIVE = Path(tempfile.gettempdir()) / "laju-rl-live.json"  # rl.train: its design of the Kelantan network right now
KELANTAN = (4.5, 101.3, 6.3, 102.7)  # south, west, north, east
CHECKPOINT_S = 60  # a training run is saved this often (and when paused): a crash loses at most this much
AI_EVERY_S = 600  # Train adds fresh designs from the deep-RL policy this often: the run gains what the AI learns meanwhile
app = FastAPI(title="LAJU engine")
jobs = {}  # run id -> {"thread", "stop", "train", "output", "error"?}: trainings started by this process
start_lock = threading.Lock()


class Stop(BaseModel):
    lat: float = Field(ge=KELANTAN[0], le=KELANTAN[2])
    lng: float = Field(ge=KELANTAN[1], le=KELANTAN[3])
    name: str = Field("", max_length=100)
    demand: float = Field(0, ge=0, le=1_000_000)


Headway = Annotated[float, Field(ge=2, le=120)]


class DesignRequest(BaseModel):
    name: str = Field("", max_length=100)
    bbox: tuple[float, float, float, float]  # south, west, north, east: the planner's map view
    n_routes: int = Field(6, ge=1, le=60)
    n_stops: int = Field(60, ge=5, le=150)  # key stops the AI places
    optimizer: Literal["baseline", "ga", "rl", "rl+ga"] = "ga"
    seconds: float = Field(20, ge=1, le=600)
    headways: tuple[Headway, Headway, Headway] = HEADWAYS  # shuttle, local, inter-town: a bus at least this often
    max_buses: int = Field(0, ge=0, le=5000)
    max_km: float = Field(0, ge=0, le=10_000)
    max_route_min: float = Field(90, ge=0, le=600)  # longest one-way trip of a route, 0 = no limit
    forced: list[Stop] = Field([], max_length=200)
    locked: list[list[Stop]] = Field([], max_length=60)
    spots: list[int] = Field([], max_length=200)  # road nodes of the AI's key stops once "Move stops" moved them; [] = AI places


def _path(rid):
    if not re.fullmatch(r"run-\d{8}-\d{6}-\d{6}", rid):
        raise HTTPException(404, "no such run")
    f = RESULTS / f"{rid}.json"
    if not f.exists():
        raise HTTPException(404, "no such run")
    return f


def _save(f, d):
    """Write a run all or nothing: an engine stopped mid-save never leaves a broken file."""
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    for attempt in range(5):
        try:
            return tmp.replace(f)
        except PermissionError:  # Windows: a reader or OneDrive holds the file for a moment
            if attempt == 4:
                raise
            time.sleep(0.2 * 2 ** attempt)


def _problem(req):
    """The stop graph for a request: (sim.Problem, drawing data, number of routes, locked routes)."""
    s, w, n, e = req.bbox
    bbox = (max(s, KELANTAN[0]), max(w, KELANTAN[1]), min(n, KELANTAN[2]), min(e, KELANTAN[3]))
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise HTTPException(422, "the map view is outside Kelantan")
    if not all(0 <= v < len(kelantan.roads()["lat"]) for v in req.spots):
        raise HTTPException(422, "unknown stop spot")
    try:
        p, net = kelantan.build(bbox, req.n_stops, [x.model_dump() for x in req.forced],
                                [[x.model_dump() for x in r] for r in req.locked], req.headways, req.max_buses, req.max_km,
                                req.max_route_min, req.spots or None)
    except (ValueError, FileNotFoundError) as err:
        raise HTTPException(422, str(err))
    return p, net, max(req.n_routes, len(req.locked)), net["locked"]


def _routes(d, p, net, lk):
    """A saved run's free routes as key-stop indices on today's stop graph; routes it no longer allows are dropped."""
    key = {int(v): z for z, v in enumerate(net["nodes"])}
    now = [[key[int(s[1:])] for s in r["stops"] if int(s[1:]) in key] for r in d["output"]["routes"]][len(lk):]
    return [r for r in now if routes.valid(p, [r])]


def _output(p, net, rs, req):
    m = evaluate(p, rs, detail=True)
    return kelantan.describe(p, net, rs, m.get("express")) | {
        "metrics": m,
        "headways": list(req.headways), "max_route_min": req.max_route_min, "service_hours": p.service_hours,
        "periods": [list(x) for x in kelantan.PERIODS],  # (name, from, to hour): metrics["periods"] has a headway for each
        "people": net["people"], "trips": net["trips"], "dosm_year": net["dosm_year"],
        "planner_snap_m": net["planner_snap_m"],
        "model": {"lat": p.xy[:, 1].round(6).tolist(), "lng": p.xy[:, 0].round(6).tolist(), "od": p.od.round(2).tolist()},
    }


@app.get("/rl/training")
def rl_training(live: int = 0):
    """The deep-RL designer's own training (python rl.py <minutes>): best cost vs the baseline on maps it never trains on,
    and its design of the Kelantan network right now, when that is not the frame number `live` the website already has."""
    try:
        d = json.loads(RL_PROGRESS.read_text())
    except (FileNotFoundError, PermissionError, ValueError):
        d = {"running": False, "history": [], "updated": 0}
    try:
        v = json.loads(RL_LIVE.read_text())
        d["live"] = v if v["n"] != live else {"n": v["n"]}
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return d | {"running": d["running"] and time.time() - d["updated"] < 300}  # a killed training stops counting as running


@app.get("/kelantan")
def boundary():
    return json.loads(shapely.to_geojson(kelantan.boundary().simplify(0.002)))


@app.post("/design")
def design(req: DesignRequest):
    p, net, k, lk = _problem(req)
    try:
        if req.optimizer == "baseline":
            rs = routes.baseline(p, k, lk)
        elif req.optimizer == "ga":
            rs = routes.ga(p, k, lk, seconds=req.seconds)
        else:
            import rl

            rs = rl.design(p, k, lk, req.seconds) if req.optimizer == "rl" else routes.ga(p, k, lk, req.seconds, seeds=rl.samples(p, k, 8, lk))
    except (ValueError, FileNotFoundError) as err:
        raise HTTPException(422, str(err))
    out = _output(p, net, rs, req)
    RESULTS.mkdir(exist_ok=True)
    rid = f"run-{datetime.now():%Y%m%d-%H%M%S-%f}"
    _save(RESULTS / f"{rid}.json", {"input": req.model_dump(), "output": out})
    return {"id": rid, **out}


@app.get("/runs")
def runs():
    out = []
    for f in sorted(RESULTS.glob("run-*.json"), reverse=True):
        d = json.loads(f.read_text())
        m = d["output"]["metrics"]
        out.append({"id": f.stem, "name": d["input"]["name"], "optimizer": d["input"]["optimizer"],
                    "routes": len(d["output"]["routes"]), "stops": len(d["output"]["stops"]),
                    "coverage": d["output"]["coverage"], "trained_s": d.get("train", {}).get("seconds", 0),
                    "training": f.stem in jobs and jobs[f.stem]["thread"].is_alive()}
                   | {k: m[k] for k in ("score", "att", "unserved", "d0", "fleet", "total_km")}
                   | {k: m.get(k) for k in ("riders", "covered")})  # score v3 runs only
    return out


@app.get("/runs/{rid}")
def run(rid: str):
    return json.loads(_path(rid).read_text())


@app.get("/runs/{rid}/geojson")
def run_geojson(rid: str):
    body = json.dumps(export.geojson(run(rid)["output"]))
    return Response(body, media_type="application/geo+json", headers={"Content-Disposition": f'attachment; filename="{rid}.geojson"'})


@app.get("/runs/{rid}/gtfs")
def run_gtfs(rid: str):
    body = export.gtfs(run(rid)["output"], rid)
    return Response(body, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{rid}-gtfs.zip"'})


@app.post("/runs/{rid}/simulate")
def simulate(rid: str, scale: float = 1.0):
    import traffic

    if not 0 < scale <= 10:
        raise HTTPException(422, "scale must be in (0, 10]")
    if rid in jobs and jobs[rid]["thread"].is_alive():
        raise HTTPException(409, "pause training first: its next save would overwrite the simulation")
    f = _path(rid)
    d = json.loads(f.read_text())
    d["output"]["sumo"] = traffic.simulate(d["output"], scale)
    _save(f, d)
    return d["output"]["sumo"]


def _ai(p, net, k, lk, seed):
    """GA seeds from the deep-RL designer: 16 route sets from its saved policy, new ones for each seed, plus the best
    design python rl.py has drawn live for this map while it trains (RL_LIVE: the Kelantan network)."""
    out = []
    try:
        best = json.loads(RL_LIVE.read_text())["best"]
        if {x["id"] for x in best["stops"] if x["key"]} <= {f"s{v}" for v in net["nodes"]}:  # drawn on these key stops
            out.append(_routes({"output": best}, p, net, lk))
    except (FileNotFoundError, PermissionError, ValueError, KeyError):
        pass
    try:
        import rl

        return out + rl.samples(p, k, 16, lk, seed)
    except FileNotFoundError:  # no trained policy yet
        return out


def _train(f, d, p, net, k, lk, req, j):
    """Training thread: the genetic algorithm keeps improving the run's routes until j["stop"] is set.
    Starts from the run's current routes, its saved population when resuming, and the deep-RL policy's designs, which it
    asks again every AI_EVERY_S: the policy keeps getting better while python rl.py trains."""
    tr = d["train"]
    now = _routes(d, p, net, lk)
    seeds = tr.get("population", []) + [now]
    h = tr["history"]  # [training seconds, best score]: generation 0, then every improvement
    best = evaluate(p, lk + now)["score"] if h else float("inf")  # by today's rules, which may have changed since
    if h and abs(h[-1][1] - best) > 1e-6 * best:  # they did: the curve shows the jump, the panel today's numbers
        h.append([round(tr["seconds"], 1), best])
        d["output"] = j["output"] = _output(p, net, lk + now, req)
    t0, s0, saved = time.time(), tr["seconds"], time.time()
    try:
        while not j["stop"].is_set():
            asked = time.time()
            ai = _ai(p, net, k, lk, tr["generation"])
            for scored in routes.evolve(p, k, lk, seed=tr["generation"], seeds=[rs for rs in seeds + ai if routes.valid(p, rs)]):
                tr["generation"] += 1
                tr["seconds"] = s0 + time.time() - t0
                tr["population"] = [rs for _, rs in scored]
                if scored[0][0] < best:
                    best = scored[0][0]
                    d["output"] = j["output"] = _output(p, net, lk + scored[0][1], req)  # before the history: see _status
                    h.append([round(tr["seconds"], 1), best])
                if j["stop"].is_set() or time.time() - asked > AI_EVERY_S:
                    break
                if time.time() - saved > CHECKPOINT_S:
                    _save(f, d)
                    saved = time.time()
            seeds = tr["population"]
    except Exception as err:  # keep what it learned and tell the planner
        j["error"] = f"training stopped: {err}"
        raise
    finally:
        _save(f, d)


def _improve(f, d, p, net, k, lk, req, j):
    """Stop-moving thread: kelantan.improve_stops moves the AI's key stops onto better roads and re-plans the routes,
    until paused or a whole pass keeps nothing. Each kept move is saved with its stop spots, so the run rebuilds moved."""
    tr, h = d["train"], d["train"]["history"]
    now = _routes(d, p, net, lk)
    s = evaluate(p, lk + now)["score"]  # by today's rules, which may have changed since
    if not h or abs(h[-1][1] - s) > 1e-6 * s:  # they did: the curve shows the jump, the panel today's numbers
        h.append([round(tr["seconds"], 1), s])
        d["output"] = j["output"] = _output(p, net, lk + now, req)
    t0, s0 = time.time(), tr["seconds"]

    def make(spots):
        return _problem(req.model_copy(update={"spots": spots}))[:2]

    try:
        for s, spots, rs, p2, net2, kept in kelantan.improve_stops(make, net["spots"], now, k, stop=j["stop"].is_set):
            tr["generation"] += 1
            tr["seconds"] = s0 + time.time() - t0
            if kept:
                req = req.model_copy(update={"spots": spots})
                d["input"] = req.model_dump()
                d["output"] = j["output"] = _output(p2, net2, net2["locked"] + rs, req)  # before the history: see _status
                h.append([round(tr["seconds"], 1), s])
                _save(f, d)
    except Exception as err:  # keep what it found and tell the planner
        j["error"] = f"moving stops stopped: {err}"
        raise
    finally:
        tr["seconds"] = s0 + time.time() - t0
        _save(f, d)


def _status(rid, since=None):
    """Training progress, plus the best design so far if the client asks `since` (its history length) and it is newer."""
    j = jobs.get(rid)
    d = j or json.loads(_path(rid).read_text())
    tr = d.get("train", {})
    h = tr.get("history", [])
    n = len(h)  # read before the design, which the trainer writes first: the design sent is never older than h[:n]
    out = {"running": bool(j and j["thread"].is_alive()), "mode": j["mode"] if j else "routes", "generation": tr.get("generation", 0),
           "seconds": tr.get("seconds", 0.0), "history": h[:n], "error": j and j.get("error")}
    return out | ({"design": d["output"]} if since is not None and n > since else {})


@app.post("/runs/{rid}/train")
def train(rid: str, stops: bool = False):
    """Start or resume training a saved run: the GA re-plans its routes or, with stops, the AI also moves its key stops
    onto better roads (slow: a rebuild per try). It runs until paused, however long that is."""
    with start_lock:
        if rid not in jobs or not jobs[rid]["thread"].is_alive():
            f = _path(rid)
            d = json.loads(f.read_text())
            req = DesignRequest(**d["input"])
            p, net, k, lk = _problem(req)
            if k <= len(lk):
                raise HTTPException(422, "every route is locked: nothing to train")
            d["input"] = req.model_dump()  # older runs get today's defaults (the route-time cap) written down
            d.setdefault("train", {"generation": 0, "seconds": 0.0, "history": []})
            j = {"stop": threading.Event(), "train": d["train"], "output": d["output"], "mode": "stops" if stops else "routes"}
            j["thread"] = threading.Thread(target=_improve if stops else _train, args=(f, d, p, net, k, lk, req, j), daemon=True)
            jobs[rid] = j
            j["thread"].start()
    return _status(rid)


@app.get("/runs/{rid}/train")
def train_status(rid: str, since: int | None = None):
    return _status(rid, since)


@app.post("/runs/{rid}/pause")
def pause(rid: str, since: int | None = None):
    """Stop training; the run is saved with its best design and population, ready to resume."""
    if rid in jobs:
        jobs[rid]["stop"].set()
        jobs[rid]["thread"].join(60)
    return _status(rid, since)
