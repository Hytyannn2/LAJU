"""Experiment records, health and benchmarks (plan.md T-A1, T-A4). Every training run and every design leaves one JSON
record in results/, with what it takes to repeat and compare it: seeds, configuration, code and score version, machine,
search effort, result. The commands below leave records too.

    python experiments.py                         -> self-check: "ok ..."
    python experiments.py health [run name]       -> the trained policy's health on the state map (plan.md 8.2), ~6 min:
                                                     how random it still is, how alike its designs are, where the score's
                                                     spread comes from, how much its training signal agrees with itself
    python experiments.py bench --methods rl+ga,sample --seeds 10 --seconds 300 [--generations 84] [--maps state,val]
                                                  -> each method from each seed on the maps, a table with intervals
    python experiments.py sweep [--routes 12,16,20,24,32] [--buses 64,100,200] [--seconds 300]
                                                  -> today's design per route count x fleet, one exam row each
    python experiments.py compare A.json B.json   -> is B better than A: paired Wilcoxon and a 95% bootstrap interval
    python experiments.py fair "Phase 5: 64 buses" ...  -> BAS.MY's timetable beside saved designs: riders, vehicle-hours,
                                                     bus-km, riders per bus-km (plan.md 6.5; E1 / E2 resources)
"""
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import sim

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIELDS = ("kind", "start", "end", "seconds", "git", "dirty", "score_version", "python", "torch", "cpu", "cores", "config",
          "seeds", "evals", "result")


def _git(*args):
    """git's answer in this repository, or None (no git on the path, not a repository)."""
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=HERE, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu():
    """The processor's name as Windows shows it (platform.processor() gives only its family), else platform's."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or None


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def record(kind, config, result, seeds=None, evals=None, start=None, folder=RESULTS):
    """One experiment record, results/exp-<time>.json (FIELDS). config: every argument of the run; start: its
    time.time() at the start; evals: sim.evaluate calls it made (default: this process's sim.EVALS; worker processes
    count their own and the caller adds them up). -> the record's path"""
    import server  # its retrying all-or-nothing writer: OneDrive holds a file for a moment

    now, status = time.time(), _git("status", "--porcelain")
    d = {"kind": kind, "start": datetime.fromtimestamp(start or now).isoformat(timespec="seconds"),
         "end": datetime.fromtimestamp(now).isoformat(timespec="seconds"), "seconds": round(now - (start or now), 1),
         "git": _git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status),
         "score_version": sim.SCORE_VERSION, "python": platform.python_version(), "torch": _version("torch"),
         "cpu": _cpu(), "cores": os.cpu_count(), "config": config, "seeds": seeds,
         "evals": sim.EVALS if evals is None else int(evals), "result": result}
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"exp-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    server._save(f, json.loads(json.dumps(d, default=str)))  # paths and numpy numbers as text
    print("record:", f, flush=True)
    return f


def iqm(x):
    """Interquartile mean: the mean of the middle half (Agarwal et al. 2021), robust to one lucky or unlucky seed."""
    x = np.sort(np.asarray(x, float))
    q = len(x) // 4
    return float(x[q: len(x) - q].mean()) if len(x) else math.nan


def interval(groups, f=iqm, n=2000, seed=0):
    """95% stratified bootstrap interval of f over all groups' values, each group (a map's seeds) resampled within
    itself."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g, float) for g in groups if len(g)]
    stats = [f(np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups])) for _ in range(n)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def _links(rs):
    return {(min(a, b), max(a, b)) for r in rs for a, b in zip(r, r[1:])}


def jaccard(sets):
    """Mean pairwise Jaccard similarity of route sets' undirected links: 1 = every set the same network."""
    links = {x: i for i, x in enumerate(sorted(set().union(*(_links(rs) for rs in sets))))}
    A = np.zeros((len(sets), len(links)))
    for i, rs in enumerate(sets):
        A[i, [links[x] for x in _links(rs)]] = 1
    both = A @ A.T
    J = both / np.maximum(A.sum(1)[:, None] + A.sum(1)[None] - both, 1e-9)
    return float(J[np.triu_indices(len(sets), 1)].mean())


def _state(name=None):
    """(Problem, routes to design, locked routes) of a saved run's map: rl.TARGET's (the state map) by default."""
    import rl
    import server

    d = rl._saved(name) if name else None
    if name and d is None:
        raise SystemExit(f"no saved run named {name!r}")
    p, _, k, lk = server._problem(server.DesignRequest(**d) if d else rl.target())
    return p, k, lk


def _policy(weights=None):
    import torch

    import rl

    if weights is None:
        return rl._load()
    net = rl.Net()
    net.load_state_dict(torch.load(weights))
    return net.eval()


def health(name=None, weights=None, n=512, grads=6):
    """The policy's health on a map (plan.md 8.2 and 20.7). Per decision of 64 designs: how sure it is (entropy, the top
    action's probability, allowed actions). Over n sampled designs: how alike they are (link Jaccard); where the spread
    of their log score comes from (riders, or the penalty multipliers); how many run over the bus budget or miss the
    coverage floor; the expected best of 8, 64 and 256. Over `grads` independent training batches: the cosine between
    their gradients (how much the training signal agrees with itself) and their norm before clipping. -> dict, recorded"""
    import types

    import torch

    import rl

    t0, e0 = time.time(), sim.EVALS
    p, k, lk = _state(name)
    net, G = _policy(weights), rl.Graph(p, k, lk)
    sets, ent, top, allowed = [], [], [], []
    with torch.no_grad():
        H = net.encode(G)
        for b in range(n // 64):
            got, trace = rl.rollout(net, G, 64, gen=torch.Generator().manual_seed(b), H=H)
            sets += got
            for x, end, m, a, live in trace if b == 0 else ():
                logp = torch.log_softmax(net(H, G, x, end).masked_fill(~m, -1e9), -1)[live]
                ent += (-(logp.exp() * logp).sum(-1)).tolist()
                top += logp.exp().max(-1).values.tolist()
                allowed += m[live].sum(-1).tolist()
    ms = [sim.evaluate(p, rs) for rs in sets]
    score = np.array([m["score"] for m in ms])
    mult = np.array([math.prod(m["penalties"].values()) for m in ms])
    rng = np.random.default_rng(0)
    real, norms, g = torch.nn.utils.clip_grad_norm_, [], []
    torch.nn.utils.clip_grad_norm_ = lambda ps, mx: norms.append(float(real(ps, math.inf)))  # measure only, never clip
    try:
        for i in range(grads):  # rl.learn's own loss, without the optimiser step
            rl.learn(net, types.SimpleNamespace(zero_grad=net.zero_grad, step=lambda: None), G, 16,
                     torch.Generator().manual_seed(100 + i))
            g.append(torch.cat([q.grad.flatten() for q in net.parameters() if q.grad is not None]).clone())
    finally:
        torch.nn.utils.clip_grad_norm_ = real
    cos = [float(torch.nn.functional.cosine_similarity(a, b, 0)) for i, a in enumerate(g) for b in g[i + 1:]]
    best = int(score.argmin())
    out = {"map": name or rl.TARGET, "weights": str(weights or rl.MODEL), "samples": len(sets),
           "decisions": len(ent) / 64, "allowed": float(np.mean(allowed)), "entropy": float(np.mean(ent)),
           "entropy_median": float(np.median(ent)), "choices": float(np.exp(np.mean(ent))), "top": float(np.mean(top)),
           "top_over_0.9": float(np.mean(np.array(top) > 0.9)), "jaccard": jaccard(sets),
           "log_score_sd": float(np.log(score).std()), "riders_sd": float(np.log(score / mult).std()),
           "penalties_sd": float(np.log(mult).std()), "over_budget": float(np.mean([m["over"] > 0 for m in ms])),
           "under_floor": float(np.mean([m["covered"] < p.min_cover for m in ms])) if p.min_cover else 0.0,
           "best": float(score[best]), "mean": float(score.mean()),
           "best_of": {N: float(np.mean([score[rng.integers(len(score), size=N)].min() for _ in range(2000)]))
                       for N in (8, 64, 256)},
           "penalties_of_best": ms[best]["penalties"], "riders_of_best": ms[best]["riders"],
           "grad_cos": float(np.mean(cos)), "grad_cos_range": [min(cos), max(cos)], "grad_norm": float(np.mean(norms))}
    if hasattr(sim, "key"):  # plan.md T-O1: how the feasibility-first order agrees with the score's
        from scipy.stats import spearmanr

        order = sorted(range(len(ms)), key=lambda i: sim.key(ms[i]))
        rank = np.empty(len(ms))
        rank[order] = np.arange(len(ms))
        top_key = ms[order[0]]
        out |= {"feasible": float(np.mean([m["violation"] == 0 for m in ms])),
                "key_vs_score_spearman": float(spearmanr(rank, score).statistic),
                "best_by_key": {"score": top_key["score"], "riders": top_key["riders"], "violation": top_key["violation"],
                                "violations": top_key["violations"]}}
    record("health", {"name": name, "weights": weights, "n": n, "grads": grads}, out,
           seeds={"samples": list(range(n // 64)), "grads": [100 + i for i in range(grads)], "best_of": 0},
           evals=sim.EVALS - e0, start=t0)
    return out


def _sample(p, k, lk, seed, seconds, generations=0, t0=None):
    """The trained policy's best design of everything it draws for `seconds` (or `generations` batches of 16), the
    first one greedy. -> (routes, curve [[s, batch, best score]], batches)"""
    import torch

    import rl

    t0, net, G, curve, b = t0 or time.time(), rl._load(), rl.Graph(p, k, lk), [], 0
    with torch.no_grad():
        H = net.encode(G)
        while True:
            sets = rl.rollout(net, G, 1, greedy=True, H=H)[0] if b == 0 else []
            sets += rl.rollout(net, G, 16 - len(sets), gen=torch.Generator().manual_seed(1000 * seed + b), H=H)[0]
            for rs in sets:
                s = sim.evaluate(p, rs)["score"]
                if not curve or s < curve[-1][2]:
                    curve.append([round(time.time() - t0, 1), b, s, rs])
            b += 1
            if (b >= generations) if generations else (time.time() - t0 >= seconds):
                return curve[-1].pop(), [c[:3] for c in curve], b


def _ga(p, k, lk, seed, seconds, generations=0, t0=None, seeds=()):
    """routes.evolve from `seeds` for `seconds` (or `generations`). -> (routes, curve, generations)"""
    import routes

    t0, curve = t0 or time.time(), []
    for g, scored in enumerate(routes.evolve(p, k, lk, seed=seed, seeds=seeds)):
        if not curve or scored[0][0] < curve[-1][2]:
            curve.append([round(time.time() - t0, 1), g, scored[0][0]])
        if (g >= generations) if generations else (time.time() - t0 >= seconds):
            return [list(r) for r in lk] + scored[0][1], curve, g


def _rlga(p, k, lk, seed, seconds, generations=0):
    """Today's product (server.design, rl+ga): 8 policy samples, the first greedy, seed the GA. Their drawing counts in
    the time."""
    import rl

    t0 = time.time()
    return _ga(p, k, lk, seed, seconds, generations, t0, seeds=rl.samples(p, k, 8, lk, seed))


METHODS = {  # name: f(p, n_routes, locked, seed, seconds, generations) -> (routes, curve, generations)
    "baseline": lambda p, k, lk, seed, seconds, generations=0: (__import__("routes").baseline(p, k, lk), [], 0),
    "ga": _ga,          # the GA from scratch
    "rl+ga": _rlga,     # today's product
    "sample": _sample,  # the policy alone, best of everything drawn in the time
}


def bench(methods, seeds=10, seconds=300.0, generations=0, maps=("state",)):
    """Every method from every seed on every map ("state": rl.TARGET's; "val": rl.train's 6 held-out Kelantan views), at
    the same wall-clock (or the same generations), on this machine. -> runs, recorded with a table: per method and map
    the interquartile mean score over the seeds and its 95% bootstrap interval."""
    import rl

    t0, e0, todo = time.time(), sim.EVALS, []
    for name in maps:
        todo += [("state", _state())] if name == "state" else [
            (f"val{i}", (p, k, [])) for i, (p, k) in enumerate(rl.kelantan_cities(6, seed=999))]
    runs = []
    for where, (p, k, lk) in todo:
        for method in methods:
            for seed in range(seeds):
                s0, c0 = time.time(), sim.EVALS
                rs, curve, gens = METHODS[method](p, k, lk, seed, seconds, generations)
                m = sim.evaluate(p, rs)
                runs.append({"method": method, "map": where, "seed": seed, "score": m["score"], "riders": m["riders"],
                             "covered": m["covered"], "fleet": m["fleet"], "over": m["over"], "evals": sim.EVALS - c0,
                             "seconds": round(time.time() - s0, 1), "generations": gens, "curve": curve, "routes": rs})
                print(f"{where} {method} seed {seed}: score {m['score']:.4f}, {m['riders']:,.0f} riders, "
                      f"{gens} generations, {runs[-1]['seconds']:.0f} s, {runs[-1]['evals']} evaluations", flush=True)
    table = {}
    for method in methods:
        mine = [r for r in runs if r["method"] == method]
        by_map = [[r["score"] for r in mine if r["map"] == w] for w, _ in todo]
        table[method] = {"score_iqm": iqm(sum(by_map, [])), "score_95": interval(by_map), "best": min(sum(by_map, [])),
                         "riders_iqm": iqm([r["riders"] for r in mine]),
                         "evals_per_s": sum(r["evals"] for r in mine) / max(sum(r["seconds"] for r in mine), 1e-9)}
        t = table[method]
        print(f"{method:>10}: score IQM {t['score_iqm']:.4f} [{t['score_95'][0]:.4f}, {t['score_95'][1]:.4f}], best "
              f"{t['best']:.4f}, riders IQM {t['riders_iqm']:,.0f}, {t['evals_per_s']:.1f} evaluations/s", flush=True)
    record("bench", {"methods": list(methods), "seeds": seeds, "seconds": seconds, "generations": generations,
                     "maps": list(maps)}, {"runs": runs, "table": table}, seeds=list(range(seeds)),
           evals=sim.EVALS - e0, start=t0)
    return runs


def sweep(routes=(12, 16, 20, 24, 32), buses=(64, 100, 200), seconds=300):
    """Today's design (server.design, rl+ga) of the state map at each route count x fleet, each judged by the exam
    (exam.exam): a row printed as it lands, and all of them recorded. -> rows"""
    import exam
    import rl
    import server

    t0, e0, rows = time.time(), sim.EVALS, []
    print(f"{'routes':>6} {'fleet':>5} {'used':>5} {'riders':>8} {'cover':>6} {'rural':>6} {'access':>6}  "
          f"{'worst 3 districts, access':<44} {'SK Kota -> KB Mall':>18} {'RM a day':>9} {'RM/rider':>8} {'shortest':>8}",
          flush=True)
    for R in routes:
        for B in buses:
            name = f"Routes {R}: {B} buses"
            server.design(rl.target().model_copy(update={"name": name, "optimizer": "rl+ga", "seconds": seconds,
                                                         "n_routes": R, "max_buses": B}))
            e = exam.exam(*exam.load(name))
            means = {d: float(np.mean(list(v.values()))) for d, v in e["districts"].items()}
            worst = sorted(means.items(), key=lambda kv: kv[1])[:3]
            worst_txt = ", ".join(f"{d} {v:.0%}" for d, v in worst)
            mins, changes = e["trips"]["SK Kota -> KB Mall"]
            trip = "no bus" if not math.isfinite(mins) else f"{mins:.0f} min, {changes} chg"
            row = {"name": name, "routes": R, "max_buses": B, "routes_run": e["routes"], "buses": e["buses"],
                   "riders": e["riders"], "covered_all": e["coverage"]["all"], "covered_rural": e["coverage"]["rural"],
                   "access": float(np.mean(list(e["access"].values()))), "worst": worst, "cost_day": e["cost_day"],
                   "rm_per_rider": e["cost_day"] / max(e["riders"], 1),
                   "shortest_route": min(e["route_minutes"], default=math.nan), "districts": means, "trips": e["trips"]}
            rows.append(row)
            print(f"{R:>6} {B:>5} {row['buses']:>5} {row['riders']:>8,.0f} {row['covered_all']:>6.0%} "
                  f"{row['covered_rural']:>6.0%} {row['access']:>6.0%}  {worst_txt:<44} {trip:>18} "
                  f"{row['cost_day']:>9,.0f} {row['rm_per_rider']:>8.1f} {row['shortest_route']:>4.0f} min", flush=True)
    record("sweep", {"routes": list(routes), "buses": list(buses), "seconds": seconds}, rows, start=t0,
           evals=sim.EVALS - e0)
    return rows


def compare(a, b, metric="score"):
    """Is B better than A? Each a bench record, "file.json" or "file.json:method". Runs pair by map and seed. -> (IQM of
    the change B / A - 1 over the pairs, its 95% stratified bootstrap interval, Wilcoxon signed-rank p, pairs where B
    is lower, pairs)"""
    from scipy.stats import wilcoxon

    def runs(x):
        f, _, method = str(x).partition(".json:")
        rs = json.loads(Path(f if f.endswith(".json") else f + ".json").read_text())["result"]["runs"]
        return {(r["map"], r["seed"]): r[metric] for r in rs if not method or r["method"] == method}

    A, B = runs(a), runs(b)
    both = sorted(set(A) & set(B))
    d = np.array([B[x] / A[x] - 1 for x in both])
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:  # every pair the same
        p = 1.0
    ci = interval([[B[x] / A[x] - 1 for x in both if x[0] == w] for w in sorted({x[0] for x in both})])
    print(f"B vs A, {metric}: IQM change {iqm(d):+.2%} [{ci[0]:+.2%}, {ci[1]:+.2%}], Wilcoxon p = {p:.3g}, "
          f"B lower in {int((d < 0).sum())} of {len(d)} pairs", flush=True)
    return iqm(d), ci, p, int((d < 0).sum()), len(d)


def fair(names):
    """The resources table (plan.md 6.5, 16.3): BAS.MY at its official timetable beside saved LAJU designs, each
    evaluated by today's rules with its own request's budgets: mode E1 (the same 64 buses) or E2 (also no more
    vehicle-hours and bus-km a day than BAS.MY's timetable). -> columns, recorded and printed"""
    import exam

    t0, e0, cols = time.time(), sim.EVALS, []
    for name in ["BAS.MY today"] + list(names):
        p, net, rs, heads, _, tt, _ = exam.load(name)
        m = sim.evaluate(p, rs, fixed=heads if tt else None)
        cols.append({"name": name, "mode": "timetable" if tt else "E2" if p.budget_vh or p.budget_km else "E1",
                     "riders": m["riders"], "boardings": m["boardings"], "fleet_k": m["fleet_k"], "vh_day": m["vh_day"],
                     "bus_km_day": m["bus_km_day"], "riders_per_km": m["riders"] / max(m["bus_km_day"], 1e-9),
                     "riders_per_vh": m["riders"] / max(m["vh_day"], 1e-9), "covered": m["covered"] / net["people"],
                     "violation": m["violation"], "violations": {k: v for k, v in m["violations"].items() if v},
                     "score": m["score"], "budget_vh": p.budget_vh, "budget_km": p.budget_km, "asc": p.asc})
    w = 26
    row = lambda label, f: print(f"{label:<30}" + "".join(f"{f(c):>{w}}" for c in cols))
    row("", lambda c: c["name"])
    row("mode", lambda c: c["mode"])
    row("riders a day", lambda c: f"{c['riders']:,.0f}")
    row("boardings a day", lambda c: f"{c['boardings']:,.0f}")
    row("buses: peak / day / eve / night", lambda c: " / ".join(map(str, c["fleet_k"])))
    row("vehicle-hours a day", lambda c: f"{c['vh_day']:,.0f}")
    row("bus-km a day", lambda c: f"{c['bus_km_day']:,.0f}")
    row("riders per bus-km", lambda c: f"{c['riders_per_km']:.3f}")
    row("riders per vehicle-hour", lambda c: f"{c['riders_per_vh']:.2f}")
    row("people near a served stop", lambda c: f"{c['covered']:.1%}")
    row("rules broken (share of limit)", lambda c: ", ".join(f"{k} {v:.3f}" for k, v in c["violations"].items()) or "none")
    record("fair", {"names": list(names)}, {"columns": cols}, evals=sim.EVALS - e0, start=t0)
    return cols


if __name__ == "__main__" and sys.argv[1:2] == ["fair"]:
    fair(sys.argv[2:])
    sys.exit()
if __name__ == "__main__" and sys.argv[1:2] == ["health"]:
    r = health(" ".join(sys.argv[2:]) or None)
    for x, v in r.items():
        print(f"{x:>18}: {v}")
    sys.exit()
if __name__ == "__main__" and sys.argv[1:2] == ["bench"]:
    import argparse

    a = argparse.ArgumentParser()
    a.add_argument("bench")
    a.add_argument("--methods", default="rl+ga,sample")
    a.add_argument("--seeds", type=int, default=10)
    a.add_argument("--seconds", type=float, default=300.0)
    a.add_argument("--generations", type=int, default=0)
    a.add_argument("--maps", default="state")
    x = a.parse_args()
    bench(x.methods.split(","), x.seeds, x.seconds, x.generations, x.maps.split(","))
    sys.exit()
if __name__ == "__main__" and sys.argv[1:2] == ["sweep"]:
    import argparse

    a = argparse.ArgumentParser()
    a.add_argument("sweep")
    a.add_argument("--routes", default="12,16,20,24,32")
    a.add_argument("--buses", default="64,100,200")
    a.add_argument("--seconds", type=float, default=300.0)
    x = a.parse_args()
    sweep(tuple(map(int, x.routes.split(","))), tuple(map(int, x.buses.split(","))), x.seconds)
    sys.exit()
if __name__ == "__main__" and sys.argv[1:2] == ["compare"]:
    compare(*sys.argv[2:4], *sys.argv[4:5])
    sys.exit()
if __name__ == "__main__":
    import tempfile

    import numpy as np

    line = np.full((3, 3), np.inf)
    line[0, 1] = line[1, 0] = line[1, 2] = line[2, 1] = 2.0
    p = sim.Problem(time=line, od=np.ones((3, 3)) - np.eye(3), xy=np.zeros((3, 2)))
    before = sim.EVALS
    for _ in range(3):
        sim.evaluate(p, [[0, 1, 2]])
    assert sim.EVALS - before == 3, sim.EVALS
    with tempfile.TemporaryDirectory() as tmp:
        f = record("self-check", {"routes": [[0, 1, 2]], "out": Path(tmp)}, {"score": 1.5}, seeds=[0],
                   evals=sim.EVALS - before, start=time.time() - 2, folder=Path(tmp))
        d = json.loads(f.read_text())
    assert tuple(d) == FIELDS and d["evals"] == 3 and d["seconds"] >= 2 and d["score_version"] == sim.SCORE_VERSION, d
    assert d["git"] and d["dirty"] is not None and d["cores"] and d["cpu"] and d["torch"] and d["config"]["out"] == tmp, d
    assert iqm([1, 2, 3, 100]) == 2.5 and iqm([5]) == 5.0  # one wild seed does not move it
    lo, hi = interval([[1, 2, 3, 4], [10, 11, 12, 13]])
    assert lo <= iqm([1, 2, 3, 4, 10, 11, 12, 13]) <= hi and lo < hi
    assert abs(jaccard([[[0, 1, 2]], [[2, 1, 0]], [[0, 1]]]) - 2 / 3) < 1e-12  # the same links either way; half of them
    print("ok", {k: d[k] for k in ("git", "dirty", "score_version", "python", "torch", "cpu", "cores")})
