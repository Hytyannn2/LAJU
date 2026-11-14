"""Route-set generators on a stop graph (sim.Problem). Routes are lists of stop indices along adjacent stops.

baseline(): busiest unserved trip first, served by its quickest path. The benchmark everything else must beat.
evolve():   genetic algorithm over whole route sets (Phase 4), operators after Mumford (2013):
            crossover alternates routes between parents favouring new stops, mutation grows/shrinks route ends;
            where the problem says where routes may end (p.rest_end), new routes run between such stops and a route
            grows to one along the quickest road or shrinks back to one.
            Runs until the caller stops asking: ga() gives it a time limit, server.py trains until the planner pauses.
"""
import random
import time

import numpy as np

from sim import evaluate, quickest


def path(pred, i, j):
    out = [int(j)]
    while out[-1] != i:
        out.append(int(pred[i, out[-1]]))
    return out[::-1]


def unturn(p, r):
    """Skip any stop the bus could only serve by U-turning, where a direct link around it exists."""
    k = 1
    while k < len(r) - 1:
        if (r[k - 1], r[k], r[k + 1]) in p.uturn and np.isfinite(p.time[r[k - 1], r[k + 1]]):
            r = r[:k] + r[k + 1:]
            k = max(1, k - 1)
        else:
            k += 1
    return r


def trim(p, r):
    """The start of route r that fits max_len stops and max_route_min minutes (but never under min_len stops)."""
    if p.max_route_min:
        t = np.cumsum([0.0] + [p.time[a, b] for a, b in zip(r, r[1:])])
        r = r[: max(p.min_len, int((t <= p.max_route_min).sum()))]
    return r[: p.max_len]


def settle(p, r):
    """Route r cut back at both ends to stops where a route may end (p.rest_end), if at least min_len stops remain."""
    k = [i for i, s in enumerate(r) if p.rest_end is None or p.rest_end[s]]
    return r[k[0]: k[-1] + 1] if k and k[-1] - k[0] + 1 >= p.min_len else r


def baseline(p, n_routes, locked=()):
    D, pred = quickest(p)
    left = (p.od + p.od.T) * np.isfinite(D)
    rs = [list(r) for r in locked]
    for r in rs:
        left[np.ix_(r, r)] = 0
    while len(rs) < n_routes and left.max() > 0:
        i, j = np.unravel_index(left.argmax(), left.shape)
        r = settle(p, trim(p, unturn(p, path(pred, i, j))))
        rs.append(r)
        left[np.ix_(r, r)] = left[i, j] = left[j, i] = 0
    return rs


_P = _LOCKED = None


def _init(p, locked):
    global _P, _LOCKED
    _P, _LOCKED = p, [list(r) for r in locked]


def _fit(rs):
    return evaluate(_P, _LOCKED + rs)["score"]


def free_gb():
    """Free physical memory in GB (Windows; elsewhere infinite)."""
    import ctypes

    class M(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(f, ctypes.c_ulonglong) for f in
                    ("total", "avail", "ptotal", "pavail", "vtotal", "vavail", "xavail")]
    try:
        m = M(len=ctypes.sizeof(M))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail / 2**30
    except AttributeError:
        return float("inf")


def pool(p, locked=(), n=4):
    """Worker processes that score route sets for evolve (plan.md T-B3), or None under 3 GB free memory.
    The caller closes it (pool.terminate()) in a finally."""
    if free_gb() < 3:
        return None
    import multiprocessing

    return multiprocessing.Pool(n, initializer=_init, initargs=(p, locked))


def ga(p, n_routes, locked=(), seconds=20.0, pop=40, seed=0, seeds=(), pool=None):
    """Best route set found within `seconds`. Locked routes are kept as-is; `seeds` are extra starting route sets."""
    if n_routes <= len(locked):
        return [list(r) for r in locked]
    deadline = time.time() + seconds
    for scored in evolve(p, n_routes, locked, pop, seed, seeds, pool):
        if time.time() >= deadline:
            return [list(r) for r in locked] + scored[0][1]


def evolve(p, n_routes, locked=(), pop=40, seed=0, seeds=(), pool=None):
    """Yields the population after every generation, forever: [(score, free routes)], best first.
    Needs at least one route that is not locked. With a pool (routes.pool, same p and locked) children are scored in
    parallel; results match the serial run for a seed."""
    rng = random.Random(seed)
    n = len(p.od)
    locked = [list(r) for r in locked]
    free = n_routes - len(locked)
    D, pred = quickest(p)
    nb = [[int(j) for j in np.flatnonzero(np.isfinite(p.time[i])) if j != i] for i in range(n)]
    weight = (p.od.sum(0) + p.od.sum(1) + 1e-9).tolist()
    rest = [] if p.rest_end is None else [int(i) for i in np.flatnonzero(p.rest_end)]
    ends = rest if len(rest) >= 2 else range(n)

    def new_route():
        while True:
            i, j = rng.choices(ends, weights=[weight[x] for x in ends], k=2)
            if i != j and np.isfinite(D[i, j]):
                r = unturn(p, path(pred, i, j))
                k = rng.randrange(len(r) - p.max_len + 1) if len(r) > p.max_len else 0
                return settle(p, trim(p, r[k:]))

    def fill(rs):
        rs = [list(r) for r in rs][:free]
        return rs + [new_route() for _ in range(free - len(rs))]

    def cross(a, b):
        child, seen, pools = [], set(), [list(a), list(b)]
        for k in range(free):
            pool = pools[k % 2] or pools[(k + 1) % 2]
            best = max(pool, key=lambda r: len(set(r) - seen) / len(r))
            pool.remove(best)
            child.append(best)
            seen |= set(best)
        return child

    def mutate(rs):
        rs = [list(r) for r in rs]
        k = rng.randrange(free)
        r, roll = rs[k], rng.random()
        if roll < 0.2:
            rs[k] = new_route()
        elif roll < 0.6 and len(r) < p.max_len:
            if rng.random() < 0.5:
                r.reverse()
            if rest:  # on to a nearby stop where a route may end (the nearest of 3 picked at random), by the quickest road
                opts = [s for s in rest if s not in r and np.isfinite(D[r[-1], s])]
                ext = path(pred, r[-1], min(rng.sample(opts, min(3, len(opts))), key=lambda s: D[r[-1], s]))[1:] if opts else r
                if not set(ext) & set(r):
                    rs[k] = settle(p, trim(p, unturn(p, r + ext)))
            else:
                opts = [s for s in nb[r[-1]] if s not in r and (len(r) < 2 or (r[-2], r[-1], s) not in p.uturn)]
                covered = {s for x in locked + rs for s in x}
                if opts:
                    r.append(rng.choice([s for s in opts if s not in covered] or opts))
        elif len(r) > p.min_len:
            r.pop(0 if rng.random() < 0.5 else -1)
            rs[k] = settle(p, r)
        return rs

    def fit(xs):
        return list(zip(pool.map(_fit, xs) if pool else [evaluate(p, locked + rs)["score"] for rs in xs], xs))

    start = [fill(baseline(p, n_routes, locked)[len(locked):])] + [fill(s) for s in seeds]
    start += [fill([]) for _ in range(pop - len(start))]
    scored = sorted(fit(start), key=lambda t: t[0])

    def pick():
        return min(rng.sample(scored, 2), key=lambda t: t[0])[1]

    while True:
        yield scored
        kids = [mutate(cross(pick(), pick())) for _ in range(pop - 2)]
        scored = sorted(scored[:2] + fit(kids), key=lambda t: t[0])


def valid(p, rs):
    return all(
        p.min_len <= len(r) <= p.max_len and len(set(r)) == len(r) and all(np.isfinite(p.time[a, b]) for a, b in zip(r, r[1:]))
        for r in rs
    )


if __name__ == "__main__":
    from sim import Problem

    side = 5  # 5x5 grid, 3-minute links, demand towards the centre
    n = side * side
    t = np.full((n, n), np.inf)
    for i in range(n):
        for j in (i + 1, i + side):
            if j < n and (j == i + side or j % side):
                t[i, j] = t[j, i] = 3.0
    xy = np.array([(i % side, i // side) for i in range(n)], float)
    pop = 1 + 10 * np.exp(-((xy - 2) ** 2).sum(1))
    od = np.outer(pop, pop)
    np.fill_diagonal(od, 0)
    p = Problem(time=t, od=od, xy=xy, max_len=7)

    b = baseline(p, 4)
    assert len(b) == 4 and valid(p, b), b
    lock = [[0, 1, 2]]
    g = ga(p, 4, locked=lock, seconds=3)
    assert g[0] == lock[0] and len(g) == 4 and valid(p, g), g
    sb, sg = evaluate(p, b)["score"], evaluate(p, ga(p, 4, seconds=3))["score"]
    assert sg <= sb, (sg, sb)
    gen = evolve(p, 4, seeds=[next(evolve(p, 4))[5][1]])  # resuming from a saved population member works
    best = [next(gen)[0][0] for _ in range(50)]
    assert best == sorted(best, reverse=True), best  # the best design never gets worse while training

    p.max_route_min = 9  # three 3-minute links
    assert trim(p, [0, 1, 2, 3, 4]) == [0, 1, 2, 3] and all(len(r) <= 4 for r in baseline(p, 4))
    p.max_route_min, p.rest_end = 0, np.isin(np.arange(n), [0, 4, 12, 20, 24])  # routes may end at the corners, centre
    assert settle(p, [1, 0, 5, 10, 11, 12, 13]) == [0, 5, 10, 11, 12]
    gen = evolve(p, 4)
    rest = [next(gen)[0][1] for _ in range(30)][-1]
    assert all(p.rest_end[r[0]] and p.rest_end[r[-1]] for r in rest), rest
    import multiprocessing  # T-B3: a pool gives the serial run's result for the same seed

    with multiprocessing.Pool(2, initializer=_init, initargs=(p, lock)) as pl:
        a, b = evolve(p, 4, lock, seed=3), evolve(p, 4, lock, seed=3, pool=pl)
        sa, sb2 = [next(a)[0][0] for _ in range(20)], [next(b)[0][0] for _ in range(20)]
    assert sa == sb2, (sa, sb2)
    print("ok baseline", round(sb, 3), "ga", round(sg, 3), "training", round(best[0], 3), "->", round(best[-1], 3))
