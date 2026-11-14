"""Design-time search on top of the trained policy (plan.md stage 2). Lazy imports keep torch out of plain GA runs."""
import time

import torch

import sim


@torch.no_grad()
def sample(p, n_routes, n=64, seconds=None, locked=(), seed=0, k=16, net=None):
    """Up to `n` distinct route sets from the policy, drawn k at a time (the very first one greedy), for at most
    `seconds` if given (one batch always runs). Returns [(sim.key, free routes)], best first; locked routes left out."""
    import rl

    if n_routes <= len(locked):
        return []
    net = net or rl._load()
    G, gen = rl.Graph(p, n_routes, locked), torch.Generator().manual_seed(seed)
    H, end, seen, out = net.encode(G), time.time() + (seconds if seconds is not None else 1e9), set(), []
    first = True
    while first or (len(seen) < n and time.time() < end):
        sets = rl.rollout(net, G, 1, greedy=True, H=H)[0] if first else []
        sets += rl.rollout(net, G, k - len(sets), gen=gen, H=H)[0]
        first = False
        for rs in sets:
            sig = tuple(sorted(tuple(r) for r in rs))
            if sig not in seen:
                seen.add(sig)
                out.append((sim.key(sim.evaluate(p, rs)), [list(r) for r in rs[len(locked):]]))
    return sorted(out, key=lambda t: t[0])[:n]  # ponytail: ranks on sim.key only; add planner pools if a task needs them


@torch.no_grad()
def lns(p, n_routes, start, seconds, locked=(), seed=0, k=16, net=None, log=None):
    """Destroy and repair (plan.md T-B2): drop 1-3 free routes (half the time the least-used per bus), let the policy redraw
    them around the rest, keep the best repair if it beats the incumbent. Returns the incumbent's free routes."""
    import random

    import rl

    free, locked = [list(r) for r in start], [list(r) for r in locked]
    if not free:
        return free
    net, rng = net or rl._load(), random.Random(seed)
    G, gen = rl.Graph(p, n_routes, locked), torch.Generator().manual_seed(seed)
    H, end = net.encode(G), time.time() + seconds
    best_m = sim.evaluate(p, locked + free)
    best = sim.key(best_m)
    while time.time() < end:
        q = rng.randint(1, min(3, len(free)))
        use = best_m.get("route_boardings")
        if use and rng.random() < 0.5:
            per = [use[len(locked) + i] / max(1, best_m["buses"][len(locked) + i]) for i in range(len(free))]
            drop = set(sorted(range(len(free)), key=per.__getitem__)[:q])
        else:
            drop = set(rng.sample(range(len(free)), q))
        G.locked = locked + [r for i, r in enumerate(free) if i not in drop]
        cand = min(((sim.key(m), rs, m) for rs in rl.rollout(net, G, k, gen=gen, H=H)[0] for m in [sim.evaluate(p, rs)]),
                   key=lambda t: t[0])
        won = cand[0] < best
        if log:
            log(q, sorted(drop), cand[0], won)
        if won:
            best, free, best_m = cand[0], cand[1][len(locked):], cand[2]
    return free


if __name__ == "__main__":  # self-check on Mandl: valid sets, sorted by key, best no worse than the greedy first draw
    import bench
    import routes

    p = bench.problem()
    res = sample(p, 6, n=32)
    assert 0 < len(res) <= 32 and all(routes.valid(p, rs) and len(rs) == 6 for _, rs in res)
    assert [k for k, _ in res] == sorted(k for k, _ in res)
    import rl

    greedy = sim.key(sim.evaluate(p, rl.rollout(rl._load(), rl.Graph(p, 6), 1, greedy=True)[0][0]))
    assert res[0][0] <= greedy
    out = lns(p, 6, res[0][1], 5)
    assert routes.valid(p, out) and len(out) == 6 and sim.key(sim.evaluate(p, out)) <= res[0][0], "lns got worse"
    print("search ok:", len(res), "distinct sets, best key", res[0][0], "-> lns", sim.key(sim.evaluate(p, out)))
