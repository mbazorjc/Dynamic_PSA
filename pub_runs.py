"""
pub_runs.py -- runs behind the publication figures and the conditional results.

  python pub_runs.py cond     results conditional on the assumed parameters
  python pub_runs.py epi      epistemic run with state occupancy and per-theta outputs
  python pub_runs.py streams  four independent replicate streams (convergence traces)
"""
import sys, json, time
import numpy as np
from dataclasses import replace
from nhmc_core import Plant, time_to_boil
from nhmc_items import (Hyd, Cfg, simulate_items, simulate_items_outcomes, sample_theta, theta_mean,
                        item_share_alpha, ITEMS, N_ITEMS)

p, hy = Plant(), Hyd()
job = sys.argv[1] if len(sys.argv) > 1 else "cond"


def at_mean(**over):
    th = theta_mean()
    for k, v in over.items():
        th[k] = np.array([v], float) if k != "w" else np.asarray(v, float)[None, :]
    return th


def run_point(th, n=50_000, seed=7, cfg=Cfg()):
    thr = {k: np.repeat(v, n, axis=0) for k, v in th.items()}
    t1 = time.perf_counter()
    r = simulate_items(thr, 1, p, hy, replace(cfg, out_every=cfg.n_steps), np.random.default_rng(seed))
    y = r["P"]
    return dict(P=float(y.mean()), se=float(y.std(ddof=1) / np.sqrt(n)),
                U=float(r["U_mean"].mean()), time_s=time.perf_counter() - t1,
                cause=(r["P_cause"].mean(0)).tolist())


if job == "cond":
    out = {"base": run_point(at_mean())}
    print("base", out["base"], flush=True)
    out["nu_j"] = {}
    for v in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        out["nu_j"][str(v)] = run_point(at_mean(nu_j=v)); print("nu_j", v, out["nu_j"][str(v)]["P"], flush=True)
    out["M"] = {}
    for M in (50e3, 75e3, 100e3, 128e3, 150e3):
        r = run_point(at_mean(M=M)); r["tb_min"] = float(60 * time_to_boil(p.t0, M, p))
        out["M"][str(M)] = r; print("M", M, r["tb_min"], r["P"], flush=True)
    out["med_a"] = {}
    for m in (20, 32.5, 45, 90, 180):
        out["med_a"][str(m)] = run_point(at_mean(med_a=m / 60)); print("m_g", m, out["med_a"][str(m)]["P"], flush=True)
    # item shares: OREDA external-leakage shares (base) vs equal shares
    out["shares_equal"] = run_point(at_mean(w=np.full(N_ITEMS, 1 / N_ITEMS)))
    print("equal shares", out["shares_equal"], flush=True)
    json.dump(out, open("pub_cond.json", "w"), indent=1)

elif job == "epi":
    n_e, n_a = 200, 2000
    th = sample_theta(n_e, np.random.default_rng(2031))
    t1 = time.perf_counter()
    r = simulate_items(th, n_a, p, hy, Cfg(bias=5.0, occupancy=True, out_every=20), np.random.default_rng(2032))
    occ = r["occupancy"]                                   # (n_e, n_out, 9)
    q = lambda a, x: np.quantile(a, x, axis=0).tolist()
    out = dict(t=r["t"].tolist(), time_s=time.perf_counter() - t1,
               occ_mean=occ.mean(0).tolist(), occ_lo=q(occ, .025), occ_hi=q(occ, .975),
               P_t_mean=r["P_t"].mean(0).tolist(), P_t_lo=q(r["P_t"], .025), P_t_hi=q(r["P_t"], .975),
               P=r["P"].tolist(), U=r["U_mean"].tolist(), hyd=r["hyd_failures"].tolist(),
               item_failures=r["item_failures"].tolist(),
               theta={k: (v.tolist()) for k, v in th.items()})
    json.dump(out, open("pub_epi.json", "w"))
    print("epi done %.0fs" % out["time_s"], np.mean(r["P"]))

elif job == "streams":
    th = theta_mean(); out = {}
    for s in range(4):
        t1 = time.perf_counter()
        y = simulate_items_outcomes(th, 50_000, p, hy, Cfg(out_every=1000), np.random.default_rng(300 + s))
        Ns = np.unique(np.logspace(2, np.log10(y.size), 60).astype(int))
        out[str(s)] = dict(N=Ns.tolist(), mean=[float(y[:k].mean()) for k in Ns],
                           se=[float(y[:k].std(ddof=1) / np.sqrt(k)) for k in Ns],
                           time_s=time.perf_counter() - t1)
        print("stream", s, y.mean(), flush=True)
    json.dump(out, open("pub_streams.json", "w"))
