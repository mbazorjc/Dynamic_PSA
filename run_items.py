"""
Runs for the item-level model (paper Sections 4-5).

  python run_items.py main     static vs dynamic, decomposition, items, unavailability
  python run_items.py diag     convergence diagnostics + time-step check
  python run_items.py sweep    entry time t0 and level-margin k_m sweeps
  python run_items.py v5v6     aggregation check and hierarchical-posterior check
"""
import sys, json, time
import numpy as np
from dataclasses import replace, asdict
from scipy.stats import spearmanr
from nhmc_core import Plant, simulate as simulate_core, Config as CoreCfg, theta_mean as core_theta_mean
from nhmc_items import *

job = sys.argv[1] if len(sys.argv) > 1 else "main"
p, hy = Plant(), Hyd()


def summ(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), p05=float(np.quantile(x, .05)),
                p50=float(np.quantile(x, .5)), p95=float(np.quantile(x, .95)))


def tolist(d):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


if job == "main":
    n_e, n_a, bias = 200, 2500, 5.0
    th = sample_theta(n_e, np.random.default_rng(2026))
    R = {"meta": dict(n_e=n_e, n_a=n_a, bias=bias, n_steps=Cfg().n_steps,
                      plant=asdict(p), hyd=asdict(hy))}
    t1 = time.time()
    lh_t0 = hydraulic_rate(th, p, hy, False, 2000, np.random.default_rng(5))
    lh_av = hydraulic_rate(th, p, hy, True, 2000, np.random.default_rng(6))
    R["lam_hyd_t0"], R["lam_hyd_avg"] = summ(lh_t0), summ(lh_av)
    print("hydraulic rates %.0fs" % (time.time() - t1), R["lam_hyd_t0"], R["lam_hyd_avg"], flush=True)
    for name, mode, hyd, lh in (("static_H", "t0", False, lh_t0), ("static_t0", "t0", True, lh_t0),
                                ("static_avg", "avg", True, lh_av)):
        Pc = static_items_P(th, lh, p, mode, hyd)
        R[name] = dict(P=Pc.sum(1).tolist(), summary=summ(Pc.sum(1)), cause_mean=Pc.mean(0).tolist())
        print(name, R[name]["summary"], flush=True)
    steps = [("mc_age", Cfg(td=False, lv=False, deg=False), lh_t0),
             ("mc_age_td", Cfg(td=True, lv=False, deg=False), lh_av),
             ("mc_age_td_lv", Cfg(td=True, lv=True, deg=False), None),
             ("dynamic", Cfg(), None)]
    for i, (name, cfg, lh) in enumerate(steps):
        t1 = time.time()
        r = simulate_items(th, n_a, p, hy, replace(cfg, bias=bias),
                           np.random.default_rng(300 + i), lam_hyd=lh)
        R[name] = dict(P=r["P"].tolist(), summary=summ(r["P"]),
                       cause_mean=r["P_cause"].mean(0).tolist(),
                       t=r["t"].tolist(), P_t_mean=r["P_t"].mean(0).tolist(),
                       U_t_mean=r["U_t"].mean(0).tolist(), U_mean=summ(r["U_mean"]),
                       item_failures_mean=r["item_failures"].mean(0).tolist(),
                       hyd_failures=summ(r["hyd_failures"]))
        print(name, "%.0fs" % (time.time() - t1), R[name]["summary"], "U", R[name]["U_mean"], flush=True)
    # epistemic distribution of item failure rates lambda_i = w_i * lambda_A
    lam_items = th["w"] * th["lam_run"][:, None]
    R["item_rates"] = {ITEMS[j]: summ(lam_items[:, j]) for j in range(N_ITEMS)}
    keys = [k for k in th if k != "w"]
    R["spearman_dynamic"] = {k: float(spearmanr(th[k], R["dynamic"]["P"])[0]) for k in keys}
    R["spearman_dynamic"]["w_seals"] = float(spearmanr(th["w"][:, ITEMS.index("Seals")], R["dynamic"]["P"])[0])
    json.dump(R, open("items_main.json", "w"))
    print("saved")

elif job == "diag":
    th = theta_mean()
    out = {}
    t1 = time.time()
    out["streams"] = diagnostics(th, 50_000, p, hy, Cfg())
    print("diag", out["streams"], "%.0fs" % (time.time() - t1), flush=True)
    out["dt"] = {}
    for ns in (250, 500, 1000, 2000):
        r = simulate_items(th, 100_000, p, hy, Cfg(n_steps=ns, out_every=max(1, ns // 100)),
                           np.random.default_rng(77))
        out["dt"][ns] = dict(P=float(r["P"][0]), U=float(r["U_mean"][0]))
        print("n_steps", ns, out["dt"][ns], flush=True)
    # running estimate vs N (one long stream)
    y = simulate_items_outcomes(th, 200_000, p, hy, Cfg(), np.random.default_rng(99))
    Ns = np.unique(np.logspace(3, np.log10(y.size), 25).astype(int))
    out["running"] = dict(N=Ns.tolist(), mean=[float(y[:n].mean()) for n in Ns],
                          se=[float(y[:n].std(ddof=1) / np.sqrt(n)) for n in Ns])
    json.dump(out, open("items_diag.json", "w"))
    print("saved")

elif job == "sweep":
    n_e, n_a = 60, 1500
    th = sample_theta(n_e, np.random.default_rng(2027))
    out = {"t0": {}, "km": {}}
    for key, grid in (("t0", (24.0, 72.0, 168.0)), ("km", (3.0, 3.5, 4.0, 4.5))):
        for v in grid:
            pp = replace(p, t0=v) if key == "t0" else replace(p, k_margin=v)
            lh0 = hydraulic_rate(th, pp, hy, False, 1000, np.random.default_rng(8))
            lha = hydraulic_rate(th, pp, hy, True, 1000, np.random.default_rng(9))
            row = {"static_H": summ(static_items_P(th, lh0, pp, "t0", False).sum(1)),
                   "static_t0": summ(static_items_P(th, lh0, pp, "t0", True).sum(1)),
                   "static_avg": summ(static_items_P(th, lha, pp, "avg", True).sum(1))}
            t1 = time.time()
            r = simulate_items(th, n_a, pp, hy, Cfg(bias=5.0), np.random.default_rng(int(10 * v)))
            row["dynamic"] = summ(r["P"])
            row["dynamic_U"] = summ(r["U_mean"])
            row["dynamic_cause_mean"] = r["P_cause"].mean(0).tolist()
            out[key][str(v)] = row
            print(key, v, "%.0fs" % (time.time() - t1),
                  {k: round(vv["mean"], 6) for k, vv in row.items() if isinstance(vv, dict)}, flush=True)
    json.dump(out, open("items_sweep.json", "w"))

elif job == "v5v6":
    out = {}
    # V5: item model with all coupling switched off reproduces the aggregated core model
    th = theta_mean()
    ri = simulate_items(th, 300_000, p, hy, Cfg(td=False, lv=False, deg=False, bias=10.0),
                        np.random.default_rng(55), lam_hyd=np.array([0.0]))
    rc = simulate_core(core_theta_mean(), 300_000, p,
                       CoreCfg(td=False, lv=False, age=True, ae=False, bias=10.0, dt=p.Tm / 1000),
                       np.random.default_rng(56))
    out["V5"] = dict(items=float(ri["P"][0]), core=float(rc["P"][0]),
                     static_H=float(static_items_P(th, np.array([0.0]), p, "t0", False).sum()))
    print("V5", out["V5"], flush=True)
    # V6: hierarchical gamma-Poisson posterior by quadrature vs. the classic
    # 10-pump data set (Gaver & O'Muircheartaigh 1987) used in the BUGS examples
    x = [5, 1, 5, 14, 3, 19, 1, 1, 4, 22]
    tau = [94.3, 15.7, 62.9, 126, 5.24, 31.4, 1.05, 1.05, 2.1, 10.5]
    post = hierarchical_posterior(x, tau)
    # independent reference: self-normalised importance sampling from the hyperprior
    rng = np.random.default_rng(66)
    a = rng.exponential(1.0, 4_000_000); b = rng.gamma(0.1, 1.0, 4_000_000)
    from scipy.special import gammaln
    lw = np.zeros_like(a)
    for xi, ti in zip(x, tau):
        lw += gammaln(a + xi) - gammaln(a) + a * np.log(b / (b + ti)) + xi * np.log(ti / (b + ti))
    wts = np.exp(lw - lw.max()); wts /= wts.sum()
    ref = [float(np.sum(wts * (a + xi) / (b + ti))) for xi, ti in zip(x, tau)]
    out["V6"] = dict(x=x, tau=tau, mean=post[:, 0].tolist(), q05=post[:, 1].tolist(),
                     q95=post[:, 2].tolist(), is_mean=ref, ess=float(1 / np.sum(wts ** 2)))
    print("V6 means", np.round(post[:, 0], 4), flush=True)
    json.dump(out, open("items_v5v6.json", "w"))

elif job == "static_curves":
    # exact static P(t) curves for the same theta / hydraulic rates as `main`
    n_e = 200
    th = sample_theta(n_e, np.random.default_rng(2026))
    lh_t0 = hydraulic_rate(th, p, hy, False, 2000, np.random.default_rng(5))
    lh_av = hydraulic_rate(th, p, hy, True, 2000, np.random.default_rng(6))
    tgrid = np.linspace(0, p.Tm, 101)
    out = {"t": tgrid.tolist()}
    for name, mode, hyd, lh in (("static_H", "t0", False, lh_t0), ("static_t0", "t0", True, lh_t0),
                                ("static_avg", "avg", True, lh_av)):
        curves = static_curve(th, lh, p, tgrid, mode, hyd)
        out[name] = curves.mean(0).tolist()
        print(name, out[name][-1], flush=True)
    out["lam_hyd_t0"], out["lam_hyd_avg"] = lh_t0.tolist(), lh_av.tolist()
    json.dump(out, open("items_static_curves.json", "w"))
