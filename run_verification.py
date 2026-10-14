"""
Verification of the dynamic Markov Monte Carlo simulator (Section 3.6 of the paper).

V1  static CTMC, no repair, no entrainment  -> closed-form solution
V1b static CTMC, repair + constant entrainment -> matrix exponential (canonical form)
V2  time-inhomogeneous CTMC (decay-heat-driven rates) -> forward Kolmogorov ODE
V3  convergence in time step dt and in number of histories N
V4  per-step i.i.d. re-sampling of the transition matrix collapses to the
    homogeneous chain with E[P] (Proposition 1) and destroys the epistemic band
"""
import json, time
import numpy as np
from dataclasses import replace
from scipy.linalg import expm
from nhmc_core import *

p = Plant()
th = theta_mean()
ths = {k: float(v[0]) for k, v in th.items()}
res = {}


def ci(P, n, bias_used):
    # binomial-type 95% half-width (conservative for IS-weighted estimator not used here)
    return 1.96 * np.sqrt(P * (1 - P) / n)


# ---------------- V1: closed form ----------------------------------------
cfg = Config(td=False, lv=False, age=False, s2="exponential", repair=False, ae=False)
q, med = ths["q_fts"], ths["med_h"]
a = (1 - ths["alpha2"]) * ths["lam_run"]
c = ths["alpha2"] * ths["lam_run"]
k23 = (1 - q) / med
k2 = 1 / med + 1 / float(time_to_boil(p.t0, ths["M"], p))
lb = (ths["lam_b1"] + ths["lam_b2"] * (p.Tm - 1)) / p.Tm
tgrid = np.arange(0, p.Tm + 1e-9, 0.5)
P_cf = closed_form_noAE(tgrid, a, c, k23, k2, lb)
G9 = markov_generator_9(p.t0, ths, p, cfg)
P_ex = np.array([1 - expm(G9 * t)[0, :4].sum() for t in tgrid])
out = {}
for bias, nrep in ((1.0, 400_000), (20.0, 400_000)):
    rng = np.random.default_rng(11)
    t1 = time.time()
    r = simulate(th, nrep, p, replace(cfg, bias=bias), rng)
    out[bias] = dict(P=float(r["P"][0]), t=r["t"].tolist(), P_t=r["P_t"][0].tolist(),
                     sec=time.time() - t1)
res["V1"] = dict(closed_form=float(P_cf[-1]), expm=float(P_ex[-1]),
                 max_abs_cf_vs_expm=float(np.max(np.abs(P_cf - P_ex))),
                 mc_analog=out[1.0]["P"], mc_is20=out[20.0]["P"],
                 t=tgrid.tolist(), P_cf=P_cf.tolist(),
                 P_t_analog=out[1.0]["P_t"], P_t_is=out[20.0]["P_t"])
print("V1", json.dumps({k: v for k, v in res["V1"].items() if not isinstance(v, list)}, indent=1))

# ---------------- V1b: repair + constant entrainment, expm + canonical ----
cfg = Config(td=False, lv=False, age=False, s2="exponential", repair=True, ae=True)
G9 = markov_generator_9(p.t0, ths, p, cfg)
P_ex = np.array([1 - expm(G9 * t)[0, :4].sum() for t in tgrid])
can = canonical(G9, 4)
rng = np.random.default_rng(12)
r = simulate(th, 400_000, p, replace(cfg, bias=10.0), rng)
res["V1b"] = dict(expm=float(P_ex[-1]), mc=float(r["P"][0]),
                  mc_cause=(r["P_cause"][0]).tolist(),
                  expm_cause=expm(G9 * p.Tm)[0, 4:].tolist(),
                  mtta_h=float(can["mtta"][0]),
                  B_abs=can["B"][0].tolist(),
                  t=tgrid.tolist(), P_ex=P_ex.tolist(), P_t=r["P_t"][0].tolist())
print("V1b", res["V1b"]["expm"], res["V1b"]["mc"], "MTTA[h]=", res["V1b"]["mtta_h"])

# ---------------- V2: time-inhomogeneous -> Kolmogorov ODE ---------------
cfg = Config(td=True, lv=False, age=False, s2="exponential", repair=True, ae=True)
Y = kolmogorov_reference(ths, p, cfg, tgrid)
P_ode = 1 - Y[:, :4].sum(1)
rng = np.random.default_rng(13)
r = simulate(th, 400_000, p, replace(cfg, bias=10.0), rng)
res["V2"] = dict(ode=float(P_ode[-1]), mc=float(r["P"][0]),
                 ode_cause=Y[-1, 4:].tolist(), mc_cause=r["P_cause"][0].tolist(),
                 t=tgrid.tolist(), P_ode=P_ode.tolist(), P_t=r["P_t"][0].tolist(),
                 static_t0=float(1 - expm(markov_generator_9(p.t0, ths, p,
                                  replace(cfg, td=False)) * p.Tm)[0, :4].sum()))
print("V2", res["V2"]["ode"], res["V2"]["mc"], "static_t0", res["V2"]["static_t0"])

# ---------------- V3a: dt convergence (V2 config, vs ODE) -----------------
dts = [8, 4, 2, 1, 0.5]
v3 = []
for dtm in dts:
    rng = np.random.default_rng(14)
    r = simulate(th, 200_000, p, replace(cfg, bias=10.0, dt=dtm / 60.0), rng)
    v3.append(float(r["P"][0]))
res["V3_dt"] = dict(dt_min=dts, P=v3, ode=float(P_ode[-1]))
print("V3 dt", v3, P_ode[-1])

# ---------------- V3b: dt convergence of the full (non-Markov) model -------
dts_full = [8, 4, 2, 1, 0.5]
v3f = []
for dtm in dts_full:
    rng = np.random.default_rng(15)
    r = simulate(th, 100_000, p, Config(bias=10.0, dt=dtm / 60.0), rng)
    v3f.append(float(r["P"][0]))
res["V3_dt_full"] = dict(dt_min=dts_full, P=v3f)
print("V3 dt full", v3f)

# ---------------- V3c: N convergence (V2 config) ---------------------------
Ns = [5_000, 20_000, 80_000]
reps = 6
v3n = []
for N in Ns:
    vals = []
    for rep in range(reps):
        rng = np.random.default_rng(1000 + rep)
        vals.append(float(simulate(th, N, p, replace(cfg, bias=10.0, dt=2 / 60), rng)["P"][0]))
    v3n.append(vals)
res["V3_N"] = dict(N=Ns, P=v3n, ode=float(P_ode[-1]))
print("V3 N rmse", [float(np.sqrt(np.mean((np.array(v) - P_ode[-1]) ** 2))) for v in v3n])

# ---------------- V4: per-step re-sampling collapse ------------------------
rng = np.random.default_rng(16)
ne = 4000
thE = sample_theta(ne, rng)
cfg4 = Config(td=False, lv=False, age=False, s2="exponential", repair=True, ae=True)
dt4 = 1 / 60
K = int(round(p.Tm / dt4))
P_hist = np.empty(ne)          # two-loop: theta fixed per history
Pbar = np.zeros((9, 9))        # E_theta[ expm(G(theta) dt) ]
for k in range(ne):
    tk = {key: float(v[k]) for key, v in thE.items()}
    G = markov_generator_9(p.t0, tk, p, cfg4)
    P_hist[k] = 1 - expm(G * p.Tm)[0, :4].sum()
    Pbar += expm(G * dt4)
Pbar /= ne
P_resample_theory = 1 - np.linalg.matrix_power(Pbar, K)[0, :4].sum()
# direct simulation of per-step re-sampling (fresh theta every step, every history)
nh = 200_000
Z = np.zeros(nh, dtype=int)
Pk_stack = []
for k in range(K):
    idx = rng.integers(0, ne, nh)                     # fresh theta per step
    # build rows only for the 4 transient states from cached step matrices
    if k == 0:
        cache = np.stack([expm(markov_generator_9(p.t0, {key: float(v[j]) for key, v in thE.items()}, p, cfg4) * dt4)
                          for j in range(ne)])
        cum = np.cumsum(cache, axis=2)
    u = rng.random(nh)
    alive = Z < 4
    rows = cum[idx[alive], Z[alive], :]
    Z[alive] = (u[alive][:, None] > rows).sum(1)
P_resample_mc = float((Z >= 4).mean())
res["V4"] = dict(two_loop_mean=float(P_hist.mean()),
                 two_loop_p05=float(np.quantile(P_hist, 0.05)),
                 two_loop_p50=float(np.quantile(P_hist, 0.5)),
                 two_loop_p95=float(np.quantile(P_hist, 0.95)),
                 resample_theory=float(P_resample_theory),
                 resample_mc=P_resample_mc,
                 P_hist=P_hist.tolist())
print("V4", {k: v for k, v in res["V4"].items() if k != "P_hist"})

json.dump(res, open("verification_results.json", "w"))
print("done")
