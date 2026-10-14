"""
dft_vs_mc.py
============
Computational cost and accuracy of Monte Carlo simulation of a time-varying
(non-homogeneous) CTMC against a standard dynamic fault tree (DFT), for the
two-pump shutdown-cooling system (1 running + 1 cold standby) over a 72-h
mid-loop mission.

Three solutions of the SAME hardware-and-operator model ("model H"):
  1. Standard DFT       : SPARE gate (cold) + FDEP (CCF) + operator basic event,
                          exponential basic events, Markov conversion + expm.
  2. Exact reference    : the same DFT semantics with time-dependent basic events
                          (Weibull item hazards, FTR<=1h / >1h, lognormal restart,
                          decay-heat-dependent time to boil), by numerical integration.
  3. Monte Carlo (CTMC) : 1000-step simulation of the non-homogeneous CTMC,
                          10 000 runs (replicated), plus a high-precision run.

And the FULL coupled model (gas entrainment, level paths, wear; nhmc_items.py),
which a DFT can only approximate through a constant hydraulic basic event.

Item hazards (Weibull, calibrated to the data mean at the reference age):
    h_i(t) = w_i * lambda_pump * ((R + t) / R_ref) ** (beta_i - 1)
which is a Weibull hazard with shape beta_i and scale
eta_i = (beta_i * R_ref**(beta_i-1) / (w_i lambda_pump)) ** (1/beta_i).
Epistemic gamma distributions (NUREG/CR-6928) are used in the 'epistemic' job;
the accuracy comparison is run at the mean parameter values.
Usage:  python dft_vs_mc.py
"""
import json, time
import numpy as np
from scipy.special import ndtr
from scipy.linalg import expm
from scipy.integrate import quad

from nhmc_core import Plant, decay_heat, time_to_boil, lognorm_cdf
from nhmc_items import (ITEMS, theta_mean, item_share_alpha, Hyd, Cfg, simulate_items,
                        hydraulic_rate, static_items_P)

p, hy = Plant(), Hyd()
TH = {k: float(np.asarray(v).ravel()[0]) for k, v in theta_mean().items() if k != "w"}
al = item_share_alpha(); W = al / al.sum()                 # mean OREDA item shares
# Weibull shape parameters (ASSUMED; wear-out items > 1, others exponential)
BETA = np.ones(len(ITEMS))
for name, b in (("Seals", 1.4), ("Impeller", 1.2), ("Bearing", 1.3),
                ("Radial bearing", 1.3), ("Thrust bearing", 1.3)):
    BETA[ITEMS.index(name)] = b
R_REF = 2250.0      # h, reference run-hours at which the data mean applies
R_A = R_B = 2250.0  # h, run-hours of pump A and B at mid-loop entry
T = p.Tm
SIG = p.sig_ln_op


# ----------------------------------------------------------------------------- hazards
def hazA(t):
    """Total hazard of running pump A at mission time t (Weibull items)."""
    t = np.asarray(t, float)
    return TH["lam_run"] * np.sum(W[:, None] * ((R_A + t[None]) / R_REF) ** (BETA[:, None] - 1), 0)


def cumA(t):
    t = np.asarray(t, float)
    return TH["lam_run"] * np.sum((W * R_REF / BETA)[:, None] *
                                  (((R_A + t[None]) / R_REF) ** BETA[:, None] - (R_A / R_REF) ** BETA[:, None]), 0)


def lamB(a):
    """Standby-category rate of B at run age a (FTR<=1h then FTR>1h)."""
    return np.where(np.asarray(a) < 1.0, TH["lam_b1"], TH["lam_b2"])


def hazB(a):
    a = np.asarray(a, float)
    return lamB(a) * np.sum(W[:, None] * ((R_B + a[None]) / R_REF) ** (BETA[:, None] - 1), 0)


def cumB(u):
    """Cumulative hazard of B over run age [0,u] (piecewise rate x Weibull factor)."""
    u = np.atleast_1d(np.asarray(u, float))
    def G(x):   # sum_i W_i R_ref/beta_i [((R_B+x)/R_ref)^beta_i - (R_B/R_ref)^beta_i]
        return np.sum((W * R_REF / BETA)[:, None] *
                      (((R_B + x[None]) / R_REF) ** BETA[:, None] - (R_B / R_REF) ** BETA[:, None]), 0)
    u1 = np.minimum(u, 1.0)
    return TH["lam_b1"] * G(u1) + TH["lam_b2"] * (G(u) - G(u1))


def tb(s):
    """Time to boil [h] if forced flow is lost at mission time s."""
    return float(time_to_boil(p.t0 + s, TH["M"], p))


# ----------------------------------------------------------------------------- 2. exact reference
def exact_reference():
    """P(loss by T) for model H by numerical integration of the DFT semantics."""
    a2, q, med = TH["alpha2"], TH["q_fts"], TH["med_h"]
    f_op = lambda x: np.exp(-0.5 * ((np.log(x) - np.log(med)) / SIG) ** 2) / (x * SIG * np.sqrt(2 * np.pi))

    def p_fail_given_A(s):
        tw = min(tb(s), T - s)
        ok = quad(lambda x: f_op(x) * np.exp(-cumB(T - s - x)[0]), 0, tw, limit=200)[0] if tw > 0 else 0.0
        survive = (1 - q) * ok
        if tb(s) > T - s:                     # mission ends while still waiting, before boiling
            survive += 1 - float(lognorm_cdf(T - s, med, SIG))
        return 1 - survive

    dens = lambda s: float(hazA(s)[0] * np.exp(-cumA(s)[0]))
    P_ccf = quad(lambda s: a2 * dens(s), 0, T, limit=200)[0]
    P_ind = quad(lambda s: (1 - a2) * dens(s) * p_fail_given_A(s), 0, T, limit=200)[0]
    return P_ccf + P_ind


# ----------------------------------------------------------------------------- 1. standard DFT
def standard_dft():
    """
    Galileo-style DFT: TOP = OR( FDEP-CCF , SPARE(A; B) with restart basic event ).
    Exponential basic events at the data means; operator restart as a basic event with
    fixed probability evaluated at POS entry. Solved by conversion to a CTMC + expm.
    States: 0 = A running, 1 = B running (A failed), 2 = TOP.
    """
    lam, a2, q = TH["lam_run"], TH["alpha2"], TH["q_fts"]
    p_late = 1 - float(lognorm_cdf(tb(0.0), TH["med_h"], SIG))
    lbar = (TH["lam_b1"] + TH["lam_b2"] * (T - 1)) / T
    G = np.zeros((3, 3))
    G[0, 1] = (1 - a2) * lam * (1 - q) * (1 - p_late)
    G[0, 2] = a2 * lam + (1 - a2) * lam * (1 - (1 - q) * (1 - p_late))
    G[1, 2] = lbar
    np.fill_diagonal(G, -G.sum(1))
    return float(expm(G * T)[0, 2])


# ----------------------------------------------------------------------------- 3. Monte Carlo
def mc_model_H(n, rng, K=1000):
    """
    Monte Carlo of the time-varying CTMC for model H on K steps.
    States: 0 S1 (A runs), 1 S2 (restart pending, heating), 2 S3 (B runs), 3 failed.
    Returns the per-history failure indicator.
    """
    dt = T / K
    Z = np.zeros(n, np.int8); age = np.zeros(n); Tc = np.full(n, p.T0)
    a2, q, med = TH["alpha2"], TH["q_fts"], TH["med_h"]
    for k in range(K):
        t = k * dt
        U = rng.random(n)
        Zn = Z.copy()
        # S1: exact step probability from the cumulative hazard of A
        i = np.flatnonzero(Z == 0)
        if i.size:
            pex = 1 - np.exp(-(cumA(t + dt)[0] - cumA(t)[0]))
            ex = U[i] < pex
            ccf = ex & (U[i] < a2 * pex)
            Zn[i[ccf]] = 3
            j = i[ex & ~ccf]; Zn[j] = 1; age[j] = 0.0; Tc[j] = p.T0
        # S2: restart race against the boiling boundary (exact within the step)
        i = np.flatnonzero(Z == 1)
        if i.size:
            heat = decay_heat(p.t0 + t, p) * 3600 / (TH["M"] * p.cp)
            tau_b = (p.Tsat - Tc[i]) / heat
            win = np.clip(tau_b, 0, dt)
            a0 = age[i]
            s0 = np.where(a0 > 0, 1 - lognorm_cdf(np.maximum(a0, 1e-12), med, SIG), 1.0)
            s1 = np.where(a0 + win > 0, 1 - lognorm_cdf(np.maximum(a0 + win, 1e-12), med, SIG), 1.0)
            pst = np.where(s0 > 1e-300, 1 - s1 / np.maximum(s0, 1e-300), 1.0)
            u = U[i]
            fts = u < pst * q; ok = (~fts) & (u < pst); boil = (~fts) & (~ok) & (tau_b <= dt)
            Tc[i] += heat * dt
            Zn[i[fts | boil]] = 3
            jj = i[ok]; Zn[jj] = 2; age[jj] = 0.0
        # S3: B runs with age-dependent hazard
        i = np.flatnonzero(Z == 2)
        if i.size:
            pex = 1 - np.exp(-(cumB(age[i] + dt) - cumB(age[i])))
            Zn[i[U[i] < pex]] = 3
        stay = Zn == Z
        age[stay] += dt
        Z = Zn
    return (Z == 3).astype(float)


def summarize(y):
    m = y.mean(); se = y.std(ddof=1) / np.sqrt(y.size)
    return dict(P=float(m), se=float(se), ci=[float(m - 1.96 * se), float(m + 1.96 * se)],
                rel=float(se / m) if m > 0 else float("nan"))


if __name__ == "__main__":
    out = {"assumptions": dict(beta={ITEMS[i]: float(BETA[i]) for i in range(len(ITEMS)) if BETA[i] != 1},
                               R_ref=R_REF, R_A=R_A, R_B=R_B, theta=TH)}

    # ---- standard DFT (repeat for stable timing)
    t1 = time.perf_counter()
    for _ in range(200):
        P_dft = standard_dft()
    t_dft = (time.perf_counter() - t1) / 200
    # ---- exact reference
    t1 = time.perf_counter(); P_ex = exact_reference(); t_ex = time.perf_counter() - t1
    out["model_H"] = dict(P_exact=P_ex, t_exact_s=t_ex, P_dft=P_dft, t_dft_s=t_dft,
                          dft_bias_rel=(P_dft - P_ex) / P_ex)
    print("exact %.4e (%.3fs)  standard DFT %.4e (%.2e s, bias %+.1f%%)" %
          (P_ex, t_ex, P_dft, t_dft, 100 * (P_dft - P_ex) / P_ex), flush=True)

    # ---- Monte Carlo, 10 000 runs, replicated R times
    R = 20; reps = []; times = []
    for r in range(R):
        t1 = time.perf_counter()
        y = mc_model_H(10_000, np.random.default_rng(1000 + r))
        times.append(time.perf_counter() - t1); reps.append(summarize(y))
    Pr = np.array([d["P"] for d in reps])
    cover = np.mean([d["ci"][0] <= P_ex <= d["ci"][1] for d in reps])
    out["mc10k_H"] = dict(replicates=reps, time_s_mean=float(np.mean(times)),
                          rmse_rel=float(np.sqrt(np.mean((Pr - P_ex) ** 2)) / P_ex),
                          mean_of_reps=float(Pr.mean()), ci_coverage=float(cover))
    print("MC 10k x%d: mean %.4e, rel RMSE %.1f%%, %.2fs/run, CI coverage %.0f%%" %
          (R, Pr.mean(), 100 * out["mc10k_H"]["rmse_rel"], np.mean(times), 100 * cover), flush=True)

    # ---- high-precision MC and cost-accuracy curve
    curve = []
    for n in (1_000, 3_000, 10_000, 30_000, 100_000, 300_000):
        t1 = time.perf_counter(); y = mc_model_H(n, np.random.default_rng(7 + n)); tt = time.perf_counter() - t1
        s = summarize(y); s.update(n=n, time_s=tt, err_rel=(s["P"] - P_ex) / P_ex); curve.append(s)
        print("  N=%7d  P=%.4e  rel.se=%.1f%%  err=%+.1f%%  %.2fs" % (n, s["P"], 100 * s["rel"], 100 * s["err_rel"], tt), flush=True)
    out["mc_curve_H"] = curve

    # ---- full coupled model: MC 10k vs DFT with a constant hydraulic basic event
    th = theta_mean()
    t1 = time.perf_counter()
    rf = simulate_items(th, 10_000, p, hy, Cfg(), np.random.default_rng(31))
    t_full = time.perf_counter() - t1
    Pf = float(rf["P"][0]); sef = float(np.sqrt(Pf * (1 - Pf) / 10_000))
    t1 = time.perf_counter()
    lh_av = hydraulic_rate(th, p, hy, True, 20_000, np.random.default_rng(32))
    t_lh = time.perf_counter() - t1
    t1 = time.perf_counter()
    P_dft_full = float(static_items_P(th, lh_av, p, "avg", True).sum())
    P_dft_t0 = float(static_items_P(th, hydraulic_rate(th, p, hy, False, 20_000, np.random.default_rng(33)),
                                    p, "t0", True).sum())
    P_dft_hw = float(static_items_P(th, np.array([0.0]), p, "t0", False).sum())
    t_dft_full = (time.perf_counter() - t1) / 3
    ref = json.load(open("items_diag.json"))["streams"]
    out["full"] = dict(P_mc10k=Pf, se_mc10k=sef, t_mc10k_s=t_full,
                       P_ref=ref["estimate"], se_ref=ref["se"], n_ref=ref["n_total"],
                       P_dft_hyd_avg=P_dft_full, P_dft_hyd_t0=P_dft_t0, P_dft_hw=P_dft_hw,
                       t_dft_s=t_dft_full, t_hydraulic_rate_s=t_lh)
    print("full: MC10k %.4e +- %.1e (%.1fs) | ref %.4e +- %.1e | DFT avg %.4e  t0 %.4e  hw %.4e (%.3fs + %.1fs for lambda_hyd)" %
          (Pf, sef, t_full, ref["estimate"], ref["se"], P_dft_full, P_dft_t0, P_dft_hw, t_dft_full, t_lh), flush=True)
    json.dump(out, open("dft_vs_mc.json", "w"), indent=1)
    print("saved dft_vs_mc.json")
