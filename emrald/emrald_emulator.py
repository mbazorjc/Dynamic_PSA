"""
emrald_emulator.py -- re-implements in Python the execution semantics of the EMRALD model of
benchmark configuration B (SDC_midloop_benchmarkB.emrald.json), to explain and quantify the
difference between EMRALD and the NHMC simulator.

EMRALD semantics reproduced:
  * failure, restart and timer events occur at continuous times (exponential / lognormal / 1-h timer);
  * the thermohydraulic variables (flow, critical submergence, level, gas fraction, head multiplier,
    limit states G_A = G_B, coolant temperature) change only at the tick, every `tick` seconds;
  * variable-condition events (G_A<=0, boiling T>=T_sat, G_B>0 gate, G_B<=0) are evaluated on state
    entry and after each tick;
  * on entering a no-flow state the coolant temperature is reset to T0 and then raised by a full
    tick of decay heat at every tick, so boiling is detected at the first tick with T >= T_sat.
As tick -> 0 these semantics approach the continuous-time model that the NHMC simulator
discretises with the exact in-step race (Eq. 6 of the paper).

Usage: python emrald_emulator.py   -> emrald_emulator.json
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nhmc_core import Plant, decay_heat, h_crit, h_set
from nhmc_items import Hyd, theta_mean, head_multiplier, gas_fraction

p, hy = Plant(), Hyd()
th = {k: float(np.ravel(v)[0]) for k, v in theta_mean().items() if k != "w"}
G_ACC = 9.80665
c_Q = (hy.H0 - hy.H_bep) / hy.Q_bep ** 2
k_L = (hy.H_req_bep - hy.H_st) / hy.Q_bep ** 2
HS = h_set(p)
S1, S2H, S2G, S2R, S3A, S3B, DONE = range(7)
CAUSE = {"F_ccf": 0, "F_fts": 1, "F_boil": 2, "F_run": 3, "F_hyd": 4}


def flow(t_after_shutdown):
    return np.clip(decay_heat(t_after_shutdown, p) / (p.rho * p.cp * p.dTc), p.Qmin, p.Qmax)


def run(n, tick_s, seed, Tm=72.0):
    rng = np.random.default_rng(seed)
    dt = tick_s / 3600.0
    K = int(round(Tm / dt))
    lamA = (1 - th["alpha2"]) * th["lam_run"]; lamC = th["alpha2"] * th["lam_run"]
    Z = np.full(n, S1)
    tA = rng.exponential(1 / (lamA + lamC), n)                 # first failure of A
    isC = rng.random(n) < lamC / (lamA + lamC)
    tR = np.full(n, np.inf); tB = np.full(n, np.inf); t1h = np.full(n, np.inf)
    T = np.full(n, p.T0)
    h = HS + p.sigma_h * rng.standard_normal(n)
    G = np.ones(n)
    cause = np.full(n, -1)
    phi = np.exp(-dt / p.tau_h); s_ou = p.sigma_h * np.sqrt(1 - phi ** 2)
    med = {"hw": th["med_h"], "hyd": th["med_a"]}

    def absorb(idx, c):
        Z[idx] = DONE; cause[idx] = CAUSE[c]

    def enter_S2(idx, kind):
        Z[idx] = S2H if kind == "hw" else S2G
        T[idx] = p.T0
        tR[idx] = now[idx] + np.exp(np.log(med[kind]) + p.sig_ln_op * rng.standard_normal(idx.size))

    def try_start(idx):
        fts = rng.random(idx.size) < th["q_fts"]
        absorb(idx[fts], "F_fts")
        ok = idx[~fts]
        Z[ok] = S3A; T[ok] = p.T0
        t1h[ok] = now[ok] + 1.0
        tB[ok] = now[ok] + rng.exponential(1 / th["lam_b1"], ok.size)

    now = np.zeros(n)
    for k in range(1, K + 1):
        tk = k * dt
        # ---- continuous-time events in (t_{k-1}, t_k], processed in time order -----------------
        while True:
            ev_t = np.full(n, np.inf); ev = np.zeros(n, int)
            for code, arr, mask in ((1, tA, Z == S1), (2, tR, (Z == S2H) | (Z == S2G)),
                                    (3, tB, (Z == S3A) | (Z == S3B)), (4, t1h, Z == S3A)):
                cand = np.where(mask, arr, np.inf)
                better = cand < ev_t
                ev_t[better] = cand[better]; ev[better] = code
            act = np.flatnonzero(ev_t <= tk)
            if act.size == 0:
                break
            now[act] = ev_t[act]
            e = ev[act]
            i = act[e == 1]                                   # failure of A
            c = isC[i]; absorb(i[c], "F_ccf"); enter_S2(i[~c], "hw")
            i = act[e == 2]                                   # restart completed
            hw = i[Z[i] == S2H]; try_start(hw)
            g = i[Z[i] == S2G]; Z[g] = S2R; tR[g] = np.inf
            okg = g[G[g] > 0]; try_start(okg)                 # gate evaluated on entry
            i = act[e == 3]; absorb(i, "F_run")               # B fails to run
            i = act[e == 4]                                   # B passes 1 h: switch rate
            t1h[i] = np.inf; tB[i] = now[i] + rng.exponential(1 / th["lam_b2"], i.size)
        # ---- tick: thermohydraulic update -------------------------------------------------------
        now[:] = tk
        Q = float(flow(p.t0 + tk)); Qd = float(decay_heat(p.t0 + tk, p))
        hc = float(h_crit(p.t0 + tk, p))
        jump = rng.random(n) < -np.expm1(-th["nu_j"] * dt)
        h = HS + (h - HS) * phi + s_ou * rng.standard_normal(n) - jump * rng.exponential(th["mu_j"], n)
        a = gas_fraction(h, hc, th["n_e"])
        MH = head_multiplier(a, th["a1"], th["a2"], hy.p_H)
        G = MH * (hy.H0 - c_Q * Q ** 2) - (hy.H_st + k_L * Q ** 2)
        noflow = (Z == S2H) | (Z == S2G) | (Z == S2R)
        T = np.where(noflow, T + Qd * dt * 3600.0 / (th["M"] * p.cp), p.T0)
        # ---- variable-condition events after the tick ------------------------------------------
        i = np.flatnonzero((Z == S1) & (G <= 0)); enter_S2(i, "hyd")
        absorb(np.flatnonzero(noflow & (Z != DONE) & (T >= p.Tsat)), "F_boil")
        i = np.flatnonzero((Z == S2R) & (G > 0)); try_start(i)
        absorb(np.flatnonzero(((Z == S3A) | (Z == S3B)) & (G <= 0)), "F_hyd")
    counts = np.bincount(cause[cause >= 0], minlength=5)
    return dict(n=n, tick_s=tick_s, k=int(counts.sum()), P=float(counts.sum() / n),
                se=float(np.sqrt(counts.sum() / n * (1 - counts.sum() / n) / n)),
                by_cause=(counts / n).tolist())


if __name__ == "__main__":
    out = []
    for tick, n, seed in ((259, 200_000, 11), (60, 200_000, 12), (15, 100_000, 13), (259, 10_000, 2026)):
        t1 = time.perf_counter()
        r = run(n, tick, seed); r["time_s"] = time.perf_counter() - t1
        print(r, flush=True); out.append(r)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "emrald_emulator.json"), "w"), indent=1)
