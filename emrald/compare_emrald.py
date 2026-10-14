"""
compare_emrald.py
=================
Compares EMRALD results for benchmark configuration B with the Python implementation.

Usage:  python compare_emrald.py <EMRALD results .json> [../bench_reference.json]

Reads the EMRALD JSON results (numRuns, keyStates[].name/count), computes the probability of
each loss-of-SDC cause with Clopper-Pearson 95% intervals, and tests agreement with the
Python reference (two-proportion z-test on the total). Writes emrald_results.json, which
make_benchmark_section.py uses to fill the EMRALD row of the benchmark table.
"""
import sys, json, os
import numpy as np
from scipy.stats import beta, norm

KEY = {"F_ccf": "CCF", "F_fts": "B fails to start", "F_boil": "Boiling before restart",
       "F_run": "B item failure (hard)", "F_hyd": "B hydraulic (G<=0)"}


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


res = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "bench_reference.json")))
n = int(res["numRuns"])
counts = {ks["name"]: int(ks["count"]) for ks in res.get("keyStates", [])}
k_tot = sum(counts.get(s, 0) for s in KEY)
P = k_tot / n
lo, hi = cp(k_tot, n)
Pr, ser = ref["n200k"]["P"], ref["n200k"]["se"]
z = (P - Pr) / np.sqrt(P * (1 - P) / n + ser ** 2)
out = dict(numRuns=n, P=P, ci95=[lo, hi], by_cause={KEY[s]: counts.get(s, 0) / n for s in KEY},
           python_ref=Pr, python_se=ser, z=float(z), p_value=float(2 * (1 - norm.cdf(abs(z)))),
           agree_at_5pct=bool(abs(z) < 1.96))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "emrald_results.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
