# Dynamic PRA of mid-loop shutdown cooling — simulation code and results

Code, input parameters 

> J.C. Mbazor, P.E. Okafor, J.O. Adibeli, A. Ekedegwa. *Dynamic probabilistic risk assessment of shutdown cooling using a thermohydraulically coupled Markov model.* Submitted to Reliability Engineering & System Safety, 2026.

The model is a Monte Carlo simulation of a non-homogeneous Markov chain on the hybrid state


## Requirements

Python 3.11 or later with NumPy, SciPy and Matplotlib (versions used: Python 3.13, NumPy 2.5,
SciPy 1.18, Matplotlib 3.11).

```bash
pip install -r requirements.txt
```


## Contents

| File | Purpose | Paper |
|---|---|---|
| `run_verification.py` | Verification against closed-form, matrix-exponential and Kolmogorov solutions (V1–V4) | Table 4, Fig. 4 |
| `run_items.py main \| static_curves \| v5v6 \| diag \| sweep` | Static versus dynamic results, decomposition, causes, checks V5–V6, convergence, entry-time and margin sweeps | Tables 4–5, Figs. 5–8 |
| `pub_runs.py cond \| epi \| streams` | Conditional results, uncertainty and sensitivity, replicate streams | Table 6, Figs. 9–12, Suppl. Fig. S1 |
| `dft_vs_mc.py` | Static and dynamic fault-tree comparators | Table 7, Suppl. Table S7 |
| `subset_simulation.py` | Subset simulation on the hardware-and-operator sub-model | Table 7 |
| `make_figures.py`, `make_pub_figures.py` | Figures from the stored results (written to `paper/`) | Figs. 4–13 |
| `emrald/SDC_midloop_benchmarkB.emrald.json` | EMRALD model of benchmark configuration B | Section 6 |
| `emrald/build_emrald_model.py` | Generates the EMRALD model from the same parameters | Section 6 |
| `emrald/emrald_results.json`, `emrald/compare_emrald.py` | EMRALD result ($10^4$ histories) and the two-proportion test | Table 7 |
| `emrald/emrald_emulator.py`, `emrald/emrald_emulator.json` | Effect of EMRALD's fixed update interval | Suppl. Table S9 |

## Usage

```bash
python run_verification.py          
python run_items.py main            
python pub_runs.py cond             
python make_pub_figures.py
```

The EMRALD model runs with the EMRALD solve engine (Idaho National Laboratory,
https://github.com/idaholab/EMRALD, release 3.3 or later, Windows):


## Data sources

Pump failure rates: U.S. nuclear industry averages (NUREG/CR-6928, 2021 update). Common-cause
alpha factors: INL 2024 update. Item shares: OREDA (external-leakage mode, provisional). Plant,
thermohydraulic and operator parameters: Table 2 of the paper; inputs without public data are
treated as assumptions and the results are given conditional on them (Section 5.3).

## Licence and citation

Please cite the paper above. Add a licence file (for example MIT for code and CC BY 4.0 for data)
before public release.
