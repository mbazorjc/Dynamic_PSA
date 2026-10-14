"""
build_emrald_model.py
=====================
Generates an EMRALD (INL, https://github.com/idaholab/EMRALD) model of the mid-loop
shutdown-cooling system for the code-to-code benchmark (benchmark configuration B:
mean parameter values, no in-mission wear, no repair; all other dynamic features on).

The model is produced from the same parameter definitions as the Python code
(nhmc_core.Plant, nhmc_items.Hyd, nhmc_items.theta_mean), so the two implementations
cannot drift apart. Output: SDC_midloop_benchmarkB.emrald.json (EMRALD schema v3.4),
validated against the schema shipped with EMRALD if that file is supplied.

Model structure
  Diagram "ThermoHydraulics" : start -> TH_Loop; a timer 'Tick' (dt = 259 s) re-enters
      TH_Loop and runs C# variable updates: decay heat Qd, flow Q, critical submergence
      hc, RCS level h (OU + drain-down jumps), gas fraction alpha, head multiplier MH,
      limit states G_A and G_B, coolant temperature Tcool (heat-up only while NoFlow=1).
  Diagram "PumpSystem" : S1 (A runs) -> S2h | S2g -> (S2g_Ready) -> S3a -> S3b,
      key states F_ccf, F_fts, F_boil, F_run, F_hyd -> terminal Done.
Usage: python build_emrald_model.py [path/to/EMRALD_JsonSchemaV3_4.json]
"""
import sys, os, json, uuid
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from nhmc_core import Plant, h_crit, h_set           # noqa: E402
from nhmc_items import Hyd, theta_mean               # noqa: E402

p, hy = Plant(), Hyd()
th = {k: float(np.asarray(v).ravel()[0]) for k, v in theta_mean().items() if k != "w"}
DT_S = int(os.environ.get("EMRALD_TICK_S", 259))   # tick [s] ~ 72 h / 1000 (override: EMRALD_TICK_S)
DT_H = DT_S / 3600.0
HSET = h_set(p)


_UID_N = [0]


def uid():
    """Deterministic ids (uuid5 of a counter), so that rebuilding gives an identical file."""
    _UID_N[0] += 1
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"SDC_midloop_benchmarkB/{_UID_N[0]}"))


V, S, A, E, D = [], [], [], [], []


def var(name, value, desc=""):
    V.append(dict(id=uid(), name=name, desc=desc, varScope="gtGlobal", value=value, type="double",
                  objType="Variable", resetOnRuns=True))


def action_cng(name, variable, code, used, desc=""):
    A.append(dict(id=uid(), name=name, desc=desc, actType="atCngVarVal", mainItem=True, objType="Action",
                  required=False, mutExcl=True, variableName=variable, scriptCode=code,
                  codeVariables=used, newStates=[]))


def action_goto(name, targets):
    """targets: list of (state, prob) ; prob -1 = remainder."""
    A.append(dict(id=uid(), name=name, desc="", actType="atTransition", mainItem=True, objType="Action",
                  required=False, mutExcl=True, codeVariables=[], variableName="", scriptCode="",
                  newStates=[dict(toState=s, prob=pr, failDesc="", varProb=None) for s, pr in targets]))


def ev_rate(name, lam):
    E.append(dict(id=uid(), name=name, desc="", evType="etFailRate", mainItem=True, required=False,
                  objType="Event", **{"lambda": lam}, lambdaTimeRate="PT1H", useVariable=False))


def ev_varcond(name, code, used):
    E.append(dict(id=uid(), name=name, desc="", evType="etVarCond", mainItem=True, required=False,
                  objType="Event", code=code, varNames=used, triggerStates=[], allItems=True))


def ev_timer(name, iso):
    E.append(dict(id=uid(), name=name, desc="", evType="etTimer", mainItem=True, required=False,
                  objType="Event", triggerStates=[], allItems=True, time=iso, fromSimStart=False))


def ev_lognormal(name, median_h, sig):
    E.append(dict(id=uid(), name=name, desc="", evType="etDistribution", mainItem=True, required=False,
                  objType="Event", distType="dtLogNormal", dfltTimeRate="trHours",
                  parameters=[dict(name="Mean", value=float(np.log(median_h)), timeRate="trHours", useVariable=False),
                              dict(name="Standard Deviation", value=sig, timeRate="trHours", useVariable=False)]))


def state(name, diagram, stype, immediate=(), events=()):
    """events: list of (eventName, [actions], exitParent)"""
    S.append(dict(id=uid(), name=name, desc="", diagramName=diagram, stateType=stype,
                  immediateActions=list(immediate), events=[e for e, _, _ in events],
                  eventActions=[dict(actions=list(a), moveFromCurrent=m) for _, a, m in events],
                  required=False, objType="State", defaultSingleStateValue="Ignore",
                  geometryInfo=dict(x=0, y=0, width=0, height=0)))


# ------------------------------------------------------------------ variables
for name, val in (("Qd", 0.0), ("Q", 0.0), ("hc", 0.0), ("h", HSET), ("alpha", 0.0), ("MH", 1.0),
                  ("G_A", 1.0), ("G_B", 1.0), ("Tcool", p.T0), ("NoFlow", 0.0)):
    var(name, val)

# ------------------------------------------------------------------ thermohydraulic updates (C#)
NE = "MathNet.Numerics.Distributions"
code = {
    "Qd": f"double tau = 3600.0*({p.t0} + CurTime); "
          f"return {p.P0:.6e}*0.066*(Math.Pow(tau,-0.2) - Math.Pow(tau + {p.T_op_h * 3600:.6e},-0.2));",
    "Q": f"return Math.Min(Math.Max(Qd/({p.rho}*{p.cp}*{p.dTc}), {p.Qmin}), {p.Qmax});",
    "hc": f"double d = {p.d}; double Fr = 4.0*Q/(Math.PI*d*d*Math.Sqrt(9.80665*d)); "
          f"return {p.C_oge}*d*Math.Pow(Fr,0.4)*(1.0 + {th['eps_c']});",
    "h": f"double phi = Math.Exp(-{DT_H:.8f}/{p.tau_h}); "
         f"double hn = {HSET:.6f} + (h - {HSET:.6f})*phi + {p.sigma_h}*Math.Sqrt(1.0 - phi*phi)*{NE}.Normal.Sample(Rand, 0.0, 1.0); "
         f"if (Rand.NextDouble() < 1.0 - Math.Exp(-{th['nu_j']:.6e}*{DT_H:.8f})) hn -= {NE}.Exponential.Sample(Rand, 1.0/{th['mu_j']}); "
         f"return hn;",
    "alpha": f"double x = (hc - h)/hc; return x > 0.0 ? Math.Pow(Math.Min(x, 1.0), {th['n_e']}) : 0.0;",
    "MH": f"double x = (alpha - {th['a1']})/({th['a2']} - {th['a1']}); x = Math.Min(1.0, Math.Max(0.0, x)); "
          f"return 1.0 - Math.Pow(x, {hy.p_H});",
    "G_A": f"return MH*({hy.H0} - {hy.a:.6f}*Q*Q) - ({hy.H_st} + {hy.K:.6f}*Q*Q);",
    "G_B": f"return MH*({hy.H0} - {hy.a:.6f}*Q*Q) - ({hy.H_st} + {hy.K:.6f}*Q*Q);",
    "Tcool": f"return NoFlow > 0.5 ? Tcool + Qd*{DT_H:.8f}*3600.0/({th['M']}*{p.cp}) : {p.T0};",
}
used = {"Qd": [], "Q": ["Qd"], "hc": ["Q"], "h": ["h"], "alpha": ["hc", "h"], "MH": ["alpha"],
        "G_A": ["MH", "Q"], "G_B": ["MH", "Q"], "Tcool": ["NoFlow", "Tcool", "Qd"]}
order = ["Qd", "Q", "hc", "h", "alpha", "MH", "G_A", "G_B", "Tcool"]
for vname in order:
    action_cng(f"upd_{vname}", vname, code[vname], used[vname])
action_cng("init_h", "h", f"return {HSET:.6f} + {p.sigma_h}*{NE}.Normal.Sample(Rand, 0.0, 1.0);", [])
action_cng("set_NoFlow", "NoFlow", "return 1.0;", [])
action_cng("clear_NoFlow", "NoFlow", "return 0.0;", [])
action_cng("reset_Tcool", "Tcool", f"return {p.T0};", [])

# ------------------------------------------------------------------ transitions
for tgt in ("TH_Loop", "S1", "S2h", "S2g", "S2g_Ready", "S3a", "S3b",
            "F_ccf", "F_fts", "F_boil", "F_run", "F_hyd", "Done"):
    action_goto(f"Goto_{tgt}", [(tgt, -1)])
action_goto("TryStart", [("F_fts", th["q_fts"]), ("S3a", -1)])

# ------------------------------------------------------------------ events
ev_timer("Tick", f"PT{DT_S}S")
ev_rate("A_HW", (1 - th["alpha2"]) * th["lam_run"])
ev_rate("CCF", th["alpha2"] * th["lam_run"])
ev_varcond("A_Hyd", "return G_A <= 0.0;", ["G_A"])
ev_lognormal("Restart_h", th["med_h"], p.sig_ln_op)
ev_lognormal("Restart_g", th["med_a"], p.sig_ln_op)
ev_varcond("Boil", f"return Tcool >= {p.Tsat};", ["Tcool"])
ev_varcond("GB_ok", "return G_B > 0.0;", ["G_B"])
ev_rate("B_HW1", th["lam_b1"])
ev_rate("B_HW2", th["lam_b2"])
ev_timer("B_1h", "PT1H")
ev_varcond("B_Hyd", "return G_B <= 0.0;", ["G_B"])

# ------------------------------------------------------------------ states
TH, PS = "ThermoHydraulics", "PumpSystem"
state("TH_Start", TH, "stStart", immediate=["init_h", "Goto_TH_Loop"])
state("TH_Loop", TH, "stStandard",
      events=[("Tick", [f"upd_{v}" for v in order] + ["Goto_TH_Loop"], True)])
state("PS_Start", PS, "stStart", immediate=["Goto_S1"])
state("S1", PS, "stStandard", events=[("A_HW", ["Goto_S2h"], True), ("CCF", ["Goto_F_ccf"], True),
                                      ("A_Hyd", ["Goto_S2g"], True)])
for s2, rs in (("S2h", "Restart_h"), ("S2g", "Restart_g")):
    nxt = ["TryStart"] if s2 == "S2h" else ["Goto_S2g_Ready"]
    state(s2, PS, "stStandard", immediate=["set_NoFlow", "reset_Tcool"],
          events=[(rs, nxt, True), ("Boil", ["Goto_F_boil"], True)])
state("S2g_Ready", PS, "stStandard", events=[("GB_ok", ["TryStart"], True), ("Boil", ["Goto_F_boil"], True)])
state("S3a", PS, "stStandard", immediate=["clear_NoFlow", "reset_Tcool"],
      events=[("B_HW1", ["Goto_F_run"], True), ("B_1h", ["Goto_S3b"], True), ("B_Hyd", ["Goto_F_hyd"], True)])
state("S3b", PS, "stStandard", events=[("B_HW2", ["Goto_F_run"], True), ("B_Hyd", ["Goto_F_hyd"], True)])
for f in ("F_ccf", "F_fts", "F_boil", "F_run", "F_hyd"):
    state(f, PS, "stKeyState", immediate=["Goto_Done"])
state("Done", PS, "stTerminal")

D.append(dict(id=uid(), name=TH, desc="RCS thermohydraulic state (decay heat, level, gas, limit states)",
              diagramType="dtMulti", diagramLabel="Plant", objType="Diagram", required=False,
              states=["TH_Start", "TH_Loop"]))
D.append(dict(id=uid(), name=PS, desc="Two SDC pumps, 1 running + 1 standby, shared hot-leg suction",
              diagramType="dtMulti", diagramLabel="System", objType="Diagram", required=False,
              states=[s["name"] for s in S if s["diagramName"] == PS]))

model = dict(id=uid(), objType="EMRALD_Model", name="SDC_midloop_benchmarkB",
             desc=("Mid-loop shutdown cooling, benchmark configuration B (mean parameters, no in-mission "
                   "wear, no repair). Generated by build_emrald_model.py; run 10,000 histories with a "
                   "maximum simulation time of 72 h."),
             emraldVersion=3.4, version=1, versionHistory=[], filename=f"SDC_midloop_benchmarkB{'' if DT_S == 259 else f'_tick{DT_S}'}.emrald.json",
             DiagramList=D, ExtSimList=[], StateList=S, ActionList=A, EventList=E, LogicNodeList=[],
             VariableList=V, templates=[])
SUFFIX = "" if DT_S == 259 else f"_tick{DT_S}"
out = os.path.join(HERE, f"SDC_midloop_benchmarkB{SUFFIX}.emrald.json")
json.dump(model, open(out, "w"), indent=1)
print("written", out, len(S), "states", len(E), "events", len(A), "actions", len(V), "variables")

if len(sys.argv) > 1:
    import jsonschema
    schema = json.load(open(sys.argv[1]))
    errs = sorted(jsonschema.Draft7Validator(schema).iter_errors(model), key=lambda e: list(e.path))
    print("schema errors:", len(errs))
    for e in errs[:20]:
        print("  ", list(e.path), e.message[:160])
