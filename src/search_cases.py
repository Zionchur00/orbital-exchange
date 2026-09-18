"""
search_cases.py -- automatic search for a *real* three-body orbital change.

It does NOT assume any particular parameter set produces orbital exchange.
Instead it scans a grid of initial conditions for Body B (mass, hyperbolic
excess speed, impact parameter, incoming direction) and Body A's orbital
phase, integrates the *full* three-body dynamics, and classifies the outcome:

    perturbation : A bound before & after,  B unbound before & after,  orbits
                   clearly changed.
    scattering   : perturbation where B is strongly deflected.
    capture      : B becomes bound to the Sun (eps_B < 0 afterwards).
    exchange     : A ejected (eps_A > 0 afterwards) AND B captured (bound).
    ejection     : A ejected, B still unbound.

Classification is by the *two-body orbital energy relative to the Sun* of each
body (an approximation, clearly labelled as such in the animation/README); the
full three-body energy is conserved independently and is checked in validate.

A close A-B encounter (min |r_AB| small) is required, otherwise "no encounter".

Outputs
-------
experiments/scan_results.csv        : full table of every trial + classification
experiments/selected_case.json     : the chosen case (re-verified w/ solve_ivp)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from dynamics import (build_initial_state, integrate_vv, integrate_solve_ivp,
                      snapshot, distance_AB, A_IDX, B_IDX, G, M_SUN)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiments"
EXP_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Scan grid (code units; see README for the unit system)
# --------------------------------------------------------------------------- #
GRID = dict(
    m_B            = [0.04, 0.08, 0.13, 0.20],
    v_inf          = [0.45, 0.65, 0.85],
    b_impact       = [-0.9, -0.6, 0.6, 0.9],
    incoming_angle = [0.0, 0.8],
    A_phase        = [0.0, 1.5, 3.0, 4.5],
)
SCAN_DT = 0.004
SCAN_T = 70.0
SCAN_N = int(SCAN_T / SCAN_DT)
SCAN_RTOL = 1e-10        # accurate adaptive integration for *trustworthy* case selection
SCAN_ATOL = 1e-12
CLOSE_ENCOUNTER = 0.45       # min |r_AB| threshold for "encounter happened"

# fixed defaults
R_A_DEFAULT = 1.0
M_A_DEFAULT = 1.0e-3


def classify(s0, sf, min_r_ab):
    """Return (category, summary_dict) for one trial."""
    epsA0, epsAf = s0["eps_A"], sf["eps_A"]
    epsB0, epsBf = s0["eps_B"], sf["eps_B"]
    A_bef = epsA0 < 0
    A_aft = epsAf < 0
    B_bef = epsB0 < 0          # B starts hyperbolic -> False
    B_aft = epsBf < 0
    encounter = min_r_ab < CLOSE_ENCOUNTER

    if not encounter:
        cat = "no_encounter"
    elif (not A_aft) and B_aft:
        cat = "exchange"
    elif B_aft and A_aft:
        cat = "capture"
    elif (not A_aft) and (not B_aft):
        cat = "ejection"
    else:
        cat = "perturbation"

    sumr = dict(
        cat=cat, min_r_ab=min_r_ab,
        epsA0=epsA0, epsAf=epsAf, dEpsA=epsAf - epsA0,
        epsB0=epsB0, epsBf=epsBf, dEpsB=epsBf - epsB0,
        aA0=s0["a_A"], aAf=sf["a_A"], eA0=s0["e_A"], eAf=sf["e_A"],
        aB0=s0["a_B"], aBf=sf["a_B"], eB0=s0["e_B"], eBf=sf["e_B"],
        A_bef=A_bef, A_aft=A_aft, B_bef=B_bef, B_aft=B_aft,
    )
    return cat, sumr


def score(sumr):
    """Rank a trial by how *clearly & visibly* the orbital state changed.

    Goals (in priority order):
      1. a real close A-B encounter happened (min_r_ab small);
      2. A's orbit changes a lot (|deps_A|/|eps_A0| and Delta-eccentricity);
      3. the *altered* orbit stays on screen (apocenter a*(1+e) in ~[2, 7]);
      4. B is also scattered (|deps_B|/|eps_B0|);
      5. near-ejection (eps_A -> 0, apocenter -> huge) is strongly penalised,
         because an orbit that flies off the frame is useless for the animation
         even though it scores high on raw fractional change.
    """
    if sumr["min_r_ab"] >= CLOSE_ENCOUNTER:
        return -1.0
    deA = abs(sumr["dEpsA"]) / max(abs(sumr["epsA0"]), 1e-6)   # 0..1
    de_e = abs(sumr["eAf"] - sumr["eA0"])
    dB = abs(sumr["dEpsB"]) / max(abs(sumr["epsB0"]), 1e-3)
    # visibility: new apocenter should be clearly bigger but on-screen
    apoc_f = sumr["aAf"] * (1.0 + sumr["eAf"]) if sumr["A_aft"] else 1e9
    vis = 0.0
    if sumr["A_aft"]:
        vis = min(apoc_f / 3.0, 1.0)                 # rises up to apoc~3
        if apoc_f > 7.0:
            vis *= max(0.0, 1.0 - (apoc_f - 7.0) / 5.0)   # rolls off >7
    s = 1.5 * deA + 1.5 * de_e + 1.0 * vis + 0.5 * dB
    cat = sumr["cat"]
    mult = dict(exchange=1.6, capture=1.3, perturbation=1.0, ejection=0.7,
                no_encounter=0.0)[cat]
    s *= mult
    # hard penalties
    if sumr["A_aft"] and sumr["aAf"] < 0.15:           # plunging into star
        s *= 0.3
    if sumr["A_aft"] and apoc_f > 9.0:                  # near-ejection / off-frame
        s *= 0.05
    # plunging orbit: pericentre a*(1-e) too small -> A would hit the Sun
    if sumr["A_aft"]:
        peri_f = sumr["aAf"] * (1.0 - sumr["eAf"])
        if peri_f < 0.3:
            s *= 0.1
    # poor conservation during the scan run itself -> unreliable, reject
    if sumr.get("e_drift_scan", 0.0) > 1e-7:
        s *= 0.05
    return s


def run_one(**params):
    r, v, m, meta = build_initial_state(r_A=R_A_DEFAULT, m_A=M_A_DEFAULT,
                                        **params)
    # accurate adaptive integration (DOP853) -- close three-body encounters are
    # chaotic, so a fixed-step coarse triage is NOT reliable for selection.
    t_eval = np.linspace(0, SCAN_T, 800)
    rh, vh, th, eh = integrate_solve_ivp(r, v, m, (0, SCAN_T), t_eval,
                                          rtol=SCAN_RTOL, atol=SCAN_ATOL,
                                          track_energy=True)
    min_r_ab = float(min(distance_AB(rr) for rr in rh))
    s0 = snapshot(rh[0], vh[0], m)
    sf = snapshot(rh[-1], vh[-1], m)
    cat, sumr = classify(s0, sf, min_r_ab)
    # also record energy drift of the *full* 3-body system during the scan run
    e_drift = abs((eh[-1] - eh[0]) / eh[0])
    sumr["e_drift_scan"] = float(e_drift)
    sumr.update(params)
    return sumr, s0, sf, meta


def main():
    t0 = time.time()
    keys = list(GRID)
    grids = [np.asarray(GRID[k]) for k in keys]
    mesh = np.meshgrid(*grids, indexing="ij")
    n_total = int(np.prod([g.size for g in grids]))
    print(f"scanning {n_total} trials  (dt={SCAN_DT}, T={SCAN_T}) ...")
    rows = []
    i = 0
    for idx in range(n_total):
        sub = tuple(int(s) for s in np.unravel_index(idx,
                    [g.size for g in grids]))
        params = {keys[j]: float(grids[j][sub[j]]) for j in range(len(keys))}
        sumr, _, _, _ = run_one(**params)
        rows.append(sumr)
        i += 1
        if i % 60 == 0:
            print(f"  ...{i}/{n_total}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df["score"] = df.apply(lambda r: score(r.to_dict()), axis=1)
    df.to_csv(str(EXP_DIR / "scan_results.csv"), index=False)

    # category counts
    counts = df["cat"].value_counts().to_dict()
    print("\n=== category counts ===")
    for k, v in counts.items():
        print(f"  {k:14s} {v}")

    # top-8 overall by score
    top = df.sort_values("score", ascending=False).head(8)
    cols = ["m_B", "v_inf", "b_impact", "incoming_angle", "A_phase",
            "cat", "min_r_ab", "aA0", "aAf", "eA0", "eAf",
            "epsA0", "epsAf", "epsB0", "epsBf", "score", "e_drift_scan"]
    print("\n=== top-8 by score ===")
    print(top[cols].to_string(index=False))

    # selection: prefer exchange > capture > perturbation (clean), pick best
    chosen = None
    for pref_cat in ["exchange", "capture", "perturbation"]:
        cand = df[(df["cat"] == pref_cat) & (df["score"] > 0)]
        if len(cand):
            chosen = cand.sort_values("score", ascending=False).iloc[0]
            print(f"\n[select] preferred category available: {pref_cat}")
            break
    if chosen is None:
        chosen = df.sort_values("score", ascending=False).iloc[0]
        print("\n[select] fallback to best overall score")

    params = {k: float(chosen[k]) for k in GRID}
    cat = str(chosen["cat"])
    print("\n=== chosen case (scan-level, solve_ivp DOP853 rtol=1e-10) ===")
    print(f"  category      : {cat}")
    print(f"  params        : {params}")
    print(f"  min_r_AB      : {chosen['min_r_ab']:.4f}")
    print(f"  a_A: {chosen['aA0']:.4f} -> {chosen['aAf']:.4f}"
          f"   e_A: {chosen['eA0']:.4f} -> {chosen['eAf']:.4f}")
    print(f"  eps_A: {chosen['epsA0']:.4f} -> {chosen['epsAf']:.4f}"
          f"   eps_B: {chosen['epsB0']:.4f} -> {chosen['epsBf']:.4f}")
    print(f"  scan energy drift (solve_ivp): {chosen['e_drift_scan']:.2e}")

    # ---- re-verify chosen case with high-accuracy solve_ivp ---------------- #
    r, v, m, meta = build_initial_state(r_A=R_A_DEFAULT, m_A=M_A_DEFAULT,
                                        **params)
    t_eval = np.linspace(0, SCAN_T, 1500)
    r_hi, v_hi, t_hi, e_hi = integrate_solve_ivp(
        r, v, m, (0, SCAN_T), t_eval, rtol=1e-12, atol=1e-12,
        track_energy=True)
    s0 = snapshot(r_hi[0], v_hi[0], m)
    sf = snapshot(r_hi[-1], v_hi[-1], m)
    min_r_hi = float(min(distance_AB(rr) for rr in r_hi))
    e_drift_hi = abs((e_hi[-1] - e_hi[0]) / e_hi[0])
    # also a fine VV run for the animation itself
    r_vv, v_vv, t_vv, e_vv = integrate_vv(
        r, v, m, dt=0.002, n_steps=int(SCAN_T / 0.002),
        record_every=20, track_energy=True)
    s0_vv = snapshot(r_vv[0], v_vv[0], m)
    sf_vv = snapshot(r_vv[-1], v_vv[-1], m)
    e_drift_vv = abs((e_vv[-1] - e_vv[0]) / e_vv[0])

    selected = dict(
        params=params,
        meta=meta,
        classification=cat,
        close_encounter_threshold=CLOSE_ENCOUNTER,
        scan=dict(dt=SCAN_DT, T=SCAN_T, min_r_AB=float(chosen["min_r_ab"]),
                  e_drift=float(chosen["e_drift_scan"]),
                  a_A0=float(chosen["aA0"]), a_Af=float(chosen["aAf"]),
                  e_A0=float(chosen["eA0"]), e_Af=float(chosen["eAf"]),
                  eps_A0=float(chosen["epsA0"]), eps_Af=float(chosen["epsAf"]),
                  eps_B0=float(chosen["epsB0"]), eps_Bf=float(chosen["epsBf"])),
        verify_solve_ivp=dict(method="DOP853", rtol=1e-12,
                              min_r_AB=min_r_hi,
                              e_drift=e_drift_hi,
                              a_A0=s0["a_A"], a_Af=sf["a_A"],
                              e_A0=s0["e_A"], e_Af=sf["e_A"],
                              eps_A0=s0["eps_A"], eps_Af=sf["eps_A"],
                              eps_B0=s0["eps_B"], eps_Bf=sf["eps_B"],
                              E_A0=s0["E_A"], E_Af=sf["E_A"],
                              E_B0=s0["E_B"], E_Bf=sf["E_B"]),
        verify_vv_fine=dict(dt=0.002, T=SCAN_T,
                            e_drift=e_drift_vv,
                            a_A0=s0_vv["a_A"], a_Af=sf_vv["a_A"],
                            e_A0=s0_vv["e_A"], e_Af=sf_vv["e_A"]),
        r_A=R_A_DEFAULT, m_A=M_A_DEFAULT,
    )
    print("\n=== re-verification (solve_ivp DOP853 rtol=1e-12) ===")
    print(f"  min_r_AB      : {min_r_hi:.4f}")
    print(f"  E_total drift : {e_drift_hi:.2e} (rel)")
    print(f"  a_A: {s0['a_A']:.4f} -> {sf['a_A']:.4f}"
          f"   e_A: {s0['e_A']:.4f} -> {sf['e_A']:.4f}")
    print(f"  eps_A: {s0['eps_A']:.4f} -> {sf['eps_A']:.4f}"
          f"   eps_B: {s0['eps_B']:.4f} -> {sf['eps_B']:.4f}")
    print(f"  E_A: {s0['E_A']:.5e} -> {sf['E_A']:.5e}   "
          f"dE_A = {sf['E_A']-s0['E_A']:+.5e}")
    print(f"  E_B: {s0['E_B']:.5e} -> {sf['E_B']:.5e}   "
          f"dE_B = {sf['E_B']-s0['E_B']:+.5e}")
    print(f"  dE_A + dE_B  = {(sf['E_A']-s0['E_A'])+(sf['E_B']-s0['E_B']):+.3e}"
          f"   (expected ~0; residual = Sun's recoil)")
    print(f"\nVV(dt=0.002) E_total drift = {e_drift_vv:.2e} (rel)")

    with open(str(EXP_DIR / "selected_case.json"), "w") as f:
        json.dump(selected, f, indent=2)
    print("\nsaved experiments/selected_case.json")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
