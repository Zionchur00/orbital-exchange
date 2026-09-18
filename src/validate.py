"""
validate.py -- Phase 3: physics verification of the selected case.

It independently re-integrates the case chosen by search_cases.py and checks
that the orbital change is a *physical* result, not a numerical artifact:

    1. total energy error           (full 3-body, both integrators)
    2. total linear momentum error  (COM frame -> should stay ~0)
    3. total angular momentum error
    4. step-size sensitivity        (VV dt = 0.001 .. 0.008)
    5. integrator consistency       (velocity-Verlet  vs  solve_ivp DOP853)

It also writes the figures required by the deliverables:

    figures/initial_orbits.png     initial state + A's bound orbit + B asymptote
    figures/final_orbits.png       full trajectories + A's *new* orbit
    figures/energy_exchange.png    E_A(t), E_B(t), E_A+E_B ~ const, E_total flat

and the machine-readable state files:

    experiments/initial_state.json
    experiments/final_state.json
    experiments/validation_report.json   (all error metrics)
    experiments/selected_case.json       (re-saved, enriched with verification)

Nothing here modifies the physics -- it only measures it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from dynamics import (G, M_SUN, SUN_IDX, A_IDX, B_IDX,
                      build_initial_state, integrate_vv, integrate_solve_ivp,
                      snapshot, total_energy, total_momentum,
                      total_angular_momentum, two_body_orbital_energy,
                      two_body_specific_energy, osculating_elements,
                      distance_AB)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiments"
FIG_DIR = PROJECT_ROOT / "figures"
EXP_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

T_VERIFY = 70.0
N_EVAL = 4000

# integrators / step sizes used for the consistency & sensitivity tables
VV_STEPS = [0.001, 0.002, 0.004, 0.008]
SOLVE_TOLS = [(1e-12, 1e-14), (1e-10, 1e-12), (1e-8, 1e-10)]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _state_dict(r, v, m, t):
    s = snapshot(r, v, m)
    return dict(t=float(t),
                r=[[float(x) for x in ri] for ri in r],
                v=[[float(x) for x in vi] for vi in v],
                E_total=float(s["E_total"]),
                P_total=[float(x) for x in s["P_total"]],
                L_total=float(s["L_total"]),
                E_A=float(s["E_A"]), E_B=float(s["E_B"]),
                eps_A=float(s["eps_A"]), eps_B=float(s["eps_B"]),
                a_A=float(s["a_A"]), e_A=float(s["e_A"]),
                a_B=float(s["a_B"]), e_B=float(s["e_B"]),
                r_AB=float(s["r_AB"]))


def _orbit_ellipse(a, e, phase=0.0, centre=(0.0, 0.0), n=400):
    """Return (x, y) of a 2-body ellipse of semi-major axis a, eccentricity e,
    rotated so that pericentre is at angle `phase` from the centre."""
    if a <= 0 or e >= 1:
        return None
    b = a * np.sqrt(1.0 - e * e)
    # parametric ellipse, focus at origin: r = a(1-e^2)/(1+e cos f)
    f = np.linspace(0, 2 * np.pi, n)
    r = a * (1 - e * e) / (1 + e * np.cos(f))
    x = r * np.cos(f + phase)
    y = r * np.sin(f + phase)
    return x + centre[0], y + centre[1]


def _pericentre_phase(r, v, body_idx, m):
    """Angle (from Sun) of pericentre of `body_idx`'s osculating orbit."""
    r_rel, v_rel = r[body_idx] - r[SUN_IDX], v[body_idx] - r[SUN_IDX] * 0 + (
        v[body_idx] - v[SUN_IDX])
    # eccentricity vector direction = pericentre direction
    mu = G * (m[SUN_IDX] + m[body_idx])
    ev = (np.dot(v_rel, v_rel) / mu - 1.0 / np.linalg.norm(r_rel)) * r_rel \
        - (np.dot(r_rel, v_rel) / mu) * v_rel
    return float(np.arctan2(ev[1], ev[0]))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    with open(EXP_DIR / "selected_case.json") as f:
        sel = json.load(f)
    P = sel["params"]
    print("selected case params:", P)

    r0, v0, m, meta = build_initial_state(r_A=sel.get("r_A", 1.0),
                                          m_A=sel.get("m_A", 1e-3), **P)
    t_eval = np.linspace(0, T_VERIFY, N_EVAL)

    # ---- reference trajectory (high accuracy solve_ivp) ------------------- #
    r_ref, v_ref, t_ref, e_ref = integrate_solve_ivp(
        r0, v0, m, (0, T_VERIFY), t_eval, rtol=1e-12, atol=1e-14,
        track_energy=True)
    s0 = snapshot(r_ref[0], v_ref[0], m)
    sf = snapshot(r_ref[-1], v_ref[-1], m)

    # ---- time series of diagnostics (used for energy_exchange figure) ---- #
    n = len(t_ref)
    E_A_t = np.empty(n); E_B_t = np.empty(n)
    epsA_t = np.empty(n); epsB_t = np.empty(n)
    aA_t = np.empty(n); eA_t = np.empty(n)
    rAB_t = np.empty(n); Etot_t = np.empty(n)
    Pt = np.empty((n, 2)); Lt = np.empty(n)
    for i in range(n):
        s = snapshot(r_ref[i], v_ref[i], m)
        E_A_t[i] = s["E_A"]; E_B_t[i] = s["E_B"]
        epsA_t[i] = s["eps_A"]; epsB_t[i] = s["eps_B"]
        aA_t[i] = s["a_A"]; eA_t[i] = s["e_A"]
        rAB_t[i] = s["r_AB"]; Etot_t[i] = s["E_total"]
        Pt[i] = s["P_total"]; Lt[i] = s["L_total"]

    # encounter time = when r_AB is minimal
    idx_min = int(np.argmin(rAB_t))
    t_enc = float(t_ref[idx_min])
    rAB_min = float(rAB_t[idx_min])

    # ------------------------------------------------------------------ #
    # 1-3. conservation errors (solve_ivp reference)
    # ------------------------------------------------------------------ #
    E0, Ef = float(e_ref[0]), float(e_ref[-1])
    dE = abs(Ef - E0)
    relE = dE / abs(E0)
    L0, Lf = float(Lt[0]), float(Lt[-1])
    dL = abs(Lf - L0); relL = dL / abs(L0)
    P0 = np.array(Pt[0]); Pf = np.array(Pt[-1])
    dP = float(np.linalg.norm(Pf - P0))
    # relative momentum: P0 ~ 0 in COM frame, so normalise by |m_A v_A0|
    Pscale = m[A_IDX] * np.linalg.norm(v0[A_IDX] - v0[SUN_IDX])
    relP = dP / Pscale

    print("\n=== conservation (solve_ivp DOP853 rtol=1e-12, T=%.0f) ===" % T_VERIFY)
    print("  E_total   : %.12e -> %.12e   |dE|=%.2e  rel=%.2e" % (E0, Ef, dE, relE))
    print("  L_total   : %.12e -> %.12e   |dL|=%.2e  rel=%.2e" % (L0, Lf, dL, relL))
    print("  P_total   : |P0|=%.2e  |Pf|=%.2e   |dP|=%.2e  rel=%.2e"
          % (np.linalg.norm(P0), np.linalg.norm(Pf), dP, relP))

    # ------------------------------------------------------------------ #
    # 4-5. step-size sensitivity & integrator consistency
    # ------------------------------------------------------------------ #
    sens = []
    print("\n=== step-size sensitivity (velocity-Verlet) ===")
    print("  %-10s %-8s %-8s %-10s %-10s %-10s" %
          ("dt", "min_r_AB", "a_Af", "e_Af", "eps_Af", "E_drift"))
    for dt in VV_STEPS:
        rh, vh, th, eh = integrate_vv(r0, v0, m, dt=dt,
                                       n_steps=int(T_VERIFY / dt),
                                       record_every=max(1, int(0.1 / dt)),
                                       track_energy=True)
        s_f = snapshot(rh[-1], vh[-1], m)
        minr = float(min(distance_AB(rr) for rr in rh))
        drift = abs((eh[-1] - eh[0]) / eh[0])
        sens.append(dict(integrator="velocity_verlet", dt=dt,
                         min_r_AB=minr, a_Af=s_f["a_A"], e_Af=s_f["e_A"],
                         eps_Af=s_f["eps_A"], eps_Bf=s_f["eps_B"],
                         E_total_drift=drift))
        print("  %-10.4f %-8.4f %-8.4f %-10.4f %-10.4f %-10.2e" %
              (dt, minr, s_f["a_A"], s_f["e_A"], s_f["eps_A"], drift))

    cons = []
    print("\n=== integrator consistency (solve_ivp DOP853) ===")
    print("  %-12s %-8s %-8s %-10s %-10s %-10s" %
          ("rtol", "min_r_AB", "a_Af", "e_Af", "eps_Af", "E_drift"))
    for rtol, atol in SOLVE_TOLS:
        rh, vh, th, eh = integrate_solve_ivp(
            r0, v0, m, (0, T_VERIFY), t_eval, rtol=rtol, atol=atol,
            track_energy=True)
        s_f = snapshot(rh[-1], vh[-1], m)
        minr = float(min(distance_AB(rr) for rr in rh))
        drift = abs((eh[-1] - eh[0]) / eh[0])
        cons.append(dict(integrator="solve_ivp_DOP853", rtol=rtol, atol=atol,
                         min_r_AB=minr, a_Af=s_f["a_A"], e_Af=s_f["e_A"],
                         eps_Af=s_f["eps_A"], eps_Bf=s_f["eps_B"],
                         E_total_drift=drift))
        print("  %-12.0e %-8.4f %-8.4f %-10.4f %-10.4f %-10.2e" %
              (rtol, minr, s_f["a_A"], s_f["e_A"], s_f["eps_A"], drift))

    # spread of final a_A across all runs -> if tiny, outcome is integrator-indep.
    all_aAf = [s["a_Af"] for s in sens + cons]
    all_eAf = [s["e_Af"] for s in sens + cons]
    spread_a = (max(all_aAf) - min(all_aAf)) / np.mean(all_aAf)
    spread_e = (max(all_eAf) - min(all_eAf)) / max(np.mean(all_eAf), 1e-9)
    print("\n  outcome spread across all integrators/dt:")
    print("    a_Af rel-spread = %.2e   e_Af rel-spread = %.2e"
          % (spread_a, spread_e))
    print("    -> %s (outcome is integrator-independent)" %
          ("ROBUST" if spread_a < 1e-2 and spread_e < 1e-2 else "SENSITIVE"))

    # ------------------------------------------------------------------ #
    # write state + report files
    # ------------------------------------------------------------------ #
    init_state = _state_dict(r_ref[0], v_ref[0], m, 0.0)
    fin_state = _state_dict(r_ref[-1], v_ref[-1], m, T_VERIFY)
    with open(EXP_DIR / "initial_state.json", "w") as f:
        json.dump(init_state, f, indent=2)
    with open(EXP_DIR / "final_state.json", "w") as f:
        json.dump(fin_state, f, indent=2)

    dEA = float(sf["E_A"] - s0["E_A"])
    dEB = float(sf["E_B"] - s0["E_B"])
    report = dict(
        params=P, meta=meta, T_verify=T_VERIFY,
        encounter_time=t_enc, min_r_AB=rAB_min,
        initial=init_state, final=fin_state,
        conservation_solve_ivp=dict(
            E0=E0, Ef=Ef, energy_error_abs=dE, energy_error_rel=relE,
            L0=L0, Lf=Lf, ang_mom_error_abs=dL, ang_mom_error_rel=relL,
            P0=list(P0), Pf=list(Pf),
            momentum_error_abs=dP, momentum_error_rel=relP),
        energy_exchange=dict(
            E_A0=float(s0["E_A"]), E_Af=float(sf["E_A"]), dE_A=dEA,
            E_B0=float(s0["E_B"]), E_Bf=float(sf["E_B"]), dE_B=dEB,
            dE_A_plus_dE_B=dEA + dEB,
            note="two-body orbital energy of each body relative to Sun; "
                 "sum ~0 up to the Sun's kinetic-energy recoil"),
        step_size_sensitivity=sens,
        integrator_consistency=cons,
        outcome_robust=bool(spread_a < 1e-2 and spread_e < 1e-2),
        outcome_spread=dict(a_Af_rel=spread_a, e_Af_rel=spread_e),
    )
    with open(EXP_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # re-save selected_case.json enriched
    sel["verification"] = dict(
        T=T_VERIFY, integrator="solve_ivp_DOP853", rtol=1e-12, atol=1e-14,
        encounter_time=t_enc, min_r_AB=rAB_min,
        initial=init_state, final=fin_state,
        energy_error_rel=relE, ang_mom_error_rel=relL,
        momentum_error_rel=relP,
        energy_exchange=dict(E_A0=float(s0["E_A"]), E_Af=float(sf["E_A"]),
                             dE_A=dEA, E_B0=float(s0["E_B"]),
                             E_Bf=float(sf["E_B"]), dE_B=dEB),
        step_size_sensitivity=sens, integrator_consistency=cons,
        outcome_robust=report["outcome_robust"],
        outcome_spread=dict(a_Af_rel=spread_a, e_Af_rel=spread_e))
    with open(EXP_DIR / "selected_case.json", "w") as f:
        json.dump(sel, f, indent=2)

    # ------------------------------------------------------------------ #
    # FIGURE: energy_exchange.png
    # ------------------------------------------------------------------ #
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw=dict(height_ratios=[3, 1]))
    ax1.axvline(t_enc, color="gray", ls=":", alpha=0.6, label="closest approach")
    ax1.plot(t_ref, E_A_t, color="#1f77b4", lw=2, label=r"$E_A$ (A rel. Sun)")
    ax1.plot(t_ref, E_B_t, color="#d62728", lw=2, label=r"$E_B$ (B rel. Sun)")
    ax1.plot(t_ref, E_A_t + E_B_t, color="#2ca02c", lw=2, ls="--",
             label=r"$E_A + E_B$  ($\approx$ const)")
    ax1.plot(t_ref, Etot_t - Etot_t[0], color="black", lw=1.2, alpha=0.8,
             label=r"$E_{\rm total} - E_{\rm total}(0)$  (3-body)")
    ax1.set_ylabel("energy  (code units)")
    ax1.set_title("Energy exchange during the three-body encounter\n"
                  r"$\Delta E_A = %+.3e$,  $\Delta E_B = %+.3e$,  "
                  r"$\Delta E_A + \Delta E_B = %+.2e$"
                  % (dEA, dEB, dEA + dEB))
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    ax1.grid(alpha=0.3)
    ax1.text(0.02, 0.02,
             "two-body orbital energy relative to Sun\n"
             "(not the full 3-body total energy)",
             transform=ax1.transAxes, fontsize=8, color="0.3",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))

    ax2.plot(t_ref, rAB_t, color="#9467bd", lw=2)
    ax2.set_ylabel(r"$r_{AB}$")
    ax2.set_xlabel("time  (code units)")
    ax2.grid(alpha=0.3)
    ax2.axvline(t_enc, color="gray", ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "energy_exchange.png", dpi=140)
    plt.close(fig)
    print("saved figures/energy_exchange.png")

    # ------------------------------------------------------------------ #
    # FIGURE: initial_orbits.png
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 9))
    # A's initial bound orbit (dashed ellipse)
    ph0 = _pericentre_phase(r_ref[0], v_ref[0], A_IDX, m)
    ell = _orbit_ellipse(s0["a_A"], s0["e_A"], phase=ph0)
    if ell:
        ax.plot(ell[0], ell[1], color="#1f77b4", ls="--", lw=1.5,
                label="A initial orbit (osculating)")
    # B incoming asymptote (line through B position along v_B)
    pB = r0[B_IDX]; vB = v0[B_IDX] - v0[SUN_IDX]
    vBh = vB / np.linalg.norm(vB)
    L = np.linspace(-12, 12, 2)
    ax.plot(pB[0] + L * vBh[0], pB[1] + L * vBh[1],
            color="#d62728", ls="--", lw=1.5, alpha=0.8,
            label="B incoming asymptote")
    # early actual trajectory (first ~3 time units) to show motion direction
    m3 = t_ref < 3.0
    ax.plot(r_ref[m3, A_IDX, 0], r_ref[m3, A_IDX, 1], color="#1f77b4", lw=1.2,
            alpha=0.6)
    ax.plot(r_ref[m3, B_IDX, 0], r_ref[m3, B_IDX, 1], color="#d62728", lw=1.2,
            alpha=0.6)
    # bodies
    ax.scatter([0], [0], color="orange", s=320, zorder=5, edgecolor="black")
    ax.text(0.15, 0.15, "Sun", color="black", fontsize=11)
    ax.scatter([r0[A_IDX, 0]], [r0[A_IDX, 1]], color="#1f77b4", s=90, zorder=5,
               edgecolor="black")
    ax.text(r0[A_IDX, 0] + 0.12, r0[A_IDX, 1] + 0.12, "A", color="#1f77b4",
            fontsize=12, fontweight="bold")
    ax.scatter([r0[B_IDX, 0]], [r0[B_IDX, 1]], color="#d62728", s=90, zorder=5,
               edgecolor="black")
    ax.text(r0[B_IDX, 0] + 0.15, r0[B_IDX, 1] + 0.15, "B", color="#d62728",
            fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-3, 10); ax.set_ylim(-5, 6)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("INITIAL STATE  (t = 0)\n"
                 r"A: bound orbit  a=%.3f, e=%.3f   |   "
                 r"B: hyperbolic incoming  $v_\infty$=%.2f"
                 % (s0["a_A"], s0["e_A"], P["v_inf"]))
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "initial_orbits.png", dpi=140)
    plt.close(fig)
    print("saved figures/initial_orbits.png")

    # ------------------------------------------------------------------ #
    # FIGURE: final_orbits.png
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 9))
    # full actual trajectories
    ax.plot(r_ref[:, A_IDX, 0], r_ref[:, A_IDX, 1], color="#1f77b4", lw=1.6,
            label="A actual trajectory")
    ax.plot(r_ref[:, B_IDX, 0], r_ref[:, B_IDX, 1], color="#d62728", lw=1.6,
            label="B actual trajectory")
    # A's final osculating orbit (the *new* orbit), dashed
    phf = _pericentre_phase(r_ref[-1], v_ref[-1], A_IDX, m)
    ellf = _orbit_ellipse(sf["a_A"], sf["e_A"], phase=phf)
    if ellf:
        ax.plot(ellf[0], ellf[1], color="#1f77b4", ls="--", lw=1.5, alpha=0.8,
                label="A new orbit (osculating, final)")
    # B outgoing asymptote
    pBf = r_ref[-1, B_IDX]; vBf = v_ref[-1, B_IDX] - v_ref[-1, SUN_IDX]
    nrm = np.linalg.norm(vBf)
    if nrm > 0:
        vBfh = vBf / nrm
        L = np.linspace(-25, 25, 2)
        ax.plot(pBf[0] + L * vBfh[0], pBf[1] + L * vBfh[1],
                color="#d62728", ls="--", lw=1.5, alpha=0.8,
                label="B outgoing asymptote")
    ax.scatter([0], [0], color="orange", s=320, zorder=5, edgecolor="black")
    ax.text(0.15, 0.15, "Sun", color="black", fontsize=11)
    ax.scatter([r_ref[-1, A_IDX, 0]], [r_ref[-1, A_IDX, 1]], color="#1f77b4",
               s=90, zorder=5, edgecolor="black")
    ax.text(r_ref[-1, A_IDX, 0] + 0.2, r_ref[-1, A_IDX, 1] + 0.2, "A",
            color="#1f77b4", fontsize=12, fontweight="bold")
    ax.scatter([r_ref[-1, B_IDX, 0]], [r_ref[-1, B_IDX, 1]], color="#d62728",
               s=90, zorder=5, edgecolor="black")
    ax.text(r_ref[-1, B_IDX, 0] + 0.2, r_ref[-1, B_IDX, 1] + 0.2, "B",
            color="#d62728", fontsize=12, fontweight="bold")
    # mark encounter
    ax.scatter([r_ref[idx_min, A_IDX, 0]], [r_ref[idx_min, A_IDX, 1]],
               marker="x", color="black", s=80, zorder=6)
    # frame focused on A's new orbit (B's far escape is clipped; the
    # outgoing asymptote is shown for direction)
    ax.set_aspect("equal")
    ax.set_xlim(-8, 10); ax.set_ylim(-9, 9)
    # clip B actual trajectory so the figure is not stretched
    # (keep A trajectory fully visible; B line is clipped automatically)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("FINAL STATE  (t = %.0f)\n"
                 r"A: NEW orbit  a=%.3f, e=%.3f   ($\Delta a$=%+.2f, $\Delta e$=%+.2f)"
                 "\nB: scattered (still hyperbolic)"
                 % (T_VERIFY, sf["a_A"], sf["e_A"],
                    sf["a_A"] - s0["a_A"], sf["e_A"] - s0["e_A"]))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_orbits.png", dpi=140)
    plt.close(fig)
    print("saved figures/final_orbits.png")

    print("\n=== summary ===")
    print("  encounter @ t=%.2f, min r_AB=%.3f" % (t_enc, rAB_min))
    print("  A: a %.3f->%.3f  e %.3f->%.3f  eps %.4f->%.4f"
          % (s0["a_A"], sf["a_A"], s0["e_A"], sf["e_A"],
             s0["eps_A"], sf["eps_A"]))
    print("  dE_A=%+.3e  dE_B=%+.3e  sum=%+.2e" % (dEA, dEB, dEA + dEB))
    print("  E rel.err=%.2e  L rel.err=%.2e  P rel.err=%.2e"
          % (relE, relL, relP))
    print("  robust=%s  (a spread=%.1e, e spread=%.1e)"
          % (report["outcome_robust"], spread_a, spread_e))
    print("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
