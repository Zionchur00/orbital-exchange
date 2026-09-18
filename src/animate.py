"""
animate.py -- Phase 4/6: render the 2D "Orbital Exchange" animation.

Reads the selected case from experiments/selected_case.json, re-integrates it
with high-order solve_ivp (the same physics validated in validate.py), and
produces:

    output/orbital_exchange.mp4    (30 fps, ~45 s)
    output/orbital_exchange.gif    (ffmpeg palette, ~30 fps)
    output/keyframes/*.png         (static key frames at narrative moments)

The visualisation is not faked: every plotted position comes from the real
numerical integration.  Lines/markers only convey geometric information
(trajectory, distance, orbital parameters) -- never a made-up force vector.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

from dynamics import (G, M_SUN, SUN_IDX, A_IDX, B_IDX,
                      build_initial_state, integrate_solve_ivp, snapshot,
                      two_body_orbital_energy, distance_AB)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiments"
OUT_DIR = PROJECT_ROOT / "output"
KEY_DIR = OUT_DIR / "keyframes"
OUT_DIR.mkdir(exist_ok=True)
KEY_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# rendering parameters
# --------------------------------------------------------------------------- #
FPS = 30
VIDEO_DURATION = 45.0      # seconds
T_ANIM = 70.0              # code units; same as validate.py reference run
N_FRAMES = int(FPS * VIDEO_DURATION)
DT_FRAME = T_ANIM / N_FRAMES

FRAME_SIZE = (12.8, 7.2)   # 1280 x 720 px at 100 dpi
DPI = 100

# viewing window (matches final_orbits figure; A and encounter stay central)
XLIM = (-8.0, 10.0)
YLIM = (-9.0, 9.0)

# narrative segment thresholds (in code units)
SEGMENTS = [
    (0.0, 4.0, "STABLE ORBIT",
     "Body A moves in a bound orbit around the Sun"),
    (4.0, 7.2, "INCOMING BODY",
     "Body B enters the system on a hyperbolic trajectory"),
    (7.2, 10.5, "GRAVITATIONAL INTERACTION",
     "Close three-body encounter exchanges energy and angular momentum"),
    (10.5, T_ANIM + 1.0, "NEW DYNAMICAL STATE",
     "A and B leave the encounter on altered trajectories"),
]

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pericentre_phase(r, v, body_idx, m):
    """Angle of the eccentricity vector (pericentre direction)."""
    r_rel = r[body_idx] - r[SUN_IDX]
    v_rel = v[body_idx] - v[SUN_IDX]
    mu = G * (m[SUN_IDX] + m[body_idx])
    ev = ((np.dot(v_rel, v_rel) / mu - 1.0 / np.linalg.norm(r_rel)) * r_rel
          - (np.dot(r_rel, v_rel) / mu) * v_rel)
    return float(np.arctan2(ev[1], ev[0]))


def _orbit_ellipse(a, e, phase=0.0, centre=(0.0, 0.0), n=400):
    if a <= 0 or e >= 1:
        return None
    f = np.linspace(0, 2 * np.pi, n)
    r = a * (1 - e * e) / (1 + e * np.cos(f))
    x = r * np.cos(f + phase) + centre[0]
    y = r * np.sin(f + phase) + centre[1]
    return x, y


def _segment_for(t):
    for t0, t1, title, subtitle in SEGMENTS:
        if t0 <= t < t1:
            return title, subtitle
    return SEGMENTS[-1][2], SEGMENTS[-1][3]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    with open(EXP_DIR / "selected_case.json") as f:
        sel = json.load(f)
    P = sel["params"]
    print("selected case:", P)

    r0, v0, m, meta = build_initial_state(r_A=sel.get("r_A", 1.0),
                                          m_A=sel.get("m_A", 1e-3), **P)
    t_frames = np.linspace(0, T_ANIM, N_FRAMES)
    print(f"integrating {N_FRAMES} animation frames (T={T_ANIM})...")
    r, v, t, _ = integrate_solve_ivp(r0, v0, m, (0, T_ANIM), t_frames,
                                      rtol=1e-12, atol=1e-14,
                                      track_energy=False)

    # diagnostics at every frame
    n = len(t)
    E_A = np.empty(n); E_B = np.empty(n); a_A = np.empty(n); e_A = np.empty(n)
    r_AB = np.empty(n); E_tot = np.empty(n); L_z = np.empty(n)
    for i in range(n):
        s = snapshot(r[i], v[i], m)
        E_A[i] = s["E_A"]; E_B[i] = s["E_B"]
        a_A[i] = s["a_A"]; e_A[i] = s["e_A"]
        r_AB[i] = s["r_AB"]
        E_tot[i] = s["E_total"]
        L_z[i] = s["L_total"]
    E_err = np.abs(E_tot - E_tot[0]) / np.abs(E_tot[0])
    L_err = np.abs(L_z - L_z[0]) / np.abs(L_z[0])
    dE_A = E_A[-1] - E_A[0]
    dE_B = E_B[-1] - E_B[0]

    # encounter time for keyframes
    i_enc = int(np.argmin(r_AB))
    t_enc = float(t[i_enc])
    print(f"  encounter at t={t_enc:.2f}, frame {i_enc}")

    # ------------------------------------------------------------------ #
    # setup figure
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=FRAME_SIZE, dpi=DPI)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.1, 1.0],
                  wspace=0.05, left=0.04, right=0.96, top=0.95, bottom=0.06)
    ax = fig.add_subplot(gs[0])
    ax_panel = fig.add_subplot(gs[1])

    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (code units)")
    ax.set_ylabel("y  (code units)")
    ax.set_title("Orbital Exchange", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25)

    # A's initial orbit (dashed, static reference)
    ph0 = _pericentre_phase(r[0], v[0], A_IDX, m)
    ell0 = _orbit_ellipse(a_A[0], e_A[0], phase=ph0)
    if ell0:
        ax.plot(ell0[0], ell0[1], color="#1f77b4", ls="--", lw=1.8,
                alpha=0.55, label="A initial orbit")

    # dynamic artists: trails, bodies, gravity line
    trail_A, = ax.plot([], [], color="#1f77b4", lw=2.0, solid_capstyle="round")
    trail_B, = ax.plot([], [], color="#d62728", lw=2.0, solid_capstyle="round")
    grav_line, = ax.plot([], [], color="magenta", lw=2.5, alpha=0.0,
                         ls="-", solid_capstyle="round")
    sun_dot = ax.scatter([], [], color="orange", s=350, zorder=5,
                         edgecolor="black", linewidth=1.2)
    a_dot = ax.scatter([], [], color="#1f77b4", s=110, zorder=6,
                       edgecolor="black", linewidth=1.0)
    b_dot = ax.scatter([], [], color="#d62728", s=110, zorder=6,
                       edgecolor="black", linewidth=1.0)
    # labels next to bodies (updated each frame)
    txt_A = ax.text(0, 0, "A", color="#1f77b4", fontsize=13, fontweight="bold",
                    ha="left", va="bottom")
    txt_B = ax.text(0, 0, "B", color="#d62728", fontsize=13, fontweight="bold",
                    ha="left", va="bottom")

    # segment banner inside main plot
    banner = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                     fontsize=13, fontweight="bold", color="white",
                     va="top", ha="left",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#333333",
                               ec="white", alpha=0.85))
    banner_sub = ax.text(0.02, 0.90, "", transform=ax.transAxes,
                         fontsize=10, color="white",
                         va="top", ha="left",
                         bbox=dict(boxstyle="round,pad=0.3", fc="#333333",
                                   ec="white", alpha=0.7))

    # final quote overlay (visible in last seconds)
    quote_box = fig.text(0.5, 0.5, "", ha="center", va="center",
                         fontsize=22, fontweight="bold", color="white",
                         wrap=True,
                         bbox=dict(boxstyle="round,pad=0.8", fc="black",
                                   ec="white", alpha=0.75))

    # static legend
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ------------------------------------------------------------------ #
    # info panel
    # ------------------------------------------------------------------ #
    ax_panel.set_xlim(0, 1); ax_panel.set_ylim(0, 1)
    ax_panel.axis("off")
    ax_panel.set_title("INFO PANEL", fontsize=14, fontweight="bold", y=0.98)

    panel_lines = [
        ("TIME", "{time:8.3f}"),
        ("r_AB", "{r_AB:8.4f}"),
        ("", ""),
        ("A orbital energy  E_A", "{E_A:+.3e}"),
        ("B orbital energy  E_B", "{E_B:+.3e}"),
        ("ΔE_A + ΔE_B", "{sum_AB:+.2e}"),
        ("", ""),
        ("A semi-major axis  a_A", "{a_A:8.3f}"),
        ("A eccentricity     e_A", "{e_A:8.3f}"),
        ("", ""),
        ("Total energy error", "{Eerr:8.2e}"),
        ("Total angular-momentum error", "{Lerr:8.2e}"),
    ]

    y_start = 0.92
    dy = 0.072
    panel_texts = []
    panel_vals = []
    for label, fmt in panel_lines:
        if label == "":
            y_start -= dy * 0.35
            panel_texts.append(None)
            panel_vals.append(None)
            continue
        tl = ax_panel.text(0.05, y_start, label, fontsize=11, va="top",
                           ha="left", color="#cccccc")
        tv = ax_panel.text(0.95, y_start, "", fontsize=11, va="top",
                           ha="right", color="white",
                           fontfamily="monospace")
        panel_texts.append(tl); panel_vals.append(tv)
        y_start -= dy

    # explanatory footnote
    ax_panel.text(0.05, 0.05,
                  "E_A and E_B are two-body orbital energies\n"
                  "relative to the Sun (osculating approximation).\n"
                  "Not the full three-body total energy.",
                  fontsize=8, va="bottom", ha="left", color="#999999",
                  transform=ax_panel.transAxes)
    ax_panel.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_panel.transAxes,
                                     fc="#1a1a1a", ec="#444444", lw=2,
                                     zorder=-1))

    # ------------------------------------------------------------------ #
    # update function
    # ------------------------------------------------------------------ #
    def update(i):
        # body positions
        rs = r[i]
        sun_dot.set_offsets([[rs[SUN_IDX, 0], rs[SUN_IDX, 1]]])
        a_dot.set_offsets([[rs[A_IDX, 0], rs[A_IDX, 1]]])
        b_dot.set_offsets([[rs[B_IDX, 0], rs[B_IDX, 1]]])

        # trails (full history; matplotlib clips automatically)
        trail_A.set_data(r[:i+1, A_IDX, 0], r[:i+1, A_IDX, 1])
        trail_B.set_data(r[:i+1, B_IDX, 0], r[:i+1, B_IDX, 1])

        # body labels
        txt_A.set_position((rs[A_IDX, 0] + 0.18, rs[A_IDX, 1] + 0.18))
        txt_B.set_position((rs[B_IDX, 0] + 0.18, rs[B_IDX, 1] + 0.18))

        # A-B gravity connection line; opacity/width encode distance
        # (distance indicator only -- not a fake force arrow)
        rab = r_AB[i]
        if rab < 2.0:
            grav_line.set_data([rs[A_IDX, 0], rs[B_IDX, 0]],
                               [rs[A_IDX, 1], rs[B_IDX, 1]])
            alpha = max(0.15, 1.0 - rab / 2.0)
            grav_line.set_alpha(alpha)
            grav_line.set_linewidth(1.5 + 2.5 * alpha)
        else:
            grav_line.set_alpha(0.0)

        # segment banner
        title, subtitle = _segment_for(t[i])
        banner.set_text(title)
        banner_sub.set_text(subtitle)

        # info panel values
        vals = dict(time=t[i], r_AB=rab,
                    E_A=E_A[i], E_B=E_B[i],
                    sum_AB=E_A[i] + E_B[i] - (E_A[0] + E_B[0]),
                    a_A=a_A[i], e_A=e_A[i],
                    Eerr=E_err[i], Lerr=L_err[i])
        for fmt, tv in zip(panel_lines, panel_vals):
            if tv is None:
                continue
            tv.set_text(fmt[1].format(**vals))
        # colour ΔE_A + ΔE_B to highlight near-zero
        sum_val = vals["sum_AB"]
        for fmt, tl, tv in zip(panel_lines, panel_texts, panel_vals):
            if tv is None or tl is None:
                continue
            if tl.get_text() == "ΔE_A + ΔE_B":
                c = "#2ca02c" if abs(sum_val) < 1e-4 else "white"
                tv.set_color(c)

        # final quote in last ~5 seconds
        if i >= N_FRAMES - 5 * FPS:
            quote_box.set_text(
                'An orbit is not a fixed place.\nIt is a dynamical state.'
            )
        else:
            quote_box.set_text("")

        return (sun_dot, a_dot, b_dot, trail_A, trail_B, grav_line,
                txt_A, txt_B, banner, banner_sub)

    # ------------------------------------------------------------------ #
    # render MP4
    # ------------------------------------------------------------------ #
    print(f"rendering {N_FRAMES} frames for MP4...")
    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000/FPS,
                         blit=False)
    writer = FFMpegWriter(fps=FPS, metadata=dict(title="Orbital Exchange",
                                                 artist="Orbital Exchange",
                                                 comment="Three-body Newtonian dynamics"),
                          codec="libopenh264", bitrate=4000,
                          extra_args=["-pix_fmt", "yuv420p"])
    mp4_path = OUT_DIR / "orbital_exchange.mp4"
    anim.save(str(mp4_path), writer=writer)
    print(f"saved {mp4_path}")

    # ------------------------------------------------------------------ #
    # save keyframes
    # ------------------------------------------------------------------ #
    key_times = [0.0, 4.0, 6.0, t_enc, 12.0, 35.0, T_ANIM]
    key_labels = ["t000_initial", "t004_stable_orbit", "t006_incoming",
                  "t_encounter", "t012_new_state", "t035_new_orbit",
                  "t070_final"]
    print("saving keyframes...")
    for kt, klbl in zip(key_times, key_labels):
        kidx = int(np.clip(np.searchsorted(t_frames, kt), 0, N_FRAMES - 1))
        update(kidx)
        fig.savefig(KEY_DIR / f"{klbl}.png", dpi=DPI)

    plt.close(fig)

    # ------------------------------------------------------------------ #
    # GIF via ffmpeg (high-quality palette)
    # ------------------------------------------------------------------ #
    gif_path = OUT_DIR / "orbital_exchange.gif"
    cmd = (f'ffmpeg -y -i "{mp4_path}" -vf '
           f'"fps={FPS},scale=640:-1:flags=lanczos,split[s0][s1];'
           f'[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" '
           f'"{gif_path}"')
    import subprocess
    print("generating GIF with ffmpeg...")
    subprocess.run(cmd, shell=True, check=True)
    print(f"saved {gif_path}")

    print(f"animation done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
