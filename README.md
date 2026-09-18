# Orbital Exchange — Three-Body Orbital Dynamics Animation

A real Newtonian three-body simulation: a star, a bound body **A**, and an incoming body **B**.
The project demonstrates that an orbit is not a fixed place but a *dynamical state* —
here a close three-body encounter changes A's orbit and scatters B.

---

## 1. What this project demonstrates

- **Real three-body Newtonian dynamics** (no hand-drawn trajectories).
- **Parameter scan** to find a clear, reproducible orbital-change case.
- **Physical validation** before animation: energy, momentum, and angular-momentum
  conservation; step-size sensitivity; cross-check between integrators.
- **Quantified energy exchange**: ΔE_A and ΔE_B are measured and shown to
  approximately cancel, with the small residual carried by the Sun's recoil.
- **2-D animation** with trajectory trails, osculating orbits, distance indicator,
  and a live information panel.

The selected case is a clean **perturbation of A** combined with **gravitational
scattering of B**: A remains bound but on a much larger, more eccentric orbit;
B remains unbound but leaves on a deflected trajectory after donating energy to A.
No true exchange/capture case appeared in the scanned parameter region, which
requires fine-tuned resonant conditions that we did not force.

---

## 2. Physical model

The dynamics is the standard 2-D Newtonian three-body problem:

```
a_i = G Σ_{j≠i} m_j (r_j − r_i) / |r_j − r_i|^3
```

All three bodies move freely in the centre-of-momentum frame
(total linear momentum = 0).  The Sun is the most massive body, but it still
recoils slightly during the encounter.

### Unit system (code units)

To keep values of order unity, we use a self-consistent scale-free system:

| Quantity | Code value | Physical meaning |
|----------|-----------|------------------|
| Gravitational constant `G` | 1 | sets force scale |
| Central star mass `M_sun` | 1 | reference mass |
| Length unit | 1 AU-equivalent | circular orbit at r=1 has v=1 |
| Time unit | chosen so that a circular orbit at r=1 has period 2π | v_circ(r=1)=1 |
| Body masses | m_A = 10⁻³ M_sun, m_B = 4×10⁻² M_sun | small compared to the star |

In these units the gravitational parameter of the star is μ = G M_sun = 1.

### Two-body orbital diagnostics (osculating, relative to the Sun)

Because the Sun is not infinitely massive, A and B feel each other as well as
the star.  We therefore report *osculating two-body elements relative to the Sun*
using the instantaneous Sun-relative position and velocity:

```
ε_A(t) = ½ |v_A − v_sun|² − G(M_sun + m_A)/|r_A − r_sun|
E_A(t) = m_A ε_A(t)
a_A(t) = −G(M_sun + m_A) / (2 ε_A)
e_A(t) = √(1 + 2 ε_A h² / μ²)
```

These are explicitly labelled as **two-body orbital energies relative to the Sun**
and are not the same as the full three-body total energy.

---

## 3. Selected initial conditions

The case was found by an automated scan over `m_B`, `v_inf`, impact parameter,
incoming angle, and A's orbital phase (see `src/search_cases.py`), then verified
independently in `src/validate.py`.

```json
{
  "m_B": 0.04,
  "v_inf": 0.85,
  "b_impact": 0.9,
  "incoming_angle": 0.0,
  "A_phase": 3.0
}
```

Setup details:

- `r_A = 1.0` (A's initial orbital radius before COM correction)
- `m_A = 0.001`
- `m_B = 0.04`
- B enters from far away (`R_in = 8` code units) along the +x direction
  (`incoming_angle = 0`) with impact parameter `b = +0.9`, i.e. it passes
  slightly above the Sun.
- `A_phase = 3.0` rad places A so that it meets B near the encounter point.

### Why this case is a *perturbation + scattering* event

| Criterion | Value | Interpretation |
|-----------|-------|----------------|
| Closest approach `min r_AB` | 0.215 | real close encounter |
| A: initial a, e | 1.014, 0.041 | nearly circular bound orbit |
| A: final a, e | 3.701, 0.737 | bound, much larger & more eccentric |
| B: initial/final specific energy | +0.397 / +0.387 | remains hyperbolic (unbound) |
| ΔE_A (A gains) | +3.58×10⁻⁴ | orbital energy pumped up |
| ΔE_B (B loses) | −3.92×10⁻⁴ | B donates energy |
| ΔE_A + ΔE_B | −3.31×10⁻⁵ | small residual ≈ Sun recoil |

A stays bound (perturbation), B stays unbound but is deflected (scattering).
The energy bookkeeping shows the dominant exchange is between A and B.

---

## 4. Validation (Phase 3)

Integration: `scipy.integrate.solve_ivp` with DOP853, `rtol=1e-12`, `atol=1e-14`.
Cross-checked with a fixed-step velocity-Verlet (symplectic) integrator at
`dt = 0.001, 0.002, 0.004, 0.008`.

### Conservation errors over T = 70 code units

| Quantity | Relative error |
|----------|----------------|
| Total energy | **4.96 × 10⁻¹²** |
| Total angular momentum | **1.23 × 10⁻¹³** |
| Total linear momentum | **1.45 × 10⁻¹³** |

### Step-size / integrator consistency

Final semi-major axis and eccentricity from all tested integrators:

```text
solve_ivp rtol=1e-12 : a_Af = 3.7014, e_Af = 0.7365
solve_ivp rtol=1e-10 : a_Af = 3.7014, e_Af = 0.7365
solve_ivp rtol=1e-08 : a_Af = 3.7014, e_Af = 0.7365
velocity-verlet dt=0.001 : a_Af = 3.7015, e_Af = 0.7365
velocity-verlet dt=0.002 : a_Af = 3.7017, e_Af = 0.7366
velocity-verlet dt=0.004 : a_Af = 3.7024, e_Af = 0.7366
velocity-verlet dt=0.008 : a_Af = 3.7055, e_Af = 0.7368
```

Relative spread of `a_Af` across all runs: **1.1 × 10⁻³**.
Relative spread of `e_Af` across all runs: **4.0 × 10⁻⁴**.

**Conclusion:** the orbital change is real and numerically robust, not a
step-size or integrator artifact.

---

## 5. Outputs

- `output/orbital_exchange.mp4` — 45 s, 30 fps, 1280×720 animation
- `output/orbital_exchange.gif` — 45 s, 30 fps, 640×360 palette-optimised GIF
- `output/keyframes/*.png` — static frames at narrative moments
- `figures/initial_orbits.png` — initial state with A's bound orbit and B asymptote
- `figures/final_orbits.png` — full trajectories and A's *new* osculating orbit
- `figures/energy_exchange.png` — E_A(t), E_B(t), E_total(t), r_AB(t)
- `experiments/selected_case.json` — full selected case + verification metrics
- `experiments/validation_report.json` — conservation / sensitivity report
- `experiments/initial_state.json` / `final_state.json` — raw state vectors

---

## 6. How to re-run

Install dependencies (NumPy, SciPy, Matplotlib, pandas):

```bash
pip install -r requirements.txt
```

Re-run the full pipeline from the project root:

```bash
cd src
python3 search_cases.py   # scan 384 trials, pick best case, save JSON
python3 validate.py       # independent physics verification + figures
python3 animate.py        # render MP4, GIF, and keyframes
```

Each script is self-contained and deterministic.
`search_cases.py` uses DOP853 with `rtol=1e-10` for the scan itself,
so the selected case is physically trustworthy (not a coarse-triage artifact).

---

## 7. Publishing to GitHub

The repository is already initialised and committed locally. To publish, create
an empty repository on GitHub (or via the API) and push:

```bash
cd /workspace/orbital-exchange

# create the remote repo (requires an authenticated GitHub token)
curl -s -X POST -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Content-Type: application/json" \
  https://api.github.com/user/repos \
  -d '{"name":"orbital-exchange","description":"Three-body orbital exchange dynamics","private":true}'

# push
git remote add origin https://oauth2:${GITHUB_TOKEN}@github.com/<your-username>/orbital-exchange.git
git branch -M main
git push -u origin main
```

Set `private` to `false` in the API call for a public repository, and replace
`<your-username>` with your GitHub account name.

Two helper scripts automate the whole process:

```bash
./publish_github.sh      # create the repo, fix the commit author, push main
./create_release.sh      # tag v1.0.0 and attach the packaged .zip archives
```

**Do not commit the large `.zip` archives into the repository.**
Source code goes into git; binary bundles belong in **GitHub Releases** as
downloadable assets, which is exactly what `create_release.sh` does.

---

## 8. Directory structure

```
orbital-exchange/
├── README.md
├── requirements.txt
├── src/
│   ├── dynamics.py        # physics: forces, integrators, orbital diagnostics
│   ├── search_cases.py    # parameter scan + case selection
│   ├── validate.py        # Phase-3 verification + figures
│   └── animate.py         # animation renderer
├── experiments/
│   ├── selected_case.json
│   ├── validation_report.json
│   ├── initial_state.json
│   ├── final_state.json
│   └── scan_results.csv
├── figures/
│   ├── initial_orbits.png
│   ├── final_orbits.png
│   └── energy_exchange.png
└── output/
    ├── orbital_exchange.mp4
    ├── orbital_exchange.gif
    └── keyframes/
```

---

## 9. Notes & caveats

- The two-body energies `E_A`, `E_B` and elements `a_A`, `e_A` are **osculating**
  approximations relative to the Sun.  They are exact only if the Sun were fixed.
  The full three-body energy is conserved to machine precision.
- During the close encounter the A-B mutual potential is non-negligible, so
  `E_A + E_B` is *not* constant at closest approach; it returns to near-constant
  once A and B separate.  The animation panel shows this honestly.
- No Blender is used.  The entire pipeline is Python / NumPy / SciPy /
  Matplotlib / FFmpeg.

---

*“An orbit is not a fixed place. It is a dynamical state.”*

*轨道不是一个静态的位置，而是一种动力学状态。*
