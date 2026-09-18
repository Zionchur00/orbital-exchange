"""
dynamics.py -- 2D Newtonian three-body physics core for the
"Orbital Exchange" project.

Unit system (documented in README)
----------------------------------
A convenient *self-consistent* set of units is used so that the numbers stay
O(1) and the physics is scale-free (Newtonian gravity is scale-invariant up to
a choice of G, masses, length and time):

    G            = 1
    M_sun        = 1            (central star mass)
    length unit  = 1 AU-equivalent (call it "AU*")
    time unit    = chosen so that a circular orbit at r=1 has period 2*pi
                   (i.e. v_circ = sqrt(G*M_sun/r) = 1  at  r=1)

In these units the gravitational parameter mu = G*M_sun = 1.  Body masses are
fractions of M_sun (m_A, m_B << 1), so the star stays near the centre while
still being a *genuine* three-body system (all three bodies move and interact).

Nothing here is faked for visual effect: accelerations come from the exact
pairwise Newtonian force, integrated with (a) a symplectic velocity-Verlet
scheme for long, stable runs and (b) scipy.solve_ivp (DOP853) as a high-order
reference / cross-check.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------- #
# Constants / units
# --------------------------------------------------------------------------- #
G: float = 1.0          # gravitational constant (code units)
M_SUN: float = 1.0      # central star mass (code units)
SUN_IDX = 0             # index of the Sun in the body array
A_IDX = 1               # index of Body A
B_IDX = 2               # index of Body B


# --------------------------------------------------------------------------- #
# Pure-physics primitives
# --------------------------------------------------------------------------- #
def accelerations(r: np.ndarray, m: np.ndarray, g: float = G) -> np.ndarray:
    """Pairwise Newtonian accelerations.

        a_i = g * sum_{j != i}  m_j * (r_j - r_i) / |r_j - r_i|^3

    Parameters
    ----------
    r : (N, 2) positions
    m : (N,)  masses
    g : gravitational constant

    Returns
    -------
    (N, 2) acceleration array.
    """
    r = np.asarray(r, dtype=float)
    m = np.asarray(m, dtype=float)
    # R[i, j] = r_j - r_i   ->  shape (N, N, 2)
    R = r[np.newaxis, :, :] - r[:, np.newaxis, :]
    dist2 = np.sum(R * R, axis=2)                     # (N, N)
    np.fill_diagonal(dist2, np.inf)                   # no self-force
    inv_d3 = dist2 ** (-1.5)                          # (N, N)
    # a[i, d] = g * sum_j  m[j] * inv_d3[i, j] * R[i, j, d]
    a = g * np.einsum('j,ij,ijd->id', m, inv_d3, R)
    return a


def total_energy(r: np.ndarray, v: np.ndarray, m: np.ndarray,
                 g: float = G) -> float:
    """Total energy of the full N-body system (kinetic + pairwise potential)."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.asarray(m, dtype=float)
    ke = 0.5 * np.sum(m * np.sum(v * v, axis=1))
    # pairwise potential  -G m_i m_j / |r_ij|
    R = r[np.newaxis, :, :] - r[:, np.newaxis, :]
    dist = np.sqrt(np.sum(R * R, axis=2))
    n = len(m)
    iu = np.triu_indices(n, k=1)
    pe = -g * np.sum(np.outer(m, m)[iu] / dist[iu])
    return float(ke + pe)


def total_momentum(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Total linear momentum (2,) vector."""
    v = np.asarray(v, dtype=float)
    m = np.asarray(m, dtype=float)
    return m @ v                          # (2,)


def total_angular_momentum(r: np.ndarray, v: np.ndarray,
                           m: np.ndarray) -> float:
    """Total angular momentum (z-component) about the coordinate origin."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.asarray(m, dtype=float)
    cross = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]      # (N,) z-components
    return float(np.sum(m * cross))


# --------------------------------------------------------------------------- #
# Two-body (relative to Sun) diagnostics
# --------------------------------------------------------------------------- #
# Everything below is the *approximate* two-body description of A or B relative
# to the Sun.  It is NOT the full three-body energy.  We label it explicitly as
# "two-body orbital energy relative to Sun" in the animation / README.
# --------------------------------------------------------------------------- #
def rel_state(r: np.ndarray, v: np.ndarray, body_idx: int,
              sun_idx: int = SUN_IDX):
    """Relative position/velocity of `body_idx` w.r.t. the Sun."""
    return r[body_idx] - r[sun_idx], v[body_idx] - v[sun_idx]


def two_body_specific_energy(r: np.ndarray, v: np.ndarray, body_idx: int,
                             m: np.ndarray, sun_idx: int = SUN_IDX,
                             g: float = G) -> float:
    """Specific orbital energy of `body_idx` treated as a test particle in the
    Sun's field (the standard "relative to Sun" approximation):

        eps = 1/2 |v_rel|^2 - G*M_sun / |r_rel|

    using mu = G*(M_sun + m) for the osculating-element formulas (reduced-mass
    correct 2-body value); since m << M_sun the difference is negligible but
    consistent.  Returned value is energy per unit mass of the body.
    """
    r_rel, v_rel = rel_state(r, v, body_idx, sun_idx)
    mu = g * (m[sun_idx] + m[body_idx])
    return float(0.5 * np.dot(v_rel, v_rel) - mu / np.linalg.norm(r_rel))


def two_body_orbital_energy(r: np.ndarray, v: np.ndarray, body_idx: int,
                            m: np.ndarray, sun_idx: int = SUN_IDX,
                            g: float = G) -> float:
    """Two-body orbital energy of `body_idx` relative to the Sun
    (specific energy * body mass):

        E_body = m_body * [ 1/2 |v_rel|^2 - G*M_sun / |r_rel| ]

    Label in animation: "two-body orbital energy relative to Sun".
    """
    eps = two_body_specific_energy(r, v, body_idx, m, sun_idx, g)
    return float(m[body_idx] * eps)


def osculating_elements(r: np.ndarray, v: np.ndarray, body_idx: int,
                        m: np.ndarray, sun_idx: int = SUN_IDX,
                        g: float = G):
    """Semi-major axis `a` and eccentricity `e` of `body_idx` about the Sun
    from the osculating two-body elements (instantaneous Sun-relative state).

        a = -mu / (2*eps)
        e = sqrt(1 + 2*eps*h^2 / mu^2)      (h = specific ang. mom., z-comp.)

    For unbound (hyperbolic) orbits eps>0 -> a<0, e>1; still reported but the
    animation labels these "if applicable".
    """
    r_rel, v_rel = rel_state(r, v, body_idx, sun_idx)
    mu = g * (m[sun_idx] + m[body_idx])
    eps = 0.5 * np.dot(v_rel, v_rel) - mu / np.linalg.norm(r_rel)
    h = float(r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0])   # specific L_z
    a = -mu / (2.0 * eps) if abs(eps) > 0 else np.inf
    e = float(np.sqrt(max(0.0, 1.0 + 2.0 * eps * h * h / (mu * mu))))
    return float(a), float(e)


def distance_AB(r: np.ndarray) -> float:
    """Instantaneous distance |r_A - r_B|."""
    return float(np.linalg.norm(r[A_IDX] - r[B_IDX]))


# --------------------------------------------------------------------------- #
# Initial-state builder (shared by search / validate / animate)
# --------------------------------------------------------------------------- #
def build_initial_state(r_A: float = 1.0,
                        m_A: float = 1.0e-3,
                        m_B: float = 5.0e-2,
                        v_inf: float = 0.6,
                        b_impact: float = 0.9,
                        incoming_angle: float = 0.0,
                        A_phase: float = 0.0,
                        M_sun: float = M_SUN,
                        g: float = G,
                        R_in: float = 8.0):
    """Construct a physically consistent initial three-body state.

    Layout
    ------
    * Sun      : near origin (slight COM offset), velocity chosen so total
                 linear momentum = 0 (centre-of-momentum frame).
    * Body A   : (near-)circular bound orbit of radius r_A about the Sun,
                 starting at orbital phase A_phase.
    * Body B   : enters from far away (R_in) on a hyperbolic approach toward
                 the Sun with hyperbolic-excess speed v_inf and (signed)
                 impact parameter b_impact; incoming direction set by
                 incoming_angle.

    The "incoming_angle" is the direction (from the Sun) that B comes *from*;
    B is placed at R_in along that direction and launched toward the Sun, with
    a perpendicular offset equal to b_impact (the undeflected miss distance).

    Returns
    -------
    r0 : (3, 2)
    v0 : (3, 2)
    m  : (3,)
    meta : dict of the parameters used (for reproducibility)
    """
    # --- Body A : circular orbit, phase A_phase ------------------------- #
    v_circ_A = np.sqrt(g * M_sun / r_A)
    r_A_vec = r_A * np.array([np.cos(A_phase), np.sin(A_phase)])
    v_A_vec = v_circ_A * np.array([-np.sin(A_phase), np.cos(A_phase)])

    # --- Body B : hyperbolic incoming trajectory ------------------------ #
    alpha = incoming_angle
    e_dir = np.array([np.cos(alpha), np.sin(alpha)])          # "from" direction
    e_perp = np.array([-np.sin(alpha), np.cos(alpha)])
    r_B_vec = R_in * e_dir + b_impact * e_perp
    v_speed = np.sqrt(v_inf**2 + 2.0 * g * M_sun / R_in)      # vis-viva at R_in
    v_B_vec = -v_speed * e_dir                                 # toward the Sun

    # --- Sun : start at rest at origin, then fix COM frame -------------- #
    r_sun = np.array([0.0, 0.0])
    v_sun = np.array([0.0, 0.0])

    m = np.array([M_sun, m_A, m_B])
    r = np.array([r_sun, r_A_vec, r_B_vec])
    v = np.array([v_sun, v_A_vec, v_B_vec])

    # shift to centre-of-momentum frame (total P = 0)
    P = total_momentum(v, m)
    v[SUN_IDX] = -P / m[SUN_IDX]                                # v_sun so P=0
    # also centre positions on COM (keeps Sun essentially at origin)
    com = (m[:, None] * r).sum(axis=0) / m.sum()
    r = r - com

    meta = dict(r_A=r_A, m_A=m_A, m_B=m_B, v_inf=v_inf,
                b_impact=b_impact, incoming_angle=incoming_angle,
                A_phase=A_phase, M_sun=M_sun, G=g, R_in=R_in,
                v_circ_A=v_circ_A, v_speed_at_Rin=v_speed,
                bodies=["Sun", "A", "B"])
    return r, v, m, meta


# --------------------------------------------------------------------------- #
# Integrators
# --------------------------------------------------------------------------- #
def integrate_vv(r0, v0, m, dt, n_steps, record_every=1, g=G,
                 track_energy=False):
    """Velocity-Verlet (symplectic, 2nd-order) fixed-step integrator.

    Returns r_hist (T,3,2), v_hist (T,3,2), t (T,)  and optionally energy (T,).
    """
    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)
    m = np.asarray(m, dtype=float)
    a = accelerations(r, m, g)
    n = len(m)

    n_rec = n_steps // record_every + 1
    r_hist = np.empty((n_rec, n, 2))
    v_hist = np.empty((n_rec, n, 2))
    t_hist = np.empty(n_rec)
    e_hist = np.empty(n_rec) if track_energy else None
    r_hist[0] = r
    v_hist[0] = v
    t_hist[0] = 0.0
    if track_energy:
        e_hist[0] = total_energy(r, v, m, g)
    rec = 1
    for k in range(1, n_steps + 1):
        r = r + v * dt + 0.5 * a * dt * dt
        a_new = accelerations(r, m, g)
        v = v + 0.5 * (a + a_new) * dt
        a = a_new
        if k % record_every == 0:
            r_hist[rec] = r
            v_hist[rec] = v
            t_hist[rec] = k * dt
            if track_energy:
                e_hist[rec] = total_energy(r, v, m, g)
            rec += 1
    r_hist = r_hist[:rec]
    v_hist = v_hist[:rec]
    t_hist = t_hist[:rec]
    if track_energy:
        e_hist = e_hist[:rec]
    return r_hist, v_hist, t_hist, e_hist


def integrate_solve_ivp(r0, v0, m, t_span, t_eval, rtol=1e-12,
                        atol=1e-12 * 1e-2, g=G, track_energy=False):
    """High-order adaptive reference integration (DOP853, 8th order).

    Returns r_hist (T,3,2), v_hist (T,3,2), t (T,) and optionally energy (T,).
    """
    m = np.asarray(m, dtype=float)
    y0 = np.concatenate([np.asarray(r0, dtype=float).ravel(),
                         np.asarray(v0, dtype=float).ravel()])

    def deriv(t, y):
        r = y[:6].reshape(3, 2)
        v = y[6:].reshape(3, 2)
        a = accelerations(r, m, g)
        return np.concatenate([v.ravel(), a.ravel()])

    sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval,
                    method="DOP853", rtol=rtol, atol=atol)
    r_hist = sol.y[:6].T.reshape(-1, 3, 2)
    v_hist = sol.y[6:].T.reshape(-1, 3, 2)
    t_hist = sol.t
    e_hist = None
    if track_energy:
        e_hist = np.array([total_energy(r_hist[i], v_hist[i], m, g)
                           for i in range(len(t_hist))])
    return r_hist, v_hist, t_hist, e_hist


# --------------------------------------------------------------------------- #
# Snapshot diagnostics (used by validate / animate)
# --------------------------------------------------------------------------- #
def snapshot(r, v, m, g=G):
    """Full diagnostic dict for one instantaneous state."""
    E_A = two_body_orbital_energy(r, v, A_IDX, m, SUN_IDX, g)
    E_B = two_body_orbital_energy(r, v, B_IDX, m, SUN_IDX, g)
    eps_A = two_body_specific_energy(r, v, A_IDX, m, SUN_IDX, g)
    eps_B = two_body_specific_energy(r, v, B_IDX, m, SUN_IDX, g)
    a_A, e_A = osculating_elements(r, v, A_IDX, m, SUN_IDX, g)
    a_B, e_B = osculating_elements(r, v, B_IDX, m, SUN_IDX, g)
    return dict(
        E_total=total_energy(r, v, m, g),
        P_total=total_momentum(v, m),
        L_total=total_angular_momentum(r, v, m),
        E_A=E_A, E_B=E_B,
        eps_A=eps_A, eps_B=eps_B,
        a_A=a_A, e_A=e_A,
        a_B=a_B, e_B=e_B,
        r_AB=distance_AB(r),
    )


if __name__ == "__main__":
    # quick self-test: stable circular orbit of A with B far away
    r, v, m, meta = build_initial_state()
    snap0 = snapshot(r, v, m)
    r_h, v_h, t_h, e_h = integrate_vv(r, v, m, dt=0.001, n_steps=20000,
                                       record_every=200, track_energy=True)
    snap1 = snapshot(r_h[-1], v_h[-1], m)
    print("initial  E_total = %.12e" % snap0["E_total"])
    print("final    E_total = %.12e" % snap1["E_total"])
    print("energy drift      = %.3e  (rel)" %
          abs((snap1["E_total"] - snap0["E_total"]) / snap0["E_total"]))
    print("initial  a_A,e_A  = %.5f, %.5f" % (snap0["a_A"], snap0["e_A"]))
    print("final    a_A,e_A  = %.5f, %.5f" % (snap1["a_A"], snap1["e_A"]))
    print("B specific eps0   = %.5f  (>0 hyperbolic incoming)" % snap0["eps_B"])
