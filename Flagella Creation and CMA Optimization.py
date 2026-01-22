'''
Implementation of the code to simulate a soft flagella using the Cosserat rod theory. 
CMA-ES was used to optimize the flagella shape for maximum propulsion.

Created by: Jacob LoPiccolo and Manuel Tovar
UIUC ME 447: Computational Design and Dynamics of Soft Systems
Fall 2025
'''


# Shared functions
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

@njit
def log_so3(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    cos_phi = 0.5 * (tr - 1.0)
    if cos_phi > 1.0: cos_phi = 1.0
    if cos_phi < -1.0: cos_phi = -1.0
    phi = np.arccos(cos_phi)
    if phi < 1e-10: return np.zeros(3)
    factor = phi / (2.0 * np.sin(phi))
    return factor * np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])

@njit
def rodrigues_rotation(omega, dt_val):
    theta = np.linalg.norm(omega) * dt_val
    if theta < 1e-12: return np.eye(3)
    u = omega / np.linalg.norm(omega)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + np.sin(theta)*K + (1.0 - np.cos(theta)) * (K @ K)

@njit
def orthonormalize_Q(Q):
    for i in range(Q.shape[0]):
        u, s, vh = np.linalg.svd(Q[i])
        Q[i] = u @ vh

@njit
def cross_product(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])


#!/usr/bin/env python3
""" Calculates the bspline given control points and end points"""

__author__ = "Tejaswin Parthasarathy"
__license__ = "GPL"

import numpy as np


def snake_bspline(t_coeff, l_centerline=1.0, keep_pts=False):
    """ Generates a bspline object that plots the spline interpolant for
    any vector x. Optionally takes in a centerline length, set to 1.0 by
    default and keep_pts for keeping record of control points

    Parameters
    ----------
    t_coeff : np.array
        The spline coefficients, denoted by :math:`beta_i`. Note that the first
        and the last values are set to zero by default.
    l_centreline : float
        The length of the centerline in meters.
    keep_pts : boolean, optional
        If True, we keep record of control point and coefficients at those points
        by returned as a (3, ) tuple. If False, returns only the splibe object.
        Defaults to False.

    Returns
    -------
    spline : scipy.interpolate.Bspline class
        A spline class that can be called as spline(x), where x are the points at
        which the spline needs to be evaluated.
    """
    # Divide into n_control_pts numbrer of points (n_ctr_pts-1) regions
    control_pts = l_centerline * np.linspace(0.0, 1.0, t_coeff.shape[0])

    # Degree of b-spline required. Set to cubic
    degree = 3

    # Update coefficients at the head and tail
    # Setting it to 0.0 here
    beta_head = 0.0
    beta_tail = 0.0

    if keep_pts:
        # Keep record of control point and coefficient at those points
        return __bspline_impl__(control_pts, t_coeff, beta_head, beta_tail,
                                degree)
    else:
        return __bspline_impl__(control_pts, t_coeff, beta_head, beta_tail,
                                degree)[0]


def __bspline_impl__(x_pts, t_c, b_head, b_tail, t_k):
    """
        """
    from scipy.interpolate import BSpline

    # Update the coefficients
    #
    t_c = np.hstack((b_head, t_c, b_tail))

    # Update the knots
    # You need 2 additional knots for the head and tail control points
    # You need degree + 1 additional knots to sink into the head and tail for
    # controlling C^k smoothness. We set it to 0.0
    n_upd = x_pts.shape[0] + 2 + (t_k + 1)

    # The first and last points are always fixed
    x_first = x_pts[0]
    x_last = x_pts[-1]
    x_pts = np.hstack((x_first, x_pts, x_last))

    # Generate the knots
    knots_updated = np.zeros(n_upd, )
    # Sink first degree-1 knots into head
    knots_updated[:t_k - 1] = x_first
    # Middle knot locations correspond to x_locations
    knots_updated[t_k - 1:n_upd - (t_k - 1)] = x_pts
    # Sink first degree-1 knots into tail
    knots_updated[n_upd - (t_k - 1):] = x_last

    return BSpline(knots_updated, t_c, t_k, extrapolate=False), x_pts, t_c


def test_bspline():
    """ Test object for the bspline function"""
    l_centre = 1.0
    t_coeff = np.abs(np.random.randn(6, ))
    my_spline, ctr_pts, ctr_coeffs = snake_bspline(t_coeff, keep_pts=True)
    my_spline(2.0)

    from matplotlib import pyplot as plt
    s = np.linspace(0.0, l_centre, 200)
    plt.plot(s, my_spline(s))
    plt.plot(ctr_pts, ctr_coeffs, 'kx')
    plt.show()

#function creation

# ============================================================================
# FLAGELLUM PHYSICS
# ============================================================================

@njit
def compute_tangents_nodes(r_nodes):
    n_n = r_nodes.shape[0]
    t_nodes = np.zeros((n_n, 3))
    for i in range(n_n - 1):
        diff = r_nodes[i+1] - r_nodes[i]
        nrm = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
        if nrm > 1e-12:
            t_nodes[i,0] = diff[0] / nrm
            t_nodes[i,1] = diff[1] / nrm
            t_nodes[i,2] = diff[2] / nrm
        else:
            t_nodes[i] = np.array([1.0, 0.0, 0.0])
    t_nodes[-1] = t_nodes[-2]
    return t_nodes


@njit
def compute_accelerations_flagella(r_nodes, Q_elems, v_nodes, omega_elems,
                                   S_mat, B_mat, J_elem,
                                   ds, gamma_r,
                                   drag_prefac, gamma_diss,
                                   d3_ref, A_m_elems):
    """
    Cosserat rod balance for the swimming flagellum.

    A_m_elems[i] is the scalar muscular activation amplitude at element i.
    """

    n_e = Q_elems.shape[0]
    n_n = r_nodes.shape[0]

    # ----- 1. Internal forces (same as Timoshenko) -----
    n_internal = np.zeros((n_e, 3))
    strain = np.zeros((n_e, 3))
    for i in range(n_e):
        tangent_g = (r_nodes[i+1] - r_nodes[i]) / ds
        # shear + extension strain in local frame
        tmp0 = Q_elems[i,0,0]*tangent_g[0] + Q_elems[i,1,0]*tangent_g[1] + Q_elems[i,2,0]*tangent_g[2]
        tmp1 = Q_elems[i,0,1]*tangent_g[0] + Q_elems[i,1,1]*tangent_g[1] + Q_elems[i,2,1]*tangent_g[2]
        tmp2 = Q_elems[i,0,2]*tangent_g[0] + Q_elems[i,1,2]*tangent_g[1] + Q_elems[i,2,2]*tangent_g[2]
        strain[i,0] = tmp0 - d3_ref[0]
        strain[i,1] = tmp1 - d3_ref[1]
        strain[i,2] = tmp2 - d3_ref[2]
        n_local = S_mat @ strain[i]
        n_internal[i,0] = n_local[0]
        n_internal[i,1] = n_local[1]
        n_internal[i,2] = n_local[2]

    # ----- 2. Internal moments (free ends) -----
    m_internal_nodes = np.zeros((n_n, 3))
    kappa = np.zeros((n_n, 3))
    # base node has zero internal moment for free BC
    for i in range(1, n_e):
        R_rel = Q_elems[i] @ Q_elems[i-1].T
        kappa[i] = log_so3(R_rel) / ds
        m_local = B_mat @ kappa[i]
        m_internal_nodes[i,0] = m_local[0]
        m_internal_nodes[i,1] = m_local[1]
        m_internal_nodes[i,2] = m_local[2]
    # tip node also left at zero for free BC

    # ----- 3. Linear momentum: ∂_s n + f^H + drag -----
    f_total = np.zeros((n_n, 3))
    n_global = np.zeros((n_e, 3))
    for i in range(n_e):
        # rotate internal force to lab frame
        n_global[i] = Q_elems[i] @ n_internal[i]

    # base
    f_total[0] = n_global[0] / ds
    # interior
    for i in range(1, n_n - 1):
        f_total[i] = (n_global[i] - n_global[i-1]) / ds
    # tip
    f_total[n_n-1] = (-n_global[n_e-1]) / ds

    # hydrodynamic drag per Eq. (11) + linear dissipation
    t_nodes = compute_tangents_nodes(r_nodes)
    for i in range(n_n):
        vi = v_nodes[i]
        ti = t_nodes[i]
        v_par = vi[0]*ti[0] + vi[1]*ti[1] + vi[2]*ti[2]

        # f^H = drag_prefac * (I - 0.5 t t^T) v
        fH0 = drag_prefac * (vi[0] - 0.5 * v_par * ti[0])
        fH1 = drag_prefac * (vi[1] - 0.5 * v_par * ti[1])
        fH2 = drag_prefac * (vi[2] - 0.5 * v_par * ti[2])

        f_total[i,0] += fH0 - gamma_diss * vi[0]
        f_total[i,1] += fH1 - gamma_diss * vi[1]
        f_total[i,2] += fH2 - gamma_diss * vi[2]

    # ----- 4. Angular momentum: ∂_s m + (d3+u)×n + τ_m/L  -----
    m_total = np.zeros((n_e, 3))
    e1 = np.array([0.0, 0.0, 1.0])   # local d3 axis

    for i in range(n_e):
        m_flux = (m_internal_nodes[i+1] - m_internal_nodes[i]) / ds

        # (d3 + u) × n term, with u = strain in local coords
        tangent_local = strain[i] + d3_ref   # ≈ Qᵀ t
        coupling = cross_product(tangent_local, n_internal[i])

        m_total[i] = m_flux + coupling

        # muscular torque density τ_m/L = Q (A_m d1)
        A_i = A_m_elems[i]
        axis_global = Q_elems[i] @ e1     # first column of Q
        m_total[i,0] += A_i * axis_global[0]
        m_total[i,1] += A_i * axis_global[1]
        m_total[i,2] += A_i * axis_global[2]

        # rotational dissipation
        m_total[i,0] -= gamma_r * omega_elems[i,0]
        m_total[i,1] -= gamma_r * omega_elems[i,1]
        m_total[i,2] -= gamma_r * omega_elems[i,2]

    return f_total, m_total


@njit
def time_step_verlet_flagella(r_nodes, v_nodes, Q_elems, omega_elems,
                              S_mat, B_mat, J_elem, m_node,
                              ds, dt, gamma_r,
                              drag_prefac, gamma_diss,
                              d3_ref, A_m_elems):
    """
    Position-Verlet step for the swimming flagellum.
    No clamping: rod is free in space.
    """
    n_e = Q_elems.shape[0]

    # Predictor
    r_half = r_nodes + 0.5 * dt * v_nodes
    Q_half = np.zeros_like(Q_elems)
    for i in range(n_e):
        Q_half[i] = rodrigues_rotation(omega_elems[i], 0.5 * dt) @ Q_elems[i]

    # Forces and torques at half-step (using old velocities)
    f_dens, m_dens = compute_accelerations_flagella(
        r_half, Q_half, v_nodes, omega_elems,
        S_mat, B_mat, J_elem,
        ds, gamma_r,
        drag_prefac, gamma_diss,
        d3_ref, A_m_elems
    )

    # Linear velocity update
    dv = (f_dens * ds) / m_node * dt
    v_new = v_nodes + dv

    # Angular velocity update
    dw = np.zeros_like(omega_elems)
    for i in range(n_e):
        torque = m_dens[i] * ds
        dw[i] = torque / J_elem * dt
    omega_new = omega_elems + dw

    # Corrector
    r_new = r_half + 0.5 * dt * v_new
    Q_new = np.zeros_like(Q_elems)
    for i in range(n_e):
        Q_new[i] = rodrigues_rotation(omega_new[i], 0.5 * dt) @ Q_half[i]

    orthonormalize_Q(Q_new)

    return r_new, v_new, Q_new, omega_new

# ============================================================================
# FLAGELLUM PARAMETERS (from Table 2)
# ============================================================================

rho_f = 1e3        # kg/m^3
E_f = 1e7          # Pa
G_f = 2.0 * E_f / 3.0
L_f = 1.0          # m
r_f = 0.025        # m

A_f  = np.pi * r_f**2
I1_f = np.pi * r_f**4 / 4.0
I2_f = I1_f
I3_f = 2.0 * I1_f

# Cosserat constitutive matrices (consistent with your Timoshenko code)
S_mat_f = np.diag([E_f * A_f, 4.0 * G_f * A_f / 3.0, 4.0 * G_f * A_f / 3.0])
B_mat_f = np.diag([G_f * I3_f, E_f * I2_f, E_f * I1_f])

# Discretization
n_f  = 50
n_nodes_f = n_f + 1
n_elems_f = n_f
ds_f = L_f / n_f

m_node_f = rho_f * A_f * ds_f
J_elem_f = rho_f * np.array([I1_f, I2_f, I3_f]) * ds_f

# Hydrodynamics & muscle parameters
mu_f     = 1e4        # kg m^-1 s^-1
gamma_diss_f = 0.625  # kg m^-1 s^-1 (linear dissipation)
T_m_f    = 1.0        # s (activation period)
dt_f     = 1e-4       # s
gamma_r_f = 0.1       # rotational damping (tunable)

# Slender-body drag prefactor: -4 pi mu / ln(L/r)
drag_prefac_f = -4.0 * np.pi * mu_f / np.log(L_f / r_f)

d3_ref_f = np.array([1.0, 0.0, 0.0])

# ============================================================================
# FLAGELLUM SIMULATION WRAPPER
# ============================================================================

def build_muscle_spline(beta_inner):
    """
    beta_inner: array-like, shape (4,)
        Internal B-spline coefficients (β1..β4). End coefficients β0, β5 are 0.
    Returns a callable beta_fun(s) giving β_m(s) along 0 <= s <= L_f.
    """
    beta_inner = np.asarray(beta_inner, dtype=float)
    # snake_bspline expects array of "non-zero" internal coefficients; it adds 0 at ends.
    beta_fun, _, _ = snake_bspline(beta_inner, keep_pts=True)
    return beta_fun


def simulate_flagellum(beta_inner, lambda_m,
                       n_relax_cycles=2, n_meas_cycles=1,   #attributes unbounded due to
                       max_node_speed=np.inf,          # m/s
                       max_com_speed=np.inf,           # m/s (per cycle scale)
                       max_disp_factor=np.inf,         # max |r - r(0)| / L_f
                       return_ok=False):
    """
    Run flagellum simulation with given muscle spline and wavelength.

    Returns either:
        v_avg                       (if return_ok=False)
    or:
        (v_avg, ok)                 (if return_ok=True)

    ok=False indicates a blowup / unphysical run, in which case v_avg is 0.
    """

    beta_inner = np.asarray(beta_inner, dtype=float)
    beta_fun   = build_muscle_spline(beta_inner)

    # Initial straight rod along x
    r = np.zeros((n_nodes_f, 3))
    for i in range(n_nodes_f):
        r[i] = np.array([i * ds_f, 0.0, 0.0])

    v = np.zeros((n_nodes_f, 3))
    Q = np.zeros((n_elems_f, 3, 3))
    for i in range(n_elems_f):
        Q[i] = np.eye(3)
    omega = np.zeros((n_elems_f, 3))

    total_cycles     = n_relax_cycles + n_meas_cycles
    steps_per_cycle  = int(T_m_f / dt_f)
    total_steps      = total_cycles * steps_per_cycle

    s_elems = (np.arange(n_elems_f) + 0.5) * ds_f

    com_x_history = []
    ok = True

    L_char = L_f
    max_node_disp = max_disp_factor * L_char

    for step in range(total_steps):
        t = step * dt_f

        # muscle torque amplitude A_m(s,t)
        A_m_elems = beta_fun(s_elems) * np.sin(
            2.0 * np.pi * t / T_m_f - 2.0 * np.pi * s_elems / lambda_m
        )

        r, v, Q, omega = time_step_verlet_flagella(
            r, v, Q, omega,
            S_mat_f, B_mat_f, J_elem_f, m_node_f,
            ds_f, dt_f, gamma_r_f,
            drag_prefac_f, gamma_diss_f,
            d3_ref_f, A_m_elems.astype(np.float64)
        )

        # ---- blowup checks ----
        # node speeds
        speed_nodes = np.sqrt(np.sum(v*v, axis=1))
        if np.any(speed_nodes > max_node_speed):
            ok = False
            break

        # node displacements (relative to initial base node)
        disp_nodes = np.sqrt(np.sum((r - r[0])**2, axis=1))
        if np.any(disp_nodes > max_node_disp):
            ok = False
            break

        # track COM over last measurement cycle only
        if step >= (total_cycles - n_meas_cycles) * steps_per_cycle:
            com_x = np.mean(r[:, 0])
            com_x_history.append(com_x)

    if not ok or len(com_x_history) < 2:
        if return_ok:
            return 0.0, False
        else:
            return 0.0

    dx = com_x_history[-1] - com_x_history[0]
    v_avg = dx / (n_meas_cycles * T_m_f)

    # also treat absurd COM speeds as invalid
    if abs(v_avg) > max_com_speed:
        if return_ok:
            return 0.0, False
        else:
            return 0.0

    if return_ok:
        return v_avg, True
    else:
        return v_avg

# ============================================================================
# CMA-ES 
# ============================================================================

rng = np.random.default_rng()

def CMA_es(func, dim, x0=None, sigma0=1.0, popsize=50,
           maxgens=100, tol=1e-12):
    """
    CMA-ES minimizing func(x)

    Returns:
        best_x                 : best solution found
        best_val               : best cost (minimum func value)
        global_best_cost_hist  : best cost seen up to each generation
        gen_best_cost_hist     : best cost in each generation
        gen_mean_cost_hist     : mean cost in each generation
        sighist                : step-size history
        ngen                   : number of generations performed
        bestcoords             : mean vector m at each generation
    """
    if x0 is None:
        x0 = np.zeros(dim)
    m = np.array(x0, dtype=float)

    C = np.eye(dim)
    lam = popsize
    mu = lam // 2
    sigma = sigma0

    # recombination weights
    raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = raw_w / np.sum(raw_w)
    mu_eff = 1.0 / np.sum(w**2)

    # strategy parameters
    c_c = 4.0 / dim
    c_sig = 4.0 / dim
    c1 = 2.0 / (dim**2)
    c_mu = min(1.0 - c1, mu_eff / (dim**2))
    d_sig = 1.0 + np.sqrt(mu_eff / dim)
    chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

    # evolution paths
    p_c = np.zeros(dim)
    p_sig = np.zeros(dim)

    # histories
    sighist = []
    bestcoords = []
    global_best_cost_hist = []
    gen_best_cost_hist = []
    gen_mean_cost_hist = []

    gen = 0
    best_x = m.copy()
    best_val = np.inf

    # progress tracking
    prog_step = max(1, maxgens // 20)  # 5% increments
    next_prog = prog_step

    while gen < maxgens:
        # sampling
        eigvals, B = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        D = np.sqrt(eigvals)
        Z = rng.standard_normal((lam, dim))
        Y = (Z * D) @ B.T
        X = m + sigma * Y

        # evaluate fitness (costs)
        fitness = np.array([func(x) for x in X])
        idx = np.argsort(fitness)
        xsort, ysort = X[idx], Y[idx]

        # per-generation stats
        gen_best_cost = fitness[idx[0]]
        gen_mean_cost = float(np.mean(fitness))
        gen_best_cost_hist.append(gen_best_cost)
        gen_mean_cost_hist.append(gen_mean_cost)

        # global best so far
        if gen == 0:
            global_best_cost = gen_best_cost
        else:
            global_best_cost = min(global_best_cost_hist[-1], gen_best_cost)
        global_best_cost_hist.append(global_best_cost)

        if gen_best_cost < best_val:
            best_val = gen_best_cost
            best_x = xsort[0].copy()

        # recombination
        sel_y = ysort[:mu]
        y_w = np.sum(w[:, None] * sel_y, axis=0)
        m_new = m + sigma * y_w
        bestcoords.append(m_new.copy())

        # sigma path
        inv_sqr_C = B @ np.diag(1.0 / D) @ B.T
        p_sig = (1 - c_sig) * p_sig + np.sqrt(c_sig*(2 - c_sig)*mu_eff) * (inv_sqr_C @ y_w)

        # hsig
        norm_p_sig = np.linalg.norm(p_sig)
        threshold = (1.4 + 2/(dim + 1)) * chi_n
        denom = np.sqrt(1 - (1 - c_sig)**(2 * (gen + 1)))
        hsig = 1 if norm_p_sig / denom < threshold else 0

        # p_c path
        p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c*(2 - c_c)*mu_eff) * y_w

        # C update
        rankmu_term = np.zeros_like(C)
        for i in range(mu):
            yi = sel_y[i]
            rankmu_term += w[i] * np.outer(yi, yi)

        C = (1 - c1 - c_mu) * C \
            + c1 * (np.outer(p_c, p_c) + hsig * c_c * (2 - c_c) * C) \
            + c_mu * rankmu_term

        # sigma update
        sigma = sigma * np.exp((c_sig / d_sig) * (norm_p_sig / chi_n - 1.0))

        m = m_new
        sighist.append(sigma)

        # progress print
        if gen >= next_prog or gen == maxgens - 1:
            pct = 100.0 * gen / maxgens
            print(f"Gen {gen:4d} / {maxgens}  ({pct:5.1f}%) | best f = {best_val:.4e}")
            next_prog += prog_step

        gen += 1

    return (best_x, best_val,
            np.array(global_best_cost_hist),
            np.array(gen_best_cost_hist),
            np.array(gen_mean_cost_hist),
            np.array(sighist),
            gen,
            np.array(bestcoords))

# ============================================================================
# Gait optimization objective: maximize forward COM velocity
# ============================================================================

beta_max = 50.0       # Nm
lambda_min = 0.1      # m 
lambda_max = 1.0      # m

def gait_objective(x):
    """
    x: array-like, shape (5,)
       x[0:4] = β1..β4
       x[4]   = λ_m
    Returns scalar cost to minimize = -average forward velocity.
    """
    x = np.asarray(x, dtype=float)

    # Extract and clip parameters
    beta_inner = np.clip(x[:4], 0, beta_max)
    lambda_m = np.clip(x[4], lambda_min, lambda_max)

    v_avg = simulate_flagellum(beta_inner, lambda_m,
                               n_relax_cycles=2, n_meas_cycles=1)

    # If something goes crazy / unstable, penalize heavily
    if not np.isfinite(v_avg):
        return 1e6
    
    v_abs=abs(v_avg)
    
    return -v_abs  # maximize v_avg by minimizing -v_avg
