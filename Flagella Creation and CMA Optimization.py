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
