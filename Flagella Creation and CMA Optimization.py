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