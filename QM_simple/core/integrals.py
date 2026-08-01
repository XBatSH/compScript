"""
integrals.py -- One- and two-electron integrals over Gaussian basis functions.

We need four kinds of integrals for Hartree-Fock:

    S_uv  = <u|v>                    overlap
    T_uv  = <u| -1/2 nabla^2 |v>     kinetic energy
    V_uv  = <u| -sum_A Z_A/r_A |v>   nuclear attraction
    (uv|ls) = two-electron repulsion integrals (ERIs)

Everything is evaluated with the McMurchie-Davidson scheme
(J. Comput. Phys. 26, 218 (1978)):

  1. The product of two Gaussians is expanded in *Hermite* Gaussians with
     coefficients E_t^{ij} obtained from a 3-term recursion.
  2. Coulomb integrals over Hermite Gaussians reduce to the Boys function
     F_n(x), which scipy gives us via the confluent hypergeometric 1F1.

This is the slow-but-clear textbook route: pure Python loops, no tricks.
Reference: Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure
Theory", chapter 9.
"""

from __future__ import annotations

import numpy as np
from scipy.special import hyp1f1


# --------------------------------------------------------------------- #
# Boys function
# --------------------------------------------------------------------- #
def boys(n, x):
    """Boys function F_n(x) = integral_0^1 t^{2n} exp(-x t^2) dt.

    Identity used:  F_n(x) = 1F1(n + 1/2; n + 3/2; -x) / (2n + 1)
    (scipy's hyp1f1 handles x = 0 and large x gracefully).
    """
    return hyp1f1(n + 0.5, n + 1.5, -x) / (2.0 * n + 1.0)


# --------------------------------------------------------------------- #
# Hermite expansion coefficients E_t^{ij}  (one Cartesian direction)
# --------------------------------------------------------------------- #
def E(i, j, t, Qx, a, b):
    """Recursive Hermite expansion coefficient.

    i, j : Cartesian powers on Gaussian a and b
    t    : Hermite Gaussian order
    Qx   : distance Ax - Bx between the two centers (this direction)
    a, b : Gaussian exponents
    """
    p = a + b
    q = a * b / p
    if t < 0 or t > i + j:
        return 0.0                       # out of bounds
    if i == j == t == 0:
        return np.exp(-q * Qx * Qx)      # Gaussian product theorem prefactor
    if j == 0:                           # decrement index i
        return (E(i - 1, j, t - 1, Qx, a, b) / (2 * p)
                - q * Qx / a * E(i - 1, j, t, Qx, a, b)
                + (t + 1) * E(i - 1, j, t + 1, Qx, a, b))
    # decrement index j
    return (E(i, j - 1, t - 1, Qx, a, b) / (2 * p)
            + q * Qx / b * E(i, j - 1, t, Qx, a, b)
            + (t + 1) * E(i, j - 1, t + 1, Qx, a, b))


# --------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------- #
def _overlap_prim(a, lmn1, A, b, lmn2, B):
    """<g_a|g_b> for two primitive Gaussians (unnormalized)."""
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    s_x = E(l1, l2, 0, A[0] - B[0], a, b)
    s_y = E(m1, m2, 0, A[1] - B[1], a, b)
    s_z = E(n1, n2, 0, A[2] - B[2], a, b)
    return s_x * s_y * s_z * (np.pi / (a + b)) ** 1.5


def overlap(bf1, bf2):
    """<phi_1|phi_2> for two contracted basis functions."""
    s = 0.0
    for ca, a in zip(bf1.coefs, bf1.exps):
        for cb, b in zip(bf2.coefs, bf2.exps):
            s += ca * cb * _overlap_prim(a, bf1.lmn, bf1.center,
                                         b, bf2.lmn, bf2.center)
    return s


# --------------------------------------------------------------------- #
# Kinetic energy
# --------------------------------------------------------------------- #
def _kinetic_prim(a, lmn1, A, b, lmn2, B):
    """<g_a| -1/2 nabla^2 |g_b>, written as a sum of overlaps.

    Differentiating a Gaussian twice shifts its Cartesian power by +-2,
    so the kinetic integral is a linear combination of overlap integrals.
    """
    l2, m2, n2 = lmn2
    term0 = b * (2 * (l2 + m2 + n2) + 3) * _overlap_prim(a, lmn1, A, b, lmn2, B)
    term1 = -2 * b ** 2 * (
        _overlap_prim(a, lmn1, A, b, (l2 + 2, m2, n2), B)
        + _overlap_prim(a, lmn1, A, b, (l2, m2 + 2, n2), B)
        + _overlap_prim(a, lmn1, A, b, (l2, m2, n2 + 2), B))
    term2 = -0.5 * (
        l2 * (l2 - 1) * _overlap_prim(a, lmn1, A, b, (l2 - 2, m2, n2), B)
        + m2 * (m2 - 1) * _overlap_prim(a, lmn1, A, b, (l2, m2 - 2, n2), B)
        + n2 * (n2 - 1) * _overlap_prim(a, lmn1, A, b, (l2, m2, n2 - 2), B))
    return term0 + term1 + term2


def kinetic(bf1, bf2):
    t = 0.0
    for ca, a in zip(bf1.coefs, bf1.exps):
        for cb, b in zip(bf2.coefs, bf2.exps):
            t += ca * cb * _kinetic_prim(a, bf1.lmn, bf1.center,
                                         b, bf2.lmn, bf2.center)
    return t


# --------------------------------------------------------------------- #
# Hermite Coulomb integrals R_{tuv}
# --------------------------------------------------------------------- #
def R(t, u, v, n, p, PCx, PCy, PCz, RPC):
    """Auxiliary Hermite Coulomb integral (recursive, downward in t,u,v)."""
    if t == u == v == 0:
        return (-2.0 * p) ** n * boys(n, p * RPC * RPC)
    if t > 0:
        val = PCx * R(t - 1, u, v, n + 1, p, PCx, PCy, PCz, RPC)
        if t > 1:
            val += (t - 1) * R(t - 2, u, v, n + 1, p, PCx, PCy, PCz, RPC)
        return val
    if u > 0:
        val = PCy * R(t, u - 1, v, n + 1, p, PCx, PCy, PCz, RPC)
        if u > 1:
            val += (u - 1) * R(t, u - 2, v, n + 1, p, PCx, PCy, PCz, RPC)
        return val
    # v > 0
    val = PCz * R(t, u, v - 1, n + 1, p, PCx, PCy, PCz, RPC)
    if v > 1:
        val += (v - 1) * R(t, u, v - 2, n + 1, p, PCx, PCy, PCz, RPC)
    return val


# --------------------------------------------------------------------- #
# Nuclear attraction
# --------------------------------------------------------------------- #
def _nuclear_prim(a, lmn1, A, b, lmn2, B, C):
    """<g_a| 1/|r - C| |g_b> for primitives; C is the nuclear position."""
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    p = a + b
    P = (a * A + b * B) / p              # Gaussian product center
    PC = P - C
    RPC = np.linalg.norm(PC)

    val = 0.0
    for t in range(l1 + l2 + 1):
        Ex = E(l1, l2, t, A[0] - B[0], a, b)
        if Ex == 0.0:
            continue
        for u in range(m1 + m2 + 1):
            Ey = E(m1, m2, u, A[1] - B[1], a, b)
            if Ey == 0.0:
                continue
            for v in range(n1 + n2 + 1):
                Ez = E(n1, n2, v, A[2] - B[2], a, b)
                if Ez == 0.0:
                    continue
                val += Ex * Ey * Ez * R(t, u, v, 0, p,
                                        PC[0], PC[1], PC[2], RPC)
    return 2.0 * np.pi / p * val


def nuclear_attraction(bf1, bf2, molecule):
    """V_uv = -sum_A Z_A <u| 1/r_A |v>  (note the attractive minus sign)."""
    v = 0.0
    for atom in molecule.atoms:
        for ca, a in zip(bf1.coefs, bf1.exps):
            for cb, b in zip(bf2.coefs, bf2.exps):
                v -= atom.Z * ca * cb * _nuclear_prim(
                    a, bf1.lmn, bf1.center, b, bf2.lmn, bf2.center, atom.coord)
    return v


# --------------------------------------------------------------------- #
# Two-electron repulsion integrals (chemists' notation (ab|cd))
# --------------------------------------------------------------------- #
def _eri_prim(a, lmn1, A, b, lmn2, B, c, lmn3, C, d, lmn4, D):
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    l3, m3, n3 = lmn3
    l4, m4, n4 = lmn4
    p = a + b                            # bra pair exponent
    q = c + d                            # ket pair exponent
    alpha = p * q / (p + q)
    P = (a * A + b * B) / p
    Q = (c * C + d * D) / q
    PQ = P - Q
    RPQ = np.linalg.norm(PQ)

    val = 0.0
    for t in range(l1 + l2 + 1):
        E1x = E(l1, l2, t, A[0] - B[0], a, b)
        for u in range(m1 + m2 + 1):
            E1y = E(m1, m2, u, A[1] - B[1], a, b)
            for v in range(n1 + n2 + 1):
                E1z = E(n1, n2, v, A[2] - B[2], a, b)
                bra = E1x * E1y * E1z
                if bra == 0.0:
                    continue
                for tau in range(l3 + l4 + 1):
                    E2x = E(l3, l4, tau, C[0] - D[0], c, d)
                    for nu in range(m3 + m4 + 1):
                        E2y = E(m3, m4, nu, C[1] - D[1], c, d)
                        for phi in range(n3 + n4 + 1):
                            E2z = E(n3, n4, phi, C[2] - D[2], c, d)
                            ket = E2x * E2y * E2z
                            if ket == 0.0:
                                continue
                            val += (bra * ket * (-1.0) ** (tau + nu + phi)
                                    * R(t + tau, u + nu, v + phi, 0, alpha,
                                        PQ[0], PQ[1], PQ[2], RPQ))
    val *= 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
    return val


def electron_repulsion(bf1, bf2, bf3, bf4):
    """(12|34) for four contracted basis functions."""
    val = 0.0
    for c1, a1 in zip(bf1.coefs, bf1.exps):
        for c2, a2 in zip(bf2.coefs, bf2.exps):
            for c3, a3 in zip(bf3.coefs, bf3.exps):
                for c4, a4 in zip(bf4.coefs, bf4.exps):
                    val += c1 * c2 * c3 * c4 * _eri_prim(
                        a1, bf1.lmn, bf1.center, a2, bf2.lmn, bf2.center,
                        a3, bf3.lmn, bf3.center, a4, bf4.lmn, bf4.center)
    return val


# --------------------------------------------------------------------- #
# Whole-molecule integral arrays
# --------------------------------------------------------------------- #
def build_one_electron(basis, molecule):
    """Return the S, T, V matrices (all symmetric, size nbf x nbf)."""
    n = len(basis)
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):           # only lower triangle, then mirror
            S[i, j] = S[j, i] = overlap(basis[i], basis[j])
            T[i, j] = T[j, i] = kinetic(basis[i], basis[j])
            V[i, j] = V[j, i] = nuclear_attraction(basis[i], basis[j], molecule)
    return S, T, V


def build_eri(basis, verbose=False):
    """Full (uv|ls) tensor, exploiting its 8-fold permutation symmetry:

        (uv|ls) = (vu|ls) = (uv|sl) = (vu|sl)
                = (ls|uv) = (sl|uv) = (ls|vu) = (sl|vu)
    """
    n = len(basis)
    eri = np.zeros((n, n, n, n))
    n_unique = 0
    for i in range(n):
        for j in range(i + 1):
            ij = i * (i + 1) // 2 + j    # compound index of the pair (i,j)
            for k in range(n):
                for l in range(k + 1):
                    kl = k * (k + 1) // 2 + l
                    if ij < kl:
                        continue         # will be filled by symmetry
                    val = electron_repulsion(basis[i], basis[j],
                                             basis[k], basis[l])
                    n_unique += 1
                    for a, b, c, d in ((i, j, k, l), (j, i, k, l),
                                       (i, j, l, k), (j, i, l, k),
                                       (k, l, i, j), (l, k, i, j),
                                       (k, l, j, i), (l, k, j, i)):
                        eri[a, b, c, d] = val
    if verbose:
        print(f"  computed {n_unique} unique ERIs "
              f"(instead of {n ** 4} without symmetry)")
    return eri
