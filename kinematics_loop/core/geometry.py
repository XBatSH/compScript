"""Low-level 3D geometry for the loop-closure kinematics.

A polypeptide backbone is a **serial kinematic chain**: each atom is placed
relative to the previous three by a bond length, a bond angle, and a torsion.
Building the chain is *forward kinematics* (`place_atom`); closing the loop is
*inverse kinematics*, which needs rotations about bond axes (`rotate_points`) and
a way to read torsions back out (`dihedral`).
"""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector along ``v``."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def place_atom(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond: float,
    angle: float,
    torsion: float,
) -> np.ndarray:
    """Forward kinematics: place atom D from three predecessors A-B-C.

    The new atom D satisfies |C-D| = ``bond``, angle(B, C, D) = ``angle`` and the
    dihedral A-B-C-D = ``torsion`` (angles in radians). This is the Natural
    Extension Reference Frame (NeRF) construction.
    """
    bc = normalize(c - b)
    n = normalize(np.cross(b - a, bc))
    nbc = np.cross(n, bc)
    # Column frame [bc, n x bc, n] maps the local offset into world space.
    m = np.stack([bc, nbc, n], axis=1)
    d_local = np.array([
        -bond * np.cos(angle),
        bond * np.sin(angle) * np.cos(torsion),
        bond * np.sin(angle) * np.sin(torsion),
    ])
    return c + m @ d_local


def rotate_points(
    points: np.ndarray, origin: np.ndarray, axis: np.ndarray, theta: float
) -> np.ndarray:
    """Rotate ``points`` by ``theta`` (rad) about the line (origin, unit axis).

    Uses Rodrigues' rotation formula; ``points`` is an (n, 3) array.
    """
    k = normalize(axis)
    v = points - origin
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotated = (
        v * cos_t
        + np.cross(k, v) * sin_t
        + np.outer(v @ k, k) * (1.0 - cos_t)
    )
    return origin + rotated


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed dihedral angle (radians) of the four points p0-p1-p2-p3."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m = np.cross(n1, normalize(b2))
    x = np.dot(n1, n2)
    y = np.dot(m, n2)
    return float(np.arctan2(y, x))


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square distance between two equal-length point sets."""
    diff = a - b
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", diff, diff))))
