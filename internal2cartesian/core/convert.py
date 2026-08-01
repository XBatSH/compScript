"""
convert.py -- Cartesian <-> internal (Z-matrix) coordinate conversion.

Core operations
---------------
* ``place_atom(a, b, c, bond, angle, torsion)``
    Forward kinematics (NeRF): place atom D from three predecessors A-B-C.

* ``internal_to_cartesian(entries)``
    Build Cartesian coordinates from a Z-matrix (list of ZMatrixEntry).

* ``cartesian_to_internal(atoms)``
    Extract the Z-matrix representation from Cartesian coordinates.

Peptide-specific helpers
------------------------
* ``peptide_ideal_geometry()``
    Return the standard bond lengths and angles for a peptide backbone.

* ``build_peptide_from_internal(sequence, phi_deg, psi_deg, omega_deg)``
    Build a peptide backbone from ideal geometry plus user-specified torsions.

* ``extract_backbone_internal(coords)``
    Given Cartesian coordinates of a backbone, extract all bond lengths,
    bond angles, and dihedral (phi / psi / omega) angles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
#  Low-level geometry
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector along *v*."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle (radians) between two vectors *u* and *v*."""
    dot = np.dot(normalize(u), normalize(v))
    return float(np.arccos(np.clip(dot, -1.0, 1.0)))


def bond_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle A-B-C (radians), i.e. the supplement of angle(b-a, c-b)."""
    return angle_between(a - b, c - b)


def dihedral(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
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


def place_atom(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond: float,
    angle: float,
    torsion: float,
) -> np.ndarray:
    """Forward kinematics (NeRF): place atom D from three predecessors A-B-C.

    The new atom D satisfies:
        |C-D| = *bond*
        angle(B, C, D) = *angle*
        dihedral(A, B, C, D) = *torsion*

    All angles are in radians.
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


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class ZMatrixEntry:
    """One line of a Z-matrix.

    Atom *i* (1-based) is the current atom being placed.  It is bonded to
    atom *bond_to*, it makes an angle with atom *angle_with*, and a dihedral
    with atom *dihedral_with*.  For the first three atoms some fields are
    ``None``.

    All geometric values are stored in **radians** (angles) and **Angstrom**
    (lengths) for internal consistency.
    """

    symbol: str
    index: int                     # 1-based index of THIS atom
    bond_to: int | None = None     # 1-based index
    bond_length: float | None = None
    angle_with: int | None = None  # 1-based index
    angle: float | None = None
    dihedral_with: int | None = None  # 1-based index
    dihedral: float | None = None


@dataclass
class InternalCoords:
    """Full internal coordinates of a molecule.

    Bonds, angles, and dihedrals are stored as (i, j, value) tuples where
    i, j are 0-based atom indices.
    """

    bonds: list[tuple[int, int, float]] = field(default_factory=list)
    angles: list[tuple[int, int, int, float]] = field(default_factory=list)
    dihedrals: list[tuple[int, int, int, int, float]] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
#  Z-matrix <-> Cartesian conversion
# ---------------------------------------------------------------------------

def internal_to_cartesian(entries: list[ZMatrixEntry]) -> np.ndarray:
    """Build Cartesian coordinates from a list of Z-matrix entries.

    Returns an (n, 3) array in Angstroms.  The Z-matrix entries must be
    given in order (atom 1 first, atom n last) and every ``bond_to`` /
    ``angle_with`` / ``dihedral_with`` must refer to atoms with smaller
    indices.
    """
    n = len(entries)
    coords = np.zeros((n, 3))

    # Atom 1: at the origin.
    e = entries[0]
    coords[0] = np.array([0.0, 0.0, 0.0])

    if n == 1:
        return coords

    # Atom 2: placed along the +z axis at bond distance from atom 1.
    e = entries[1]
    coords[1] = np.array([0.0, 0.0, e.bond_length])

    if n == 2:
        return coords

    # Atom 3: placed in the xz-plane.
    e = entries[2]
    bond = e.bond_length
    ang = e.angle  # angle(atom1, atom2, atom3)
    # atom2 is at (0,0,bond_prev); atom1 at origin.
    b_prev = entries[1].bond_length  # distance between atoms 1 and 2
    coords[2] = np.array([
        bond * np.sin(ang),
        0.0,
        b_prev - bond * np.cos(ang),
    ])

    # Atoms 4+: use the NeRF algorithm.
    for k in range(3, n):
        e = entries[k]
        a = coords[e.dihedral_with - 1]
        b = coords[e.angle_with - 1]
        c = coords[e.bond_to - 1]
        coords[k] = place_atom(a, b, c, e.bond_length, e.angle, e.dihedral)

    return coords


def cartesian_to_internal(
    atoms: list[tuple[str, float, float, float]],
    bonds: list[tuple[int, int]] | None = None,
) -> list[ZMatrixEntry]:
    """Extract the Z-matrix from Cartesian coordinates.

    *atoms* is a list of (symbol, x, y, z), one per atom, in the order
    they appear in the Z-matrix.

    If *bonds* is provided (list of 0-based (i,j) pairs), each atom
    selects its bond partner from the preceding atom connected by a
    bond.  Otherwise the nearest preceding atom is used (which can give
    incorrect results for compact structures like helices).
    """
    n = len(atoms)
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    entries: list[ZMatrixEntry] = []

    # Build adjacency for fast lookup of bonded preceding atoms.
    bonded_preceding: list[list[int]] = [[] for _ in range(n)]
    if bonds is not None:
        for i, j in bonds:
            if j < i:
                bonded_preceding[i].append(j)
            elif i < j:
                bonded_preceding[j].append(i)
        # Sort by distance within each list so the nearest bonded atom comes first.
        for k in range(n):
            if bonded_preceding[k]:
                bonded_preceding[k].sort(
                    key=lambda idx: float(np.linalg.norm(coords[k] - coords[idx]))
                )

    for i in range(n):
        sym = atoms[i][0]
        if i == 0:
            entries.append(ZMatrixEntry(sym, 1))
        elif i == 1:
            d = float(np.linalg.norm(coords[1] - coords[0]))
            entries.append(ZMatrixEntry(sym, 2, bond_to=1, bond_length=d))
        elif i == 2:
            d = float(np.linalg.norm(coords[2] - coords[1]))
            ang = bond_angle(coords[0], coords[1], coords[2])
            entries.append(
                ZMatrixEntry(sym, 3, bond_to=2, bond_length=d,
                             angle_with=1, angle=ang)
            )
        else:
            # Pick reference atoms: prefer bonded atoms, fall back to nearest.
            b_idx, a_idx, d_idx = _pick_references(
                coords, i, bonded_preceding[i]
            )
            bond = float(np.linalg.norm(coords[i] - coords[b_idx - 1]))
            ang = bond_angle(coords[a_idx - 1], coords[b_idx - 1], coords[i])
            dih = dihedral(
                coords[d_idx - 1], coords[a_idx - 1],
                coords[b_idx - 1], coords[i],
            )
            # Negate the dihedral: the NeRF convention used by place_atom
            # and build_peptide_from_internal is the opposite of the IUPAC
            # convention returned by dihedral().  Storing the negated value
            # ensures the round-trip (Z-matrix -> Cartesian) is exact.
            entries.append(
                ZMatrixEntry(sym, i + 1,
                             bond_to=b_idx, bond_length=bond,
                             angle_with=a_idx, angle=ang,
                             dihedral_with=d_idx, dihedral=-dih)
            )

    return entries


def _pick_references(
    coords: np.ndarray,
    i: int,
    bonded: list[int],
) -> tuple[int, int, int]:
    """Pick (bond_to, angle_with, dihedral_with) as 1-based indices.

    Prefers bonded atoms when available; fills remaining slots from the
    nearest preceding atoms not already chosen.
    """
    chosen: list[int] = []
    used: set[int] = set()

    # First: prefer bonded atoms (closest first, already sorted).
    for b in bonded:
        if b not in used:
            chosen.append(b)
            used.add(b)
        if len(chosen) >= 3:
            break

    # Fill remaining slots from nearest preceding atoms.
    if len(chosen) < 3:
        dists = [(float(np.linalg.norm(coords[i] - coords[j])), j)
                 for j in range(i) if j not in used]
        dists.sort()
        for _, j in dists:
            chosen.append(j)
            if len(chosen) >= 3:
                break

    return chosen[0] + 1, chosen[1] + 1, chosen[2] + 1


# ---------------------------------------------------------------------------
#  Peptide backbone helpers
# ---------------------------------------------------------------------------

# Standard peptide backbone geometry (Cremer & Pople, 1975; Engh & Huber, 1991).
# All values in Angstroms and radians.

_PEPTIDE_IDEAL = {
    # bond lengths
    "N_CA": 1.458,     # N - CA
    "CA_C": 1.525,     # CA - C
    "C_N": 1.330,      # C - N  (peptide bond, partial double-bond character)
    # bond angles
    "N_CA_C": np.deg2rad(111.0),     # N-CA-C
    "CA_C_N": np.deg2rad(116.0),     # CA-C-N
    "C_N_CA": np.deg2rad(122.0),     # C-N-CA
}


def peptide_ideal_geometry() -> dict:
    """Return the ideal peptide backbone geometry.

    Returns a dict with bond lengths (Angstrom) and bond angles (radians).

    >>> g = peptide_ideal_geometry()
    >>> round(g["N_CA"], 2)
    1.46
    >>> round(np.rad2deg(g["N_CA_C"]), 1)
    111.0
    """
    return dict(_PEPTIDE_IDEAL)


def build_peptide_from_internal(
    sequence: str,
    phi_deg: list[float] | None = None,
    psi_deg: list[float] | None = None,
    omega_deg: float = 180.0,
) -> tuple[np.ndarray, list[str]]:
    """Build a peptide backbone from ideal geometry and user-specified torsions.

    Parameters
    ----------
    sequence : str
        One-letter amino-acid codes, e.g. ``"AAAA"``.
    phi_deg, psi_deg : list[float] | None
        Backbone torsions in **degrees**, one per residue.  If ``None``,
        defaults to an extended conformation (-135, +135).
    omega_deg : float
        Peptide-bond dihedral (degrees).  180 = trans (default).

    Returns
    -------
    coords : np.ndarray (3*n_res, 3)
        Cartesian coordinates in Angstroms.
    atom_names : list[str]
        Labels for each row of *coords*, e.g. ``["N1","CA1","C1","N2",...]``.
    """
    geo = peptide_ideal_geometry()
    L = len(sequence)

    if phi_deg is None:
        phi_deg = [-135.0] * L
    if psi_deg is None:
        psi_deg = [135.0] * L

    phi = np.deg2rad(phi_deg)
    psi = np.deg2rad(psi_deg)
    omega = np.deg2rad(omega_deg)

    n_atoms = 3 * L
    coords = np.zeros((n_atoms, 3))
    names: list[str] = []

    # ------ Residue 1 ------
    # Place the first three atoms "by hand".
    # N1 at origin, CA1 along +z, C1 in the xz-plane.
    coords[0] = np.array([0.0, 0.0, 0.0])
    names.append("N1")

    coords[1] = np.array([0.0, 0.0, geo["N_CA"]])
    names.append("CA1")

    r = geo["CA_C"]
    ang = np.pi - geo["N_CA_C"]  # angle(N1, CA1, C1) is the supplement
    b12 = geo["N_CA"]
    coords[2] = np.array([
        r * np.sin(ang),
        0.0,
        b12 + r * np.cos(ang),
    ])
    names.append("C1")

    # ------ Residues 2 .. L ------
    for i in range(1, L):
        idx_prev_C = 3 * i - 1         # C of previous residue
        idx_prev_CA = 3 * i - 2        # CA of previous residue
        idx_prev_N = 3 * i - 3         # N of previous residue

        idx_N = 3 * i
        idx_CA = 3 * i + 1
        idx_C = 3 * i + 2

        # --- Place N_i ---
        # Reference: N_{i-1} - CA_{i-1} - C_{i-1} - N_i = psi_{i-1}
        a_N = coords[idx_prev_N]
        b_N = coords[idx_prev_CA]
        c_N = coords[idx_prev_C]
        coords[idx_N] = place_atom(
            a_N, b_N, c_N,
            geo["C_N"],           # C-N bond
            geo["CA_C_N"],        # angle CA-C-N
            -psi[i - 1],          # psi of previous residue (negated for IUPAC->NeRF convention)
        )
        names.append(f"N{i + 1}")

        # --- Place CA_i ---
        # Reference: CA_{i-1} - C_{i-1} - N_i - CA_i = omega (trans peptide bond)
        a_CA = coords[idx_prev_CA]
        b_CA = coords[idx_prev_C]
        c_CA = coords[idx_N]
        coords[idx_CA] = place_atom(
            a_CA, b_CA, c_CA,
            geo["N_CA"],           # N-CA bond
            geo["C_N_CA"],         # angle C-N-CA
            omega,                 # trans peptide bond dihedral
        )
        names.append(f"CA{i + 1}")

        # --- Place C_i ---
        # Reference: N_i - CA_i - C_i
        # Dihedral: phi_i (C_{i-1} - N_i - CA_i - C_i)
        a_C = coords[idx_prev_C]   # C_{i-1}
        b_C = coords[idx_N]        # N_i
        c_C = coords[idx_CA]       # CA_i
        coords[idx_C] = place_atom(
            a_C, b_C, c_C,
            geo["CA_C"],           # CA-C bond
            geo["N_CA_C"],         # angle N-CA-C
            -phi[i - 1],           # phi of THIS residue (negated for IUPAC->NeRF convention)
        )
        names.append(f"C{i + 1}")

    return coords, names


def extract_backbone_internal(
    coords: np.ndarray,
) -> InternalCoords:
    """Extract all backbone internal coordinates from Cartesian positions.

    Assumes the standard backbone atom order: N1, CA1, C1, N2, CA2, C2, ...

    Returns
    -------
    InternalCoords
        Bonds, angles, and dihedrals with residue labels.
    """
    n = coords.shape[0]
    assert n % 3 == 0, f"expected 3N backbone atoms, got {n}"
    L = n // 3
    result = InternalCoords()

    # --- Bond lengths ---
    for i in range(L):
        ni, cai, ci = 3 * i, 3 * i + 1, 3 * i + 2
        result.bonds.append((ni, cai,
                             float(np.linalg.norm(coords[cai] - coords[ni]))))
        result.bonds.append((cai, ci,
                             float(np.linalg.norm(coords[ci] - coords[cai]))))
        if i < L - 1:
            result.bonds.append((ci, 3 * (i + 1),
                                 float(np.linalg.norm(coords[3 * (i + 1)] - coords[ci]))))

    # --- Bond angles ---
    for i in range(L):
        ni, cai, ci = 3 * i, 3 * i + 1, 3 * i + 2
        result.angles.append(
            (ni, cai, ci, bond_angle(coords[ni], coords[cai], coords[ci])))
        if i < L - 1:
            nnext = 3 * (i + 1)
            result.angles.append(
                (cai, ci, nnext, bond_angle(coords[cai], coords[ci], coords[nnext])))
        if i > 0:
            cprev = 3 * i - 1
            result.angles.append(
                (cprev, ni, cai, bond_angle(coords[cprev], coords[ni], coords[cai])))

    # --- Dihedrals (phi, psi, omega) ---
    for i in range(L):
        ni, cai, ci = 3 * i, 3 * i + 1, 3 * i + 2
        # phi_i: C_{i-1} - N_i - CA_i - C_i
        if i > 0:
            cprev = 3 * i - 1
            d = dihedral(coords[cprev], coords[ni], coords[cai], coords[ci])
            result.dihedrals.append((cprev, ni, cai, ci, d))
        # psi_i: N_i - CA_i - C_i - N_{i+1}
        if i < L - 1:
            nnext = 3 * (i + 1)
            d = dihedral(coords[ni], coords[cai], coords[ci], coords[nnext])
            result.dihedrals.append((ni, cai, ci, nnext, d))
        # omega_i: CA_{i-1} - C_{i-1} - N_i - CA_i
        if i > 0:
            caprev = 3 * i - 2
            cprev = 3 * i - 1
            d = dihedral(coords[caprev], coords[cprev], coords[ni], coords[cai])
            result.dihedrals.append((caprev, cprev, ni, cai, d))

    return result
