"""A coarse backbone energy for ranking closed loop conformations.

Loop closure is under-determined: many conformations satisfy the same two
anchors. CCD only makes the ends *meet* -- it says nothing about whether the
resulting fold is physically sensible. To pick the *good* closures we score each
one with a simple, physically-motivated energy and prefer the lowest:

    E = E_vdw + w_rama * E_rama

* E_vdw   - Lennard-Jones (12-6) van der Waals energy summed over non-adjacent
            backbone atom pairs. Steric clashes cost a large positive energy;
            comfortable packing is mildly negative.
* E_rama  - a smooth Ramachandran pseudo-energy: each residue pays a penalty that
            grows with its angular distance from the nearest favored (phi, psi)
            basin, so backbones sitting in allowed regions score lower.

Because we model only the N-CA-C trace the energy is deliberately coarse; it is
meant for *ranking* closures, not for reporting absolute stabilities.
"""

from __future__ import annotations

import numpy as np

# Bondi van der Waals radii (Angstrom) and rough LJ well depths (kcal/mol).
_VDW = {"N": (1.55, 0.16), "C": (1.70, 0.11)}
# Coarse Ramachandran basin centres (phi, psi in degrees).
_RAMA_BASINS = np.array([
    [-63.0, -43.0],    # right-handed alpha helix
    [-120.0, 130.0],   # beta sheet
    [-65.0, 145.0],    # polyproline II / extended
    [60.0, 45.0],      # left-handed alpha (Gly)
])
_RAMA_SIGMA = 35.0     # degrees; width of the favored wells


def _elements(n_atoms: int) -> list[str]:
    """Element per atom for the C0, N1, CA1, C1, N2, ... trace.

    Index 0 is the fixed preceding carbonyl carbon; from index 1 the pattern
    repeats N, CA, C (CA and C are both carbon).
    """
    elems = ["C"]  # C0
    for idx in range(1, n_atoms):
        elems.append("N" if (idx - 1) % 3 == 0 else "C")
    return elems


def vdw_energy(coords: np.ndarray, gap: int = 2) -> float:
    """Lennard-Jones (12-6) energy over backbone atom pairs at least ``gap``+1
    apart in the chain (excludes 1-2 and 1-3 bonded neighbours by default)."""
    elems = _elements(len(coords))
    radii = np.array([_VDW[e][0] for e in elems])
    eps = np.array([_VDW[e][1] for e in elems])
    n = len(coords)
    total = 0.0
    for i in range(n):
        for j in range(i + gap + 1, n):
            r = float(np.linalg.norm(coords[i] - coords[j]))
            rmin = radii[i] + radii[j]
            epsij = float(np.sqrt(eps[i] * eps[j]))
            # Cap the separation so a hard overlap yields a large but finite cost.
            ratio = rmin / max(r, 0.5 * rmin)
            total += epsij * (ratio**12 - 2.0 * ratio**6)
    return total


def _ang_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def rama_energy(phi: np.ndarray, psi: np.ndarray) -> float:
    """Smooth Ramachandran pseudo-energy (degrees in, arbitrary units out).

    Each residue is scored by its squared angular distance to the nearest favored
    basin, normalized by the basin width, so residues deep inside an allowed
    region contribute ~0 and outliers contribute a growing positive penalty.
    """
    total = 0.0
    for f, p in zip(phi, psi):
        best = min(
            _ang_dist(f, cf) ** 2 + _ang_dist(p, cp) ** 2
            for cf, cp in _RAMA_BASINS
        )
        total += best / (2.0 * _RAMA_SIGMA**2)
    return total


def backbone_energy(backbone, w_rama: float = 1.0) -> float:
    """Total ranking energy: van der Waals plus weighted Ramachandran penalty."""
    phi, psi = backbone.torsions()
    return vdw_energy(backbone.coords) + w_rama * rama_energy(phi, psi)
