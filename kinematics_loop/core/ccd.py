"""Cyclic Coordinate Descent (CCD) loop closure.

CCD treats the loop as a robot arm: the fixed N-anchor is the base, the loop's
C-terminal atoms are the *end-effector*, and the known C-anchor is the target the
end-effector must reach. One backbone torsion is adjusted at a time, each set to
the angle that *analytically* minimizes the distance between the moving end and
its target. Sweeping over all torsions repeatedly drives that distance to zero.

Reference: Canutescu & Dunbrack, "Cyclic coordinate descent: A robotics algorithm
for protein loop closure", Protein Science 2003.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo
from .backbone import LoopBackbone
from .energy import backbone_energy

# Coarse Ramachandran basin centres (phi, psi in degrees) for reporting.
_RAMA_BASINS = [(-63.0, -43.0), (-120.0, 130.0), (-65.0, 145.0), (60.0, 45.0)]
_RAMA_RADIUS = 50.0


# ----------------------------------------------------------------------
# The analytic CCD step
# ----------------------------------------------------------------------
def optimal_angle(
    moving: np.ndarray, targets: np.ndarray, origin: np.ndarray, axis_unit: np.ndarray
) -> float:
    """Angle that minimizes sum |R(theta)(M_j) - F_j|^2 about a fixed axis.

    Rotating the moving atoms only changes the term ``b*cos(theta) + c*sin(theta)``
    of the objective, so the optimum is ``atan2(c, b)`` (a closed-form solution,
    no line search).
    """
    b = 0.0
    c = 0.0
    for m, f in zip(moving, targets):
        r = m - origin
        r_perp = r - np.dot(r, axis_unit) * axis_unit
        s = np.cross(axis_unit, r)          # = axis x r_perp, the "sin" direction
        t = f - origin
        b += float(np.dot(r_perp, t))
        c += float(np.dot(s, t))
    return float(np.arctan2(c, b))


@dataclass
class ClosureResult:
    converged: bool
    iterations: int
    rmsd: float
    history: list[float] = field(default_factory=list)


def close_loop(
    backbone: LoopBackbone,
    targets: np.ndarray,
    max_iter: int = 5000,
    tol: float = 0.08,
) -> ClosureResult:
    """Run CCD in place until the end-effector RMSD to ``targets`` < ``tol``."""
    axes = backbone.rotatable_axes()
    history: list[float] = []
    for it in range(1, max_iter + 1):
        for kind, i in axes:
            a, b = backbone.axis_atoms(kind, i)
            origin = backbone.coords[b]
            axis_unit = geo.normalize(backbone.coords[b] - backbone.coords[a])
            theta = optimal_angle(backbone.end_effector(), targets, origin, axis_unit)
            backbone.apply_rotation(kind, i, theta)
        r = geo.rmsd(backbone.end_effector(), targets)
        history.append(r)
        if r < tol:
            return ClosureResult(True, it, r, history)
    return ClosureResult(False, max_iter, history[-1] if history else np.inf, history)


# ----------------------------------------------------------------------
# Conformation quality (coarse, for ranking)
# ----------------------------------------------------------------------
def _ang_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def rama_violations(phi: np.ndarray, psi: np.ndarray) -> int:
    """Count residues whose (phi, psi) falls outside all favored basins (coarse)."""
    bad = 0
    for f, p in zip(phi, psi):
        near = any(
            np.hypot(_ang_dist(f, cf), _ang_dist(p, cp)) < _RAMA_RADIUS
            for cf, cp in _RAMA_BASINS
        )
        if not near:
            bad += 1
    return bad


def backbone_clashes(backbone: LoopBackbone, cutoff: float = 2.0, gap: int = 2) -> int:
    """Count non-adjacent backbone atom pairs closer than ``cutoff`` Angstrom."""
    xyz = backbone.coords
    n = len(xyz)
    count = 0
    for i in range(n):
        for j in range(i + gap + 1, n):
            if np.linalg.norm(xyz[i] - xyz[j]) < cutoff:
                count += 1
    return count


# ----------------------------------------------------------------------
# High-level problem + multi-start solver
# ----------------------------------------------------------------------
@dataclass
class Solution:
    backbone: LoopBackbone
    phi: np.ndarray            # degrees
    psi: np.ndarray            # degrees
    rmsd: float
    iterations: int
    clashes: int
    rama_bad: int
    energy: float              # coarse backbone ranking energy (lower is better)


@dataclass
class LoopProblem:
    """Close a loop of a given sequence between two fixed backbone anchors."""

    sequence: str
    seed: np.ndarray           # N-anchor base atoms (C0, N1, CA1)
    targets: np.ndarray        # C-anchor target atoms (N_L, CA_L, C_L)

    @classmethod
    def from_reference(
        cls, sequence: str, phi_deg, psi_deg, seed: np.ndarray | None = None
    ) -> tuple["LoopProblem", LoopBackbone]:
        """Build a self-consistent test case from known torsions.

        Returns the problem (anchors taken from the reference ends) and the
        reference backbone, so a solver can be validated against ground truth.
        In a real task the anchors instead come from the residues flanking the
        loop in a known structure.
        """
        ref = LoopBackbone.from_torsions(
            sequence, np.radians(phi_deg), np.radians(psi_deg), seed
        )
        return cls(sequence, ref.seed.copy(), ref.end_effector().copy()), ref

    def solve(
        self,
        n_solutions: int = 5,
        max_tries: int = 200,
        tol: float = 0.08,
        w_rama: float = 1.0,
        candidate_pool: int | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> list[Solution]:
        """Find good closed conformations via random-restart CCD ranked by energy.

        Loop closure is under-determined (many conformations fit the same ends),
        so we restart CCD from random torsions and collect a *pool* of closures it
        manages to complete, then keep the ``n_solutions`` with the lowest coarse
        backbone **energy** (van der Waals + a smooth Ramachandran penalty; see
        ``energy.backbone_energy``). CCD only makes the ends meet -- the energy is
        what decides which closure is *good*.

        ``candidate_pool`` sets how many converged closures to score before
        ranking (default ``max(5*n_solutions, 25)``); a larger pool searches more
        of the loop's conformational freedom at the cost of more CCD runs.
        """
        rng = random.Random(seed)
        L = len(self.sequence)
        pool = candidate_pool or max(5 * n_solutions, 25)
        solutions: list[Solution] = []
        for _ in range(max_tries):
            if len(solutions) >= pool:
                break
            phi0 = np.array([rng.uniform(-180, 180) for _ in range(L)])
            psi0 = np.array([rng.uniform(-180, 180) for _ in range(L)])
            bb = LoopBackbone.from_torsions(
                self.sequence, np.radians(phi0), np.radians(psi0), self.seed
            )
            res = close_loop(bb, self.targets, tol=tol)
            if not res.converged:
                continue
            phi, psi = bb.torsions()
            sol = Solution(
                backbone=bb, phi=phi, psi=psi, rmsd=res.rmsd,
                iterations=res.iterations,
                clashes=backbone_clashes(bb),
                rama_bad=rama_violations(phi, psi),
                energy=backbone_energy(bb, w_rama=w_rama),
            )
            solutions.append(sol)
            if verbose:
                print(f"    candidate: energy={sol.energy:8.2f} rmsd={sol.rmsd:.3f} "
                      f"iters={sol.iterations} clashes={sol.clashes} "
                      f"rama_bad={sol.rama_bad}")
        solutions.sort(key=lambda s: s.energy)
        return solutions[:n_solutions]
