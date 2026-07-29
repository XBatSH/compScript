"""kinematics_loop: protein loop closure as robot inverse kinematics.

Public API:
    LoopBackbone      - the N-CA-C serial chain (forward kinematics + PDB export)
    LoopProblem       - close a loop of a sequence between two fixed anchors
    Solution          - a single closed conformation
    close_loop        - the CCD closure routine (operates on a LoopBackbone)
    optimal_angle     - the analytic per-torsion CCD step
"""

from .geometry import place_atom, rotate_points, dihedral, rmsd, normalize
from .backbone import LoopBackbone
from .energy import backbone_energy, vdw_energy, rama_energy
from .ccd import (
    optimal_angle,
    close_loop,
    ClosureResult,
    LoopProblem,
    Solution,
    backbone_clashes,
    rama_violations,
)

__all__ = [
    "place_atom",
    "rotate_points",
    "dihedral",
    "rmsd",
    "normalize",
    "LoopBackbone",
    "backbone_energy",
    "vdw_energy",
    "rama_energy",
    "optimal_angle",
    "close_loop",
    "ClosureResult",
    "LoopProblem",
    "Solution",
    "backbone_clashes",
    "rama_violations",
]
