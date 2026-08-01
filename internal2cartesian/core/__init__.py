"""
internal2cartesian -- Convert between Cartesian and internal (Z-matrix)
coordinate systems.

Uses a peptide backbone as the model system.  The core operation is the
Natural Extension Reference Frame (NeRF) algorithm: place a fourth atom from
three predecessors given a bond length, bond angle, and dihedral angle.

For a peptide, the natural internal coordinates are the backbone dihedrals
(phi, psi, omega) together with the nearly-constant bond lengths and angles
that make the backbone a serial kinematic chain -- the same machinery that
the `kinematics_loop` module uses for loop closure.
"""

from .convert import (
    ZMatrixEntry,
    InternalCoords,
    cartesian_to_internal,
    internal_to_cartesian,
    extract_backbone_internal,
    build_peptide_from_internal,
    peptide_ideal_geometry,
    angle_between,
    bond_angle,
    dihedral,
    normalize,
    place_atom,
)

__all__ = [
    "ZMatrixEntry",
    "InternalCoords",
    "cartesian_to_internal",
    "internal_to_cartesian",
    "extract_backbone_internal",
    "build_peptide_from_internal",
    "peptide_ideal_geometry",
    "angle_between",
    "bond_angle",
    "dihedral",
    "normalize",
    "place_atom",
]
