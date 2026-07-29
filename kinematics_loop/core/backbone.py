"""The backbone as a serial kinematic chain of N-CA-C atoms.

We model only the main-chain trace (N, CA, C per residue) plus a fixed carbonyl
`C0` in front of the loop so that phi of the first residue is well defined. Atoms
are stored in build order:

    C0, N1, CA1, C1, N2, CA2, C2, ..., N_L, CA_L, C_L

Index formulas (residue i is 1-based):  N_i = 3i-2,  CA_i = 3i-1,  C_i = 3i.
Rotating the phi torsion turns everything downstream of the N-CA bond; rotating
psi turns everything downstream of the CA-C bond. That "downstream block" is what
the CCD solver rotates to drive the loop end onto its target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo

# Ideal peptide backbone geometry (Engh & Huber). Bond lengths in Angstrom.
BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329
# Bond angles in radians.
ANGLE_N_CA_C = np.radians(111.2)
ANGLE_CA_C_N = np.radians(116.2)
ANGLE_C_N_CA = np.radians(121.7)
OMEGA = np.radians(180.0)  # planar trans peptide bond

# Minimal one-letter -> three-letter map for PDB export / labelling.
AA1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _default_seed() -> np.ndarray:
    """Canonical fixed base atoms (C0, N1, CA1) for the loop's N-anchor."""
    return np.array([
        [-1.459, 0.000, 0.000],   # C0 (preceding carbonyl, fixed)
        [0.000, 0.000, 0.000],    # N1
        [0.000, 1.458, 0.000],    # CA1
    ])


@dataclass
class LoopBackbone:
    """A main-chain N-CA-C trace built from per-residue phi/psi torsions."""

    sequence: str
    coords: np.ndarray                    # (3L+1, 3) atom coordinates
    seed: np.ndarray = field(repr=False)  # fixed base atoms C0, N1, CA1

    @property
    def n_res(self) -> int:
        return len(self.sequence)

    # -- index helpers (1-based residue) --------------------------------
    @staticmethod
    def idx_N(i: int) -> int:
        return 3 * i - 2

    @staticmethod
    def idx_CA(i: int) -> int:
        return 3 * i - 1

    @staticmethod
    def idx_C(i: int) -> int:
        return 3 * i

    # -- construction ---------------------------------------------------
    @classmethod
    def from_torsions(
        cls,
        sequence: str,
        phi: np.ndarray,
        psi: np.ndarray,
        seed: np.ndarray | None = None,
    ) -> "LoopBackbone":
        """Forward kinematics: build the trace from phi/psi arrays (radians)."""
        seed = _default_seed() if seed is None else np.asarray(seed, float)
        L = len(sequence)
        coords = np.zeros((3 * L + 1, 3))
        coords[0:3] = seed  # C0, N1, CA1

        # C1 from (C0, N1, CA1) using phi_1.
        coords[cls.idx_C(1)] = geo.place_atom(
            coords[0], coords[1], coords[2], BOND_CA_C, ANGLE_N_CA_C, phi[0]
        )
        for i in range(1, L):
            n_i, ca_i, c_i = coords[cls.idx_N(i)], coords[cls.idx_CA(i)], coords[cls.idx_C(i)]
            # N_{i+1} from (N_i, CA_i, C_i) via psi_i
            n_next = geo.place_atom(n_i, ca_i, c_i, BOND_C_N, ANGLE_CA_C_N, psi[i - 1])
            # CA_{i+1} from (CA_i, C_i, N_{i+1}) via omega
            ca_next = geo.place_atom(ca_i, c_i, n_next, BOND_N_CA, ANGLE_C_N_CA, OMEGA)
            # C_{i+1} from (C_i, N_{i+1}, CA_{i+1}) via phi_{i+1}
            c_next = geo.place_atom(c_i, n_next, ca_next, BOND_CA_C, ANGLE_N_CA_C, phi[i])
            coords[cls.idx_N(i + 1)] = n_next
            coords[cls.idx_CA(i + 1)] = ca_next
            coords[cls.idx_C(i + 1)] = c_next
        return cls(sequence=sequence, coords=coords, seed=seed)

    def copy(self) -> "LoopBackbone":
        return LoopBackbone(self.sequence, self.coords.copy(), self.seed)

    # -- kinematics -----------------------------------------------------
    def rotatable_axes(self) -> list[tuple[str, int]]:
        """List of adjustable torsions as ("phi"|"psi", residue). psi_L is omitted
        because nothing downstream of the last CA-C bond is part of the trace."""
        axes: list[tuple[str, int]] = []
        for i in range(1, self.n_res + 1):
            axes.append(("phi", i))
            if i < self.n_res:
                axes.append(("psi", i))
        return axes

    def axis_atoms(self, kind: str, i: int) -> tuple[int, int]:
        """Return (index_a, index_b) of the bond the torsion rotates about."""
        if kind == "phi":
            return self.idx_N(i), self.idx_CA(i)
        return self.idx_CA(i), self.idx_C(i)

    def downstream_slice(self, kind: str, i: int) -> slice:
        """Atoms that move when this torsion is rotated (everything after axis b)."""
        _, b = self.axis_atoms(kind, i)
        return slice(b + 1, len(self.coords))

    def apply_rotation(self, kind: str, i: int, theta: float) -> None:
        """Rotate the downstream block by ``theta`` about the given bond axis."""
        a, b = self.axis_atoms(kind, i)
        axis = self.coords[b] - self.coords[a]
        sl = self.downstream_slice(kind, i)
        self.coords[sl] = geo.rotate_points(self.coords[sl], self.coords[b], axis, theta)

    def end_effector(self) -> np.ndarray:
        """The last three atoms (N_L, CA_L, C_L) that must reach the C-anchor."""
        return self.coords[-3:]

    # -- reporting ------------------------------------------------------
    def torsions(self) -> tuple[np.ndarray, np.ndarray]:
        """Recover (phi, psi) in degrees from the current coordinates."""
        phi = np.zeros(self.n_res)
        psi = np.zeros(self.n_res)
        for i in range(1, self.n_res + 1):
            phi[i - 1] = np.degrees(geo.dihedral(
                self.coords[self.idx_C(i) - 3] if i > 1 else self.coords[0],
                self.coords[self.idx_N(i)],
                self.coords[self.idx_CA(i)],
                self.coords[self.idx_C(i)],
            ))
            if i < self.n_res:
                psi[i - 1] = np.degrees(geo.dihedral(
                    self.coords[self.idx_N(i)],
                    self.coords[self.idx_CA(i)],
                    self.coords[self.idx_C(i)],
                    self.coords[self.idx_N(i + 1)],
                ))
        return phi, psi

    def to_pdb_block(self) -> str:
        """Minimal PDB (N, CA, C trace) for viewing in PyMOL/VMD."""
        lines = []
        serial = 1
        names = ("N", "CA", "C")
        for i in range(1, self.n_res + 1):
            resn = AA1TO3.get(self.sequence[i - 1].upper(), "ALA")
            for name, idx in zip(names, (self.idx_N(i), self.idx_CA(i), self.idx_C(i))):
                x, y, z = self.coords[idx]
                lines.append(
                    f"ATOM  {serial:5d}  {name:<3s} {resn} A{i:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}"
                )
                serial += 1
        lines.append("TER")
        return "\n".join(lines) + "\n"

    def write_pdb(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write(self.to_pdb_block())
