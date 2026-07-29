"""Build a peptide from sequence and manipulate its side-chain chi angles.

The :class:`Peptide` wraps an RDKit molecule that carries PDB atom/residue names
(as produced by ``Chem.MolFromSequence``). Those names let us locate the four
atoms of every chi dihedral and set/read them with ``rdMolTransforms``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms as rmt

from .residues import chi_atom_names, n_chi
from .rotamer_lib import Rotamer


@dataclass(frozen=True)
class ResidueInfo:
    """Lightweight description of one residue in the peptide."""

    number: int   # PDB residue number (1-based, in sequence order)
    name: str     # 3-letter residue name, e.g. "LYS"
    n_chi: int    # number of rotatable chi angles


class Peptide:
    """A 3D peptide whose side-chain rotamers can be set and queried."""

    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        # Build a (residue_number, atom_name) -> atom index lookup.
        self._index: dict[tuple[int, str], int] = {}
        residues: dict[int, str] = {}
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is None:
                continue
            resnum = info.GetResidueNumber()
            name = info.GetName().strip()
            self._index[(resnum, name)] = atom.GetIdx()
            residues[resnum] = info.GetResidueName().strip()
        self._residues = [
            ResidueInfo(number=num, name=residues[num], n_chi=n_chi(residues[num]))
            for num in sorted(residues)
        ]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_sequence(cls, sequence: str, seed: int = 0xF00D) -> "Peptide":
        """Build a peptide from a 1-letter sequence and embed a 3D structure.

        The backbone is given a reasonable starting geometry via ETKDG embedding
        followed by a short MMFF relaxation. Side-chain rotamers are then set on
        top of this backbone.
        """
        mol = Chem.MolFromSequence(sequence)
        if mol is None:
            raise ValueError(f"Could not build peptide from sequence: {sequence!r}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("3D embedding failed")
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        return cls(mol)

    def copy(self) -> "Peptide":
        """Return a deep copy (independent conformer) of this peptide."""
        return Peptide(Chem.Mol(self.mol))

    # ------------------------------------------------------------------
    # Residue / atom access
    # ------------------------------------------------------------------
    @property
    def residues(self) -> list[ResidueInfo]:
        return list(self._residues)

    def residue(self, number: int) -> ResidueInfo:
        for res in self._residues:
            if res.number == number:
                return res
        raise KeyError(f"No residue with number {number}")

    def _atom_index(self, resnum: int, atom_name: str) -> int:
        try:
            return self._index[(resnum, atom_name)]
        except KeyError:
            raise KeyError(
                f"Atom {atom_name!r} not found in residue {resnum}"
            ) from None

    # ------------------------------------------------------------------
    # Chi angle get / set
    # ------------------------------------------------------------------
    def get_chi(self, resnum: int, chi_index: int) -> float:
        """Return chi angle ``chi_index`` (1-based) of a residue, in degrees."""
        res = self.residue(resnum)
        names = chi_atom_names(res.name)
        if not (1 <= chi_index <= len(names)):
            raise IndexError(f"{res.name}{resnum} has no chi{chi_index}")
        idxs = [self._atom_index(resnum, nm) for nm in names[chi_index - 1]]
        return rmt.GetDihedralDeg(self.mol.GetConformer(), *idxs)

    def set_chi(self, resnum: int, chi_index: int, angle_deg: float) -> None:
        """Set chi angle ``chi_index`` (1-based) of a residue to ``angle_deg``."""
        res = self.residue(resnum)
        names = chi_atom_names(res.name)
        if not (1 <= chi_index <= len(names)):
            raise IndexError(f"{res.name}{resnum} has no chi{chi_index}")
        idxs = [self._atom_index(resnum, nm) for nm in names[chi_index - 1]]
        rmt.SetDihedralDeg(self.mol.GetConformer(), *idxs, float(angle_deg))

    def get_all_chi(self, resnum: int) -> tuple[float, ...]:
        """Return all chi angles of a residue as a tuple (degrees)."""
        res = self.residue(resnum)
        return tuple(self.get_chi(resnum, i + 1) for i in range(res.n_chi))

    def set_rotamer(self, resnum: int, rotamer: Rotamer) -> None:
        """Apply every chi angle of a :class:`Rotamer` to a residue.

        Chi angles are set from chi1 outward so that rotating an inner bond does
        not disturb an already-placed outer angle.
        """
        for i, angle in enumerate(rotamer.chi):
            self.set_chi(resnum, i + 1, angle)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def to_pdb_block(self) -> str:
        return Chem.MolToPDBBlock(self.mol)

    def write_pdb(self, path: str) -> None:
        Chem.MolToPDBFile(self.mol, path)
