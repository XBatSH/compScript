"""
molecule.py -- Molecular structure for the teaching Hartree-Fock code.

A Molecule is just a list of atoms (element symbol, nuclear charge Z,
Cartesian coordinates in ATOMIC UNITS / Bohr) plus a total charge.

Two ways to build one:
  1. Molecule.from_atoms(...)   -- give coordinates yourself (textbook examples)
  2. Molecule.from_smiles(...)  -- let RDKit generate a 3D geometry from SMILES

Units: quantum chemistry works in atomic units.
  1 Angstrom = 1.8897259886 Bohr
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Conversion factor: Angstrom -> Bohr (atomic unit of length)
ANGSTROM_TO_BOHR = 1.8897259886

# Elements supported by our little STO-3G basis set library (see basis.py)
ELEMENT_Z = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
}


@dataclass
class Atom:
    symbol: str                 # element symbol, e.g. "O"
    Z: int                      # nuclear charge
    coord: np.ndarray           # position in Bohr, shape (3,)


@dataclass
class Molecule:
    atoms: list[Atom]
    charge: int = 0
    name: str = "molecule"

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_atoms(cls, atom_list, charge=0, unit="angstrom", name="molecule"):
        """Build a molecule from [(symbol, (x, y, z)), ...].

        unit : "angstrom" (default) or "bohr"
        """
        scale = ANGSTROM_TO_BOHR if unit.lower().startswith("ang") else 1.0
        atoms = []
        for symbol, xyz in atom_list:
            symbol = symbol.capitalize()
            if symbol not in ELEMENT_Z:
                raise ValueError(f"Element '{symbol}' not supported (H-F only).")
            atoms.append(Atom(symbol, ELEMENT_Z[symbol],
                              np.asarray(xyz, dtype=float) * scale))
        return cls(atoms, charge=charge, name=name)

    @classmethod
    def from_smiles(cls, smiles, charge=None, name=None, seed=42):
        """Build a 3D structure from a SMILES string using RDKit.

        RDKit gives us a *force-field* geometry (ETKDG embedding + MMFF94
        optimization).  It is not the HF minimum, but it is a perfectly
        reasonable starting structure for a single-point HF calculation.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES '{smiles}'")
        mol = Chem.AddHs(mol)                       # SMILES hides hydrogens!

        params = AllChem.ETKDGv3()                  # distance-geometry embed
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"3D embedding failed for '{smiles}'")
        # Force-field refinement (MMFF94 if parameters exist, else UFF)
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        atom_list = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            atom_list.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))

        if charge is None:
            charge = Chem.GetFormalCharge(mol)
        return cls.from_atoms(atom_list, charge=charge, unit="angstrom",
                              name=name or smiles)

    # ------------------------------------------------------------------ #
    # Simple properties
    # ------------------------------------------------------------------ #
    @property
    def n_electrons(self) -> int:
        return sum(a.Z for a in self.atoms) - self.charge

    def nuclear_repulsion(self) -> float:
        """E_nn = sum_{A<B} Z_A Z_B / |R_A - R_B|   (atomic units)."""
        e_nn = 0.0
        for i, a in enumerate(self.atoms):
            for b in self.atoms[i + 1:]:
                e_nn += a.Z * b.Z / np.linalg.norm(a.coord - b.coord)
        return e_nn

    def __str__(self):
        lines = [f"Molecule: {self.name}  (charge {self.charge:+d}, "
                 f"{self.n_electrons} electrons)"]
        lines.append(f"{'atom':>4} {'x/Bohr':>12} {'y/Bohr':>12} {'z/Bohr':>12}")
        for a in self.atoms:
            x, y, z = a.coord
            lines.append(f"{a.symbol:>4} {x:12.6f} {y:12.6f} {z:12.6f}")
        return "\n".join(lines)
