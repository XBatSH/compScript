"""
core -- A minimal Hartree-Fock program for teaching quantum chemistry.

Pure Python: numpy + scipy for the math, RDKit for 3D structure generation.

Quick start:
    from core import Molecule, rhf
    mol = Molecule.from_smiles("O")     # water
    result = rhf(mol)
    print(result["energy"])             # total RHF/STO-3G energy in Hartree
"""

from .molecule import Molecule, Atom
from .basis import build_basis, BasisFunction
from .scf import rhf

__all__ = ["Molecule", "Atom", "BasisFunction", "build_basis", "rhf"]
