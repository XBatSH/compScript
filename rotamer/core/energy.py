"""MMFF94 energy evaluation and minimization for peptides.

We reuse RDKit's MMFF94 force field as the energy model. Two operations are
exposed:

* :func:`mmff_energy` - single-point energy of the current geometry.
* :func:`minimize` - relax the geometry; optionally the backbone atoms are held
  fixed so that only the side chains move (the usual setting after placing
  rotamers, analogous to SCWRL-style side-chain packing).
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit.Chem import AllChem

# Backbone atom names held fixed during side-chain relaxation.
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def _force_field(mol):
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        raise RuntimeError("MMFF parameters unavailable for this molecule")
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    if ff is None:
        raise RuntimeError("Could not build MMFF force field")
    ff.Initialize()
    return ff


def mmff_energy(mol) -> float:
    """Return the MMFF94 single-point energy (kcal/mol) of the current geometry."""
    return float(_force_field(mol).CalcEnergy())


@dataclass
class MinimizeResult:
    energy: float      # final MMFF energy (kcal/mol)
    converged: bool    # whether the minimizer reported convergence


def _backbone_indices(peptide) -> list[int]:
    idxs = []
    for atom in peptide.mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is not None and info.GetName().strip() in BACKBONE_ATOMS:
            idxs.append(atom.GetIdx())
    return idxs


def minimize(
    peptide,
    max_iters: int = 1000,
    restrain_backbone: bool = True,
) -> MinimizeResult:
    """Minimize a peptide's MMFF energy in place.

    Parameters
    ----------
    peptide : Peptide
        The peptide to relax (its conformer is modified in place).
    max_iters : int
        Maximum minimization iterations.
    restrain_backbone : bool
        If True, backbone atoms (N, CA, C, O) are fixed so only side chains move.

    Returns
    -------
    MinimizeResult
    """
    ff = _force_field(peptide.mol)
    if restrain_backbone:
        for idx in _backbone_indices(peptide):
            ff.AddFixedPoint(idx)
    status = ff.Minimize(maxIts=max_iters)
    return MinimizeResult(energy=float(ff.CalcEnergy()), converged=(status == 0))
