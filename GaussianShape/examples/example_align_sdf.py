"""Demo: align a randomly displaced molecule back onto a reference, then export
SDF files for PyMOL visualization.

Scenario
--------
1. Build a reference molecule in 3D with RDKit.
2. Make a copy and move it FAR away with a random rotation + large translation.
3. Confirm the shape overlap is ~0 in that displaced pose.
4. Recover the rigid transform with Gaussian-shape BFGS alignment.
5. Apply the transform to the actual molecule and write three SDF files:
   reference, displaced-start, and aligned. Load them in PyMOL to inspect.

Run from the project root::

    python examples/example_align_sdf.py

Then in PyMOL::

    load examples/output/ref.sdf
    load examples/output/mobile_start.sdf
    load examples/output/mobile_aligned.sdf
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation

from core.gaussian_shape import (
    GaussianShape,
    align_shapes_bfgs,
    apply_transform_to_mol,
)


def make_mol_3d(smiles: str, seed: int = 0):
    """Build an RDKit molecule with hydrogens and one embedded 3D conformer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"3D embedding failed for {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def write_sdf(mol, path: str, name: str) -> None:
    """Write a single-conformer molecule to an SDF file with a title."""
    mol = Chem.Mol(mol)  # work on a copy so we can set the title safely
    mol.SetProp("_Name", name)
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Reference molecule (ibuprofen -- a nicely anisotropic, flexible shape).
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    ref_mol = make_mol_3d(smiles, seed=1)
    ref_shape = GaussianShape.from_rdkit_mol(ref_mol, radius_scale=0.8)

    # 2) Copy and displace FAR away with a random rotation + large translation.
    mobile_mol = Chem.Mol(ref_mol)
    rng = np.random.default_rng(42)
    rand_rot = Rotation.random(random_state=42).as_matrix()
    rand_trans = np.array([35.0, -28.0, 22.0]) + rng.uniform(-5, 5, size=3)
    apply_transform_to_mol(mobile_mol, rand_rot, rand_trans)
    mobile_shape = GaussianShape.from_rdkit_mol(mobile_mol, radius_scale=0.8)

    ref_c = ref_shape.centroid()
    mob_c = mobile_shape.centroid()
    print("Reference centroid:", np.round(ref_c, 2))
    print("Displaced centroid:", np.round(mob_c, 2))
    print(f"Centroid separation: {np.linalg.norm(mob_c - ref_c):.2f} Angstrom")

    # 3) Overlap in the displaced pose (should be essentially zero).
    before = ref_shape.tanimoto(mobile_shape)
    print(f"\nTanimoto before alignment: {before:.4f}")

    # 4) Recover the transform via Gaussian-shape BFGS alignment.
    result = align_shapes_bfgs(mobile_shape, ref_shape, n_starts=16)
    print(f"Tanimoto after alignment : {result.tanimoto:.4f}")
    print("Recovered rotation-translation matrix:")
    print(np.array2string(result.matrix, precision=3, suppress_small=True))

    # 5) Apply the transform to the actual molecule and export SDF files.
    aligned_mol = Chem.Mol(mobile_mol)
    apply_transform_to_mol(aligned_mol, result.rotation, result.translation)

    p_ref = os.path.join(out_dir, "ref.sdf")
    p_start = os.path.join(out_dir, "mobile_start.sdf")
    p_aligned = os.path.join(out_dir, "mobile_aligned.sdf")
    write_sdf(ref_mol, p_ref, "reference")
    write_sdf(mobile_mol, p_start, "mobile_displaced")
    write_sdf(aligned_mol, p_aligned, "mobile_aligned")

    # Sanity check: aligned molecule matches the aligned shape to machine eps.
    aligned_shape = GaussianShape.from_rdkit_mol(aligned_mol, radius_scale=0.8)
    max_dev = float(np.abs(aligned_shape.centers - result.aligned.centers).max())
    print(f"\nMax coord deviation (aligned mol vs aligned shape): {max_dev:.2e}")

    print("\nSDF files written:")
    print(f"  {p_ref}")
    print(f"  {p_start}")
    print(f"  {p_aligned}")
    print("\nIn PyMOL:")
    print(f"  load {p_ref}")
    print(f"  load {p_start}")
    print(f"  load {p_aligned}")


if __name__ == "__main__":
    main()
