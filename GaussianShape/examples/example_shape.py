"""Demo: Gaussian molecular shape representation and similarity comparison.

Pipeline
--------
1. Build several molecules from SMILES and embed a 3D conformer with RDKit.
2. Convert each 3D structure into a Gaussian shape.
3. Compare shapes with the analytical overlap / Tanimoto similarity.
4. Optionally align a pair of shapes to maximize overlap.
5. Save 2D density-slice figures for visual inspection.

Run from the project root so the ``core`` / ``visualize`` packages import::

    python examples/example_shape.py
"""

from __future__ import annotations

import os
import sys

# Allow running the script directly from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

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


def similarity_matrix(shapes, names, align: bool = True):
    """Print a pairwise Tanimoto shape-similarity matrix.

    Shape similarity is only meaningful once the two molecules are placed in a
    common frame, so by default each off-diagonal pair is optimally aligned
    (BFGS, maximizing overlap) before the Tanimoto coefficient is computed.
    """
    n = len(shapes)
    header = "".join(f"{name:>10s}" for name in names)
    print(f"{'':>10s}{header}")
    for i in range(n):
        row = f"{names[i]:>10s}"
        for j in range(n):
            if i == j:
                sim = 1.0
            elif align:
                sim = align_shapes_bfgs(shapes[j], shapes[i], n_starts=8).tanimoto
            else:
                sim = shapes[i].tanimoto(shapes[j])
            row += f"{sim:>10.3f}"
        print(row)


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    # A small panel of molecules of increasingly different shape.
    library = {
        "benzene": "c1ccccc1",
        "toluene": "Cc1ccccc1",
        "cyclohexane": "C1CCCCC1",
        "n-hexane": "CCCCCC",
    }

    shapes = {}
    for name, smi in library.items():
        mol = make_mol_3d(smi)
        shape = GaussianShape.from_rdkit_mol(mol, radius_scale=0.8)
        shapes[name] = shape
        print(
            f"{name:>12s}: {shape.n_atoms:2d} atoms, "
            f"self-overlap S_AA = {shape.self_overlap():.2f}"
        )

    print("\nPairwise Tanimoto shape similarity (aligned)")
    print("-" * 60)
    names = list(shapes.keys())
    similarity_matrix([shapes[n] for n in names], names, align=True)

    # --- Alignment demo: two conformers of the same flexible molecule -----
    print("\nAlignment demo (two random conformers of n-hexane)")
    print("-" * 60)
    mol_a = make_mol_3d("CCCCCC", seed=1)
    mol_b = make_mol_3d("CCCCCC", seed=7)
    shape_a = GaussianShape.from_rdkit_mol(mol_a, radius_scale=0.8)
    shape_b = GaussianShape.from_rdkit_mol(mol_b, radius_scale=0.8)

    before = shape_a.tanimoto(shape_b)
    result = align_shapes_bfgs(shape_b, shape_a, n_starts=12)
    aligned_b, after = result.aligned, result.tanimoto
    print(f"Tanimoto before alignment: {before:.3f}")
    print(f"Tanimoto after alignment : {after:.3f}")
    print("Rotation-translation matrix (maps original mol_b coords -> aligned):")
    print(np.array2string(result.matrix, precision=3, suppress_small=True))

    # Apply the SAME transform to the source RDKit molecule so the actual
    # 3D structure is repositioned into the aligned frame. Rebuilding the shape
    # from the moved molecule must reproduce the aligned shape exactly.
    apply_transform_to_mol(mol_b, result.rotation, result.translation)
    shape_b_moved = GaussianShape.from_rdkit_mol(mol_b, radius_scale=0.8)
    max_dev = float(np.abs(shape_b_moved.centers - aligned_b.centers).max())
    print(f"Max coord deviation (moved mol vs aligned shape): {max_dev:.2e}")

    # --- Figures ----------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt

    from visualize.plot_shape import plot_density_slice, plot_shape_overlay

    # 1) Single-shape density slice.
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_density_slice(shapes["benzene"], axis="z", ax=ax,
                       title="Benzene shape density (z-slice)")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "benzene_density.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # 2) Overlay of two molecules before alignment.
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_shape_overlay(shapes["benzene"], shapes["cyclohexane"], axis="z", ax=ax,
                       labels=("benzene", "cyclohexane"),
                       title="Benzene vs cyclohexane")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "benzene_vs_cyclohexane.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)

    # 3) n-hexane conformers, aligned.
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_shape_overlay(shape_a, aligned_b, axis="z", ax=ax,
                       labels=("conf A", "conf B (aligned)"),
                       title=f"n-hexane conformers (T={after:.2f})")
    fig.tight_layout()
    p3 = os.path.join(out_dir, "hexane_aligned.png")
    fig.savefig(p3, dpi=130)
    plt.close(fig)

    print(f"\nSaved figures to:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()
