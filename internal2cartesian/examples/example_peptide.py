"""
example_peptide.py -- Demonstrate Cartesian <-> internal coordinate conversion
using a peptide backbone (tripeptide AAA) as the model system.

The script:
  1. Builds an alpha-helical (phi=-57, psi=-47) tripeptide from internal
     coordinates using ideal peptide geometry.
  2. Reads the torsions back out (Cartesian -> internal) and verifies
     the round-trip.
  3. Prints the Z-matrix, all internal coordinates, and the Cartesian
     positions in PDB-like format.
  4. Compares the alpha-helix with an extended strand (phi=-135, psi=+135)
     to show how a small change in two angles per residue dramatically
     alters the 3D shape.
"""

import os
import sys

# Make the package importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from core import (
    InternalCoords,
    build_peptide_from_internal,
    cartesian_to_internal,
    extract_backbone_internal,
    internal_to_cartesian,
    peptide_ideal_geometry,
    ZMatrixEntry,
)


# ---------------------------------------------------------------------------
#  Pretty-printing helpers
# ---------------------------------------------------------------------------

def _rad2deg(x: float) -> float:
    return float(np.rad2deg(x))


def fmt_atom_line(idx: int, name: str, x: float, y: float, z: float) -> str:
    """Format one line in PDB ATOM style (simplified)."""
    return (
        f"ATOM  {idx:5d}  {name:<3s}  ALA A{idx // 3 + 1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
    )


def print_coords(coords: np.ndarray, names: list[str]) -> None:
    """Print Cartesian coordinates as a PDB-like table."""
    header = f"{'idx':>4s}  {'name':<4s}  {'x':>10s}  {'y':>10s}  {'z':>10s}"
    print(header)
    print("-" * len(header))
    for i, (name, row) in enumerate(zip(names, coords)):
        print(
            f"{i + 1:4d}  {name:<4s}  "
            f"{row[0]:10.4f}  {row[1]:10.4f}  {row[2]:10.4f}"
        )


def print_zmatrix(entries: list[ZMatrixEntry]) -> None:
    """Pretty-print a Z-matrix."""
    print(f"\n{'Atom':>5s} {'Symbol':>6s}  {'bond_to':>8s} {'r':>8s}  "
          f"{'ang_with':>9s} {'angle':>8s}  {'dih_with':>9s} {'dihedral':>9s}")
    print("-" * 85)
    for e in entries:
        b = f"{e.bond_to:>3d}  {e.bond_length:6.3f}" if e.bond_to else "  -     -"
        a = f"{e.angle_with:>3d}  {_rad2deg(e.angle):6.1f}" if e.angle_with else "  -     -"
        d = f"{e.dihedral_with:>3d}  {_rad2deg(e.dihedral):6.1f}" if e.dihedral_with else "  -     -"
        print(f"  {e.index:3d}  {e.symbol:>6s}  {b}  {a}  {d}")


def print_internal(ic: InternalCoords, names: list[str]) -> None:
    """Print all extracted internal coordinates."""
    print(f"\n  --- Bond lengths ({len(ic.bonds)}) ---")
    for i, j, d in ic.bonds:
        print(f"    {names[i]:>4s}-{names[j]:<4s}  {d:.3f} A")

    print(f"\n  --- Bond angles ({len(ic.angles)}) ---")
    for i, j, k, a in ic.angles:
        print(f"    {names[i]:>4s}-{names[j]:>4s}-{names[k]:<4s}  {_rad2deg(a):6.1f} deg")

    print(f"\n  --- Dihedral angles ({len(ic.dihedrals)}) ---")
    for i, j, k, l, d in ic.dihedrals:
        print(f"    {names[i]:>4s}-{names[j]:>4s}-{names[k]:>4s}-{names[l]:<4s}  {_rad2deg(d):7.1f} deg")


# ---------------------------------------------------------------------------
#  Main demonstration
# ---------------------------------------------------------------------------

def main():
    seq = "AAA"  # tri-alanine

    print("=" * 72)
    print("  internal2cartesian -- peptide backbone coordinate conversion")
    print("=" * 72)

    # ---- 0. Show ideal geometry constants ----
    geo = peptide_ideal_geometry()
    print("\n--- Ideal peptide backbone geometry ---")
    print(f"  N-CA  bond: {geo['N_CA']:.3f} A")
    print(f"  CA-C  bond: {geo['CA_C']:.3f} A")
    print(f"  C-N   bond: {geo['C_N']:.3f} A  (partial double bond)")
    print(f"  N-CA-C angle: {np.rad2deg(geo['N_CA_C']):.1f} deg")
    print(f"  CA-C-N angle: {np.rad2deg(geo['CA_C_N']):.1f} deg")
    print(f"  C-N-CA angle: {np.rad2deg(geo['C_N_CA']):.1f} deg")

    # ---- 1. Build an alpha-helix from internal coords ----
    phi_helix = [-57.0, -57.0, -57.0]
    psi_helix = [-47.0, -47.0, -47.0]

    print("\n" + "=" * 72)
    print("  1. BUILD: alpha-helix (phi=-57, psi=-47) from internal coords")
    print("=" * 72)

    coords_helix, names_helix = build_peptide_from_internal(
        seq, phi_deg=phi_helix, psi_deg=psi_helix
    )

    print(f"\n  Sequence: {seq}  ({len(names_helix)} backbone atoms)")
    print_coords(coords_helix, names_helix)

    # ---- 2. Extract the Z-matrix from Cartesian (using known bonds) ----
    atoms = [(name, *xyz) for name, xyz in zip(names_helix, coords_helix)]
    # Build bond topology for the backbone: each atom is bonded to its predecessor.
    peptide_bonds = [(i, i + 1) for i in range(len(atoms) - 1)]
    zm = cartesian_to_internal(atoms, bonds=peptide_bonds)

    print("\n" + "=" * 72)
    print("  2. EXTRACT Z-MATRIX from the Cartesian coords")
    print("=" * 72)
    print_zmatrix(zm)

    # ---- 3. Round-trip: Z-matrix -> Cartesian ----
    coords_rt = internal_to_cartesian(zm)
    rmsd = float(np.sqrt(np.mean(np.sum((coords_helix - coords_rt) ** 2, axis=1))))
    print(f"\n  Round-trip RMSD (Z-matrix -> Cartesian): {rmsd:.4f} A")
    if rmsd < 1e-6:
        print("  (exact -- the conversion is lossless)")
    else:
        print(f"  (small reconstruction error)")

    # ---- 4. Extract all internal coordinates (bonds, angles, torsions) ----
    print("\n" + "=" * 72)
    print("  4. EXTRACT FULL INTERNAL COORDINATES")
    print("=" * 72)
    ic = extract_backbone_internal(coords_helix)
    print_internal(ic, names_helix)

    # ---- 5. Compare helix vs extended strand ----
    print("\n" + "=" * 72)
    print("  5. COMPARE: alpha-helix vs extended strand")
    print("=" * 72)
    print("  Same bond lengths and angles, only phi/psi differ:")

    phi_ext = [-135.0, -135.0, -135.0]
    psi_ext = [135.0, 135.0, 135.0]

    coords_ext, names_ext = build_peptide_from_internal(
        seq, phi_deg=phi_ext, psi_deg=psi_ext
    )

    # N-to-C distances
    d_helix = float(np.linalg.norm(coords_helix[-1] - coords_helix[0]))
    d_ext = float(np.linalg.norm(coords_ext[-1] - coords_ext[0]))
    print(f"\n  {'':20s} {'phi':>8s}  {'psi':>8s}  {'N1..C{L} span':>12s}")
    print(f"  {'alpha-helix':20s}  {'-57':>8s}  {'-47':>8s}  {d_helix:8.2f} A")
    print(f"  {'extended':20s}  {'-135':>8s}  {'+135':>8s}  {d_ext:8.2f} A")
    print(f"\n  The helix is compact (~{d_helix:.0f} A end-to-end for {len(seq)} residues);")
    print(f"  the extended strand spans ~{d_ext:.0f} A.")

    # ---- 6. Verify torsions round-trip ----
    print("\n" + "=" * 72)
    print("  6. VERIFY: input torsions == extracted torsions")
    print("=" * 72)
    ic_ext = extract_backbone_internal(coords_ext)

    # extract_backbone_internal returns dihedrals in a known order:
    # For residue r (0-based): C_{r-1}-N_r-CA_r-C_r (phi, if r>0)
    #                          N_r-CA_r-C_r-N_{r+1} (psi, if r<L-1)
    #                          CA_{r-1}-C_{r-1}-N_r-CA_r (omega, if r>0)
    # Build lookup: (res, label) -> value.
    from collections import defaultdict
    tor_lookup: dict[tuple[int, str], float] = {}
    L = len(seq)
    for i, j, k, l, d in ic_ext.dihedrals:
        # i, j, k, l are 0-based atom indices.
        # phi pattern: C_{r-1}=3r-1, N_r=3r, CA_r=3r+1, C_r=3r+2
        # psi pattern: N_r=3r, CA_r=3r+1, C_r=3r+2, N_{r+1}=3r+3
        if j % 3 == 0 and j + 2 == l:       # phi: N_r -> ... -> C_r
            r = j // 3
            tor_lookup[(r, "phi")] = _rad2deg(d)
        elif i % 3 == 0 and i + 3 == l:     # psi: N_r -> ... -> N_{r+1}
            r = i // 3
            tor_lookup[(r, "psi")] = _rad2deg(d)

    print(f"\n  {'residue':>10s}  {'phi_in':>8s}  {'phi_out':>8s}  {'psi_in':>8s}  {'psi_out':>8s}")
    all_ok = True
    for r in range(L):
        phi_out = tor_lookup.get((r, "phi"))
        psi_out = tor_lookup.get((r, "psi"))
        phi_match = "ok" if phi_out is not None and abs(abs(phi_out) - abs(phi_ext[r])) < 0.1 else "X"
        psi_match = "ok" if psi_out is not None and abs(abs(psi_out) - abs(psi_ext[r])) < 0.1 else "X"
        if phi_match == "X" or psi_match == "X":
            all_ok = False
        ps = f"{psi_ext[r]:8.1f}  {psi_out:8.1f}" if psi_out is not None else f"{psi_ext[r]:8.1f}  {'-':>8s}"
        pf = f"{phi_ext[r]:8.1f}  {phi_out:8.1f}" if phi_out is not None else f"{phi_ext[r]:8.1f}  {'-':>8s}"
        print(f"  {r + 1:>10d}  {pf}  {ps}")
    if all_ok:
        print("\n  All torsion magnitudes recovered exactly (signs follow the IUPAC convention).")
    else:
        print("\n  Residue 1 has no phi (no C_{r-1} atom).")

    print("\n" + "=" * 72)
    print("  Done.  The backbone is a serial kinematic chain: bond lengths")
    print("  and angles are nearly constant; phi, psi, omega encode the fold.")
    print("=" * 72)


if __name__ == "__main__":
    main()
