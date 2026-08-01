"""
example_torsion_propagation.py — What happens when you change ONE torsion?

If you change a single dihedral angle in a molecule described by internal
coordinates, every atom placed *after* the rotation axis must move — even
if its own bond length, bond angle, and dihedral stay exactly the same.

This is forward kinematics: the molecule is a kinematic chain, and each
torsion is a revolute joint.  Rotating joint k rotates every link from
k+1 to the end of the chain.

This demo builds a penta-alanine backbone (15 atoms), then:

  1. Rotates ψ₂ by +30° and shows which atoms moved.
  2. Rotates φ₃ by +30° and shows which atoms moved.
  3. Explains the "downstream slice" concept used in the kinematics_loop
     module for loop closure.

The key insight
---------------
In internal coordinates, atom D is defined by:
    D = f(A, B, C, bond, angle, torsion)

If we change torsion, D moves.  Any atom E that was defined using D as
one of its reference atoms (A, B, or C) will also move — even though
E's *own* internal coordinates are unchanged — because its reference
frame has been rotated.

For a peptide backbone with atoms [N₁, CA₁, C₁, N₂, CA₂, C₂, …, N_L, CA_L, C_L]:

  φᵢ = C_{i-1} – N_i – CA_i – C_i       rotates C_i and everything after
  ψᵢ = N_i – CA_i – C_i – N_{i+1}       rotates N_{i+1} and everything after

The "downstream slice" of a torsion is the list of atoms that move when
that torsion changes.  For φᵢ the slice starts at C_i; for ψᵢ the slice
starts at N_{i+1}.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from core import (
    build_peptide_from_internal,
    extract_backbone_internal,
    normalize,
)


def rotate_points(points, origin, axis, theta):
    """Rodrigues rotation: rotate *points* by *theta* (rad) around the
    line through *origin* with direction *axis*.

    This is the same operation as kinematics_loop/core/geometry.py:rotate_points.
    """
    k = normalize(axis)
    v = points - origin
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotated = (
        v * cos_t
        + np.cross(k, v) * sin_t
        + np.outer(v @ k, k) * (1.0 - cos_t)
    )
    return origin + rotated


def apply_torsion_change(
    coords: np.ndarray,
    names: list[str],
    torsion_atoms: tuple[int, int, int, int],  # 0-based indices of A, B, C
    slice_start: int,                           # 0-based index of first moving atom
    delta_deg: float,
) -> np.ndarray:
    """Rotate all atoms from *slice_start* onward around the B–C bond axis.

    Parameters
    ----------
    coords : (n, 3) array of Cartesian positions before the change.
    names : atom labels (for display only, not modified).
    torsion_atoms : (a, b, c, _) — A, B, C define the axis B→C.
    slice_start : first atom index (0-based) that should rotate.
    delta_deg : rotation angle in **degrees**.

    Returns
    -------
    new_coords : (n, 3) array with the rotated positions.
    """
    a, b, c, _ = torsion_atoms
    origin = coords[b]
    axis = coords[c] - coords[b]
    delta_rad = np.deg2rad(delta_deg)

    moved = coords.copy()
    moved[slice_start:] = rotate_points(
        coords[slice_start:], origin, axis, delta_rad
    )
    return moved


# ---------------------------------------------------------------------------
def main():
    seq = "AAAAA"  # penta-alanine
    L = len(seq)

    # Build an alpha-helix.
    phi = [-57.0] * L
    psi = [-47.0] * L
    coords_ref, names = build_peptide_from_internal(seq, phi_deg=phi, psi_deg=psi)

    print("=" * 68)
    print("  Torsion propagation: change ONE angle, watch the chain move")
    print("=" * 68)
    print(f"\n  Reference: alpha-helical {seq}  (φ=-57°, ψ=-47°)")
    print(f"  {len(names)} atoms:  ", "  ".join(names))
    print(f"  End-to-end N1…C{L} span: {np.linalg.norm(coords_ref[-1] - coords_ref[0]):.2f} Å")

    # ---- Example 1: rotate ψ₂ by +30° ----
    # ψ₂ = N₂ – CA₂ – C₂ – N₃, involving atoms 3,4,5,6 (0-based)
    # The rotation axis is CA₂ → C₂ (atoms 4 → 5).
    # The downstream slice starts at N₃ (atom 6).
    print("\n" + "-" * 68)
    print("  Example 1: rotate ψ₂ (N₂–CA₂–C₂–N₃) by +30°")
    print("-" * 68)
    print("  ψ₂ is the torsion at residue 2.")
    print("  Rotation axis: CA₂ → C₂  (atoms 4→5, 0-based)")
    print("  Downstream atoms: N₃, CA₃, C₃, N₄, CA₄, C₄, N₅, CA₅, C₅")
    print("                    (indices 6 through 14)")

    psi2_atoms = (3, 4, 5, 6)   # N₂, CA₂, C₂, N₃  (0-based)
    psi2_slice = 6               # N₃ is the first atom that moves

    coords_new = apply_torsion_change(
        coords_ref, names, psi2_atoms, psi2_slice, delta_deg=+30.0
    )

    # Show displacement of each atom.
    disp = np.linalg.norm(coords_new - coords_ref, axis=1)
    print(f"\n  {'atom':>6s}  {'index':>5s}  {'displacement (Å)':>18s}  {'moved?':>8s}")
    print("  " + "-" * 42)
    for i, (name, d) in enumerate(zip(names, disp)):
        marker = "  <<<" if i >= psi2_slice else ""
        print(f"  {name:>6s}  {i:>5d}  {d:18.4f}{marker}")
    print(f"\n  ✓ Atoms 0-5 (N₁…C₂) are stationary.")
    print(f"  ✓ Atoms 6-14 (N₃…C₅) all moved — by up to {disp[psi2_slice:].max():.2f} Å.")

    # Read the new ψ₂ and confirm only that one torsion changed.
    ic_ref = extract_backbone_internal(coords_ref)
    ic_new = extract_backbone_internal(coords_new)

    print(f"\n  Torsion changes (before → after):")
    for (i, j, k, l, d_old), (_, _, _, _, d_new) in zip(ic_ref.dihedrals, ic_new.dihedrals):
        delta = np.rad2deg(d_new - d_old)
        # Normalise delta to [-180, 180].
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360
        label = f"{names[i]}-{names[j]}-{names[k]}-{names[l]}"
        marker = " ← ψ₂" if (i, j, k, l) == psi2_atoms else ""
        if abs(delta) > 0.5:
            print(f"    {label:30s}  {np.rad2deg(d_old):7.1f}° → {np.rad2deg(d_new):7.1f}°  (Δ = {delta:+6.1f}°){marker}")
        else:
            print(f"    {label:30s}  {np.rad2deg(d_old):7.1f}° → {np.rad2deg(d_new):7.1f}°  (Δ = {delta:+6.1f}°)")

    # ---- Example 2: rotate φ₃ by +30° ----
    # φ₃ = C₂ – N₃ – CA₃ – C₃, involving atoms 5,6,7,8 (0-based)
    # The rotation axis is N₃ → CA₃ (atoms 6 → 7).
    # The downstream slice starts at C₃ (atom 8).
    print("\n" + "-" * 68)
    print("  Example 2: rotate φ₃ (C₂–N₃–CA₃–C₃) by +30°")
    print("-" * 68)
    print("  φ₃ is the torsion at residue 3.")
    print("  Rotation axis: N₃ → CA₃  (atoms 6→7, 0-based)")
    print("  Downstream atoms: C₃, N₄, CA₄, C₄, N₅, CA₅, C₅")
    print("                    (indices 8 through 14)")

    phi3_atoms = (5, 6, 7, 8)    # C₂, N₃, CA₃, C₃  (0-based)
    phi3_slice = 8                # C₃ is the first atom that moves

    coords_new2 = apply_torsion_change(
        coords_ref, names, phi3_atoms, phi3_slice, delta_deg=+30.0
    )

    disp2 = np.linalg.norm(coords_new2 - coords_ref, axis=1)
    print(f"\n  {'atom':>6s}  {'index':>5s}  {'displacement (Å)':>18s}  {'moved?':>8s}")
    print("  " + "-" * 42)
    for i, (name, d) in enumerate(zip(names, disp2)):
        marker = "  <<<" if i >= phi3_slice else ""
        print(f"  {name:>6s}  {i:>5d}  {d:18.4f}{marker}")
    print(f"\n  ✓ Atoms 0-7 (N₁…CA₃) are stationary — the N-terminal half is frozen.")
    print(f"  ✓ Atoms 8-14 (C₃…C₅) all moved.")

    # ---- Conceptual: side-chain torsions ----
    print("\n" + "=" * 68)
    print("  Extension to side chains")
    print("=" * 68)
    print("""
  The same principle applies to side-chain χ (chi) torsions.

  Consider a lysine side chain attached to CA:

      CA – CB – CG – CD – CE – NZ

  Each χ angle is a revolute joint:
      χ₁ = N–CA–CB–CG     rotates CG, CD, CE, NZ
      χ₂ = CA–CB–CG–CD    rotates CD, CE, NZ
      χ₃ = CB–CG–CD–CE    rotates CE, NZ
      χ₄ = CG–CD–CE–NZ    rotates NZ only

  Changing χ₁ moves 4 atoms; changing χ₄ moves only 1 atom.  This is
  why rotamer libraries store all χ angles — the cumulative effect of
  changing χ₁ is much larger than changing χ₄.

  This is exactly what the `rotamer` module does: it sets all χ angles
  on a residue, then recomputes the Cartesian positions from the
  internal coordinates using forward kinematics (via RDKit's SetTorsion
  or equivalent).
""")

    # ---- Summary ----
    print("=" * 68)
    print("  Summary")
    print("=" * 68)
    print("""
  Internal → Cartesian conversion is a *forward kinematics* problem:
  each atom's position depends on the atoms placed before it.  Changing
  one torsion rotates the entire "downstream" segment of the chain.

  This is why the `kinematics_loop` module's backbone has a
  `downstream_slice` property and an `apply_rotation(kind, i, theta)`
  method — they implement exactly the rotation operation demonstrated
  above.

  For side chains, the same physics applies: χ₁ rotates everything from
  Cγ onward, χ₂ rotates from Cδ onward, etc.
""")


if __name__ == "__main__":
    main()
