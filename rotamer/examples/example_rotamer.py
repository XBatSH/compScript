"""Demo: build a peptide, place side-chain rotamers, and minimize.

Run from the ``rotamer/`` directory:

    python examples/example_rotamer.py

It (1) builds a small peptide from its sequence, (2) shows the per-rotamer energy
scan of one residue, (3) greedily constructs a low-energy conformation from the
rotamer library, (4) minimizes it, and (5) writes start/optimized PDB files.
"""

import os
import sys

# Make the package importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import (
    Peptide,
    build_low_energy_conformation,
    enumerate_rotamers,
    score_residue_rotamers,
    solve_rotamers,
)

OUTPUT_DIR = os.path.join(_HERE, "output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # A small peptide with several rotatable side chains.
    sequence = "KLVFF"  # Lys-Leu-Val-Phe-Phe (an amyloid-beta core motif)
    print(f"Building peptide: {sequence}")
    peptide = Peptide.from_sequence(sequence)

    print("\nResidues and rotatable chi angles:")
    for res in peptide.residues:
        n_rot = len(enumerate_rotamers(res.name, max_chi=2))
        print(f"  {res.name}{res.number}: n_chi={res.n_chi}, "
              f"chi1/chi2 rotamers={n_rot}")

    # --- Rotamer energy scan for the first flexible residue (Lys1). ---
    print("\nRotamer energy scan for LYS1 (chi1/chi2 staggered rotamers):")
    for score in score_residue_rotamers(peptide, 1, max_chi=2)[:5]:
        chi_txt = ", ".join(f"{c:+.0f}" for c in score.rotamer.chi)
        print(f"  rotamer {score.rotamer.name:2s} chi=({chi_txt})"
              f"  E={score.energy:8.2f} kcal/mol")

    # --- Build the low-energy conformation. ---
    print("\nGreedy rotamer construction + minimization:")
    result = build_low_energy_conformation(peptide, max_chi=2, verbose=True)

    print("\nChosen rotamers:")
    for resnum in sorted(result.assignments):
        print(f"  residue {resnum}: {result.assignments[resnum]}")

    print("\nEnergy summary (kcal/mol):")
    print(f"  embedded start   : {result.energy_initial:8.2f}")
    print(f"  after rotamers   : {result.energy_constructed:8.2f}")
    print(f"  after minimize   : {result.energy_minimized:8.2f}")

    # --- Compare combinatorial solvers on the same peptide. ---
    print("\nCombinatorial solvers (Dead-End Elimination / simulated annealing):")
    header = f"  {'method':8s} {'packing':>9s} {'constructed':>12s} {'minimized':>10s}"
    print(header)
    print(f"  {'greedy':8s} {'-':>9s} {result.energy_constructed:12.2f} "
          f"{result.energy_minimized:10.2f}")
    best = result
    for method in ("sa", "dee", "dee+sa"):
        r = solve_rotamers(peptide, method=method, max_chi=2, sa_steps=4000, seed=1)
        pack = "-" if r.packing_energy is None else f"{r.packing_energy:9.2f}"
        print(f"  {method:8s} {pack:>9s} {r.energy_constructed:12.2f} "
              f"{r.energy_minimized:10.2f}  -> {r.assignments}")
        if r.energy_minimized < best.energy_minimized:
            best = r

    # --- Write structures for viewing (PyMOL / VMD). ---
    start_path = os.path.join(OUTPUT_DIR, "peptide_start.pdb")
    opt_path = os.path.join(OUTPUT_DIR, "peptide_optimized.pdb")
    peptide.write_pdb(start_path)
    best.peptide.write_pdb(opt_path)
    print(f"\nBest method: {best.method}  (E={best.energy_minimized:.2f} kcal/mol)")
    print(f"Wrote:\n  {start_path}\n  {opt_path}")


if __name__ == "__main__":
    main()
