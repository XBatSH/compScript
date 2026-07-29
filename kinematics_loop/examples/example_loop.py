"""Demo: close a protein loop between two fixed anchors with CCD.

Scenario: we know the loop *sequence* and the backbone positions of the two
residues that flank it (the N-anchor and the C-anchor). We want backbone
conformations of the loop that connect the anchors.

For a self-checking demo we synthesize the anchors from a known reference loop,
then throw the torsions away and let CCD rediscover closed conformations from
random starts. In a real task the anchors instead come from a PDB structure.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from core import LoopProblem, close_loop, LoopBackbone

OUTPUT_DIR = os.path.join(_HERE, "output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # An 8-residue loop. The reference torsions define where the C-anchor sits.
    sequence = "GSDGKTPN"
    rng = np.random.default_rng(7)
    ref_phi = rng.uniform(-120, -40, size=len(sequence))
    ref_psi = rng.uniform(-60, 150, size=len(sequence))

    problem, reference = LoopProblem.from_reference(sequence, ref_phi, ref_psi)
    print(f"Loop sequence : {sequence}  ({len(sequence)} residues)")
    print(f"N-anchor (C0,N1,CA1):\n{np.round(problem.seed, 2)}")
    print(f"C-anchor (N_L,CA_L,C_L):\n{np.round(problem.targets, 2)}")
    span = np.linalg.norm(problem.targets[1] - problem.seed[2])
    print(f"anchor CA-CA span: {span:.2f} A")

    # --- Single closure from an extended start, showing convergence. ---
    print("\nSingle CCD closure from an extended start:")
    ext = LoopBackbone.from_torsions(
        sequence,
        np.radians(np.full(len(sequence), -135.0)),
        np.radians(np.full(len(sequence), 135.0)),
        problem.seed,
    )
    res = close_loop(ext, problem.targets)
    print(f"  converged={res.converged}  iterations={res.iterations}  "
          f"final RMSD={res.rmsd:.4f} A")
    milestones = [0, 1, 4, 9]
    shown = [(m + 1, res.history[m]) for m in milestones if m < len(res.history)]
    print("  RMSD by sweep: " + ", ".join(f"#{k}:{v:.2f}" for k, v in shown))

    # --- Multi-start solver: diverse conformations ranked by energy. ---
    print("\nMulti-start CCD (closed conformations, ranked by energy):")
    solutions = problem.solve(n_solutions=5, max_tries=300, seed=1, verbose=True)
    print(f"\nFound {len(solutions)} closed conformations (best energy first):")
    print(f"  {'#':>2} {'energy':>9} {'rmsd':>7} {'iters':>6} {'clashes':>8} {'rama_bad':>9}")
    for k, s in enumerate(solutions, 1):
        print(f"  {k:>2} {s.energy:9.2f} {s.rmsd:7.3f} {s.iterations:6d} "
              f"{s.clashes:8d} {s.rama_bad:9d}")

    # --- Write the reference and the best solution for viewing. ---
    ref_path = os.path.join(OUTPUT_DIR, "loop_reference.pdb")
    reference.write_pdb(ref_path)
    written = [ref_path]
    if solutions:
        best_path = os.path.join(OUTPUT_DIR, "loop_solution_best.pdb")
        solutions[0].backbone.write_pdb(best_path)
        written.append(best_path)
    print("\nWrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
