"""Construct low-energy peptide conformations from the rotamer library.

The workflow implements the user's two-step recipe:

1. **Construct** candidate side-chain conformations by placing rotamers from the
   library (:mod:`rotamer_lib`) onto the peptide backbone.
2. **Minimize** to find the low-energy conformation, using the MMFF energy model
   (:mod:`energy`) both to score rotamers and to relax the final structure.

Selection uses a simple greedy pass (SCWRL-like): each flexible residue is set to
the rotamer that minimizes the whole-molecule energy given the current placement
of the others. A few passes let residues re-optimize against their updated
neighbours. A final backbone-restrained minimization relaxes the side chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import energy as energy_mod
from .peptide import Peptide
from .rotamer_lib import Rotamer, enumerate_rotamers


@dataclass
class RotamerScore:
    resnum: int
    resname: str
    rotamer: Rotamer
    energy: float


@dataclass
class SearchResult:
    peptide: Peptide                       # the optimized peptide (a copy)
    assignments: dict[int, str]            # resnum -> chosen rotamer name
    energy_initial: float                  # energy of the embedded start
    energy_constructed: float              # energy after rotamer placement
    energy_minimized: float                # energy after final minimization
    history: list[RotamerScore] = field(default_factory=list)
    method: str = "greedy"                 # selection method used
    packing_energy: float | None = None    # decomposable LJ energy of the choice


def score_residue_rotamers(
    peptide: Peptide,
    resnum: int,
    max_chi: int = 2,
) -> list[RotamerScore]:
    """Score every library rotamer of one residue by whole-molecule MMFF energy.

    Each rotamer is placed on a copy of the peptide and evaluated as a single
    point. Results are sorted from best (lowest energy) to worst.
    """
    res = peptide.residue(resnum)
    rotamers = enumerate_rotamers(res.name, max_chi=max_chi)
    scores: list[RotamerScore] = []
    for rot in rotamers:
        trial = peptide.copy()
        trial.set_rotamer(resnum, rot)
        e = energy_mod.mmff_energy(trial.mol)
        scores.append(RotamerScore(resnum, res.name, rot, e))
    scores.sort(key=lambda s: s.energy)
    return scores


def build_low_energy_conformation(
    peptide: Peptide,
    max_chi: int = 2,
    n_passes: int = 2,
    minimize_final: bool = True,
    restrain_backbone: bool = True,
    verbose: bool = False,
) -> SearchResult:
    """Place rotamers greedily to build a low-energy conformation, then minimize.

    Parameters
    ----------
    peptide : Peptide
        Starting peptide (not modified; a copy is optimized and returned).
    max_chi : int
        How many chi angles per residue to vary when enumerating rotamers.
    n_passes : int
        Number of greedy sweeps over the flexible residues. More passes let
        residues re-optimize against their updated neighbours.
    minimize_final : bool
        Run a final MMFF minimization after all rotamers are placed.
    restrain_backbone : bool
        Keep the backbone fixed during the final minimization.
    verbose : bool
        Print per-residue choices.

    Returns
    -------
    SearchResult
    """
    work = peptide.copy()
    energy_initial = energy_mod.mmff_energy(work.mol)

    flexible = [res for res in work.residues if res.n_chi > 0]
    assignments: dict[int, str] = {}
    history: list[RotamerScore] = []

    for sweep in range(n_passes):
        for res in flexible:
            rotamers = enumerate_rotamers(res.name, max_chi=max_chi)
            if not rotamers:
                continue
            best: RotamerScore | None = None
            for rot in rotamers:
                work.set_rotamer(res.number, rot)
                e = energy_mod.mmff_energy(work.mol)
                if best is None or e < best.energy:
                    best = RotamerScore(res.number, res.name, rot, e)
            # Commit the best rotamer for this residue.
            work.set_rotamer(res.number, best.rotamer)
            assignments[res.number] = best.rotamer.name
            history.append(best)
            if verbose:
                print(
                    f"  pass {sweep + 1} {res.name}{res.number}: "
                    f"rotamer {best.rotamer.name} -> {best.energy:.2f} kcal/mol"
                )

    energy_constructed = energy_mod.mmff_energy(work.mol)

    energy_minimized = energy_constructed
    if minimize_final:
        result = energy_mod.minimize(
            work, restrain_backbone=restrain_backbone
        )
        energy_minimized = result.energy

    return SearchResult(
        peptide=work,
        assignments=assignments,
        energy_initial=energy_initial,
        energy_constructed=energy_constructed,
        energy_minimized=energy_minimized,
        history=history,
    )
