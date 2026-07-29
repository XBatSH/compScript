"""Combinatorial side-chain optimization: Dead-End Elimination + simulated annealing.

The greedy sweep in :mod:`search` picks one residue at a time and can get stuck in
a local minimum. The rotamer-packing problem is really a **discrete combinatorial
optimization**: choose one rotamer per residue to minimize the total energy

    E(choice) = sum_i  E_self(i, r_i)  +  sum_{i<j}  E_pair(i, r_i; j, r_j)

To make this tractable we use a **decomposable** energy: every term depends on at
most two residues, so it can be precomputed into a self-energy vector and a
pairwise-energy matrix. Two solvers then operate purely on those numbers:

* :func:`dead_end_elimination` - provably prunes rotamers that cannot be part of
  the global minimum (Goldstein criterion).
* :func:`simulated_annealing` - stochastic search over the surviving rotamers.

Energy model
------------
We use a simple Lennard-Jones (12-6) van der Waals energy between atoms, which is
the dominant, cleanly-decomposable signal for steric packing. Bonded (1-2) and
1-3 atom pairs are excluded. This LJ energy drives the *selection*; the final
geometry is still relaxed with the full MMFF force field via :mod:`energy`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from . import energy as energy_mod
from .peptide import Peptide
from .rotamer_lib import Rotamer, enumerate_rotamers
from .search import SearchResult

# Bondi van der Waals radii (Angstrom) and rough LJ well depths (kcal/mol).
VDW_RADIUS = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
}
VDW_EPS = {
    "H": 0.02, "C": 0.07, "N": 0.16, "O": 0.20, "S": 0.45,
    "P": 0.20, "F": 0.08, "CL": 0.28, "BR": 0.32, "I": 0.40,
}
DEFAULT_RADIUS = 1.70
DEFAULT_EPS = 0.10

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
LJ_CUTOFF = 8.0  # Angstrom


# ----------------------------------------------------------------------
# Atom bookkeeping
# ----------------------------------------------------------------------
def _atom_element(atom) -> str:
    return atom.GetSymbol().upper()


def _residue_of_atom(atom):
    """Return (resnum, is_sidechain) for an atom, following H to its heavy neighbor."""
    info = atom.GetPDBResidueInfo()
    if info is None:
        # Hydrogen added by AddHs: inherit from its single heavy neighbor.
        nbrs = atom.GetNeighbors()
        if not nbrs:
            return None, False
        return _residue_of_atom(nbrs[0])
    name = info.GetName().strip()
    return info.GetResidueNumber(), name not in BACKBONE_ATOMS


# ----------------------------------------------------------------------
# The decomposable energy matrix
# ----------------------------------------------------------------------
@dataclass
class RotamerEnergyMatrix:
    """Precomputed self- and pair-energies for all rotamers of every residue."""

    resnums: list[int]                              # flexible residue numbers
    resnames: dict[int, str]                        # resnum -> 3-letter name
    rotamers: dict[int, list[Rotamer]]              # resnum -> rotamer list
    e_self: dict[int, np.ndarray]                   # resnum -> (R_i,)
    e_pair: dict[tuple[int, int], np.ndarray]       # (i<j) -> (R_i, R_j)

    def pair(self, i: int, ri: int, j: int, rj: int) -> float:
        if i < j:
            return float(self.e_pair[(i, j)][ri, rj])
        return float(self.e_pair[(j, i)][rj, ri])

    def total_energy(self, choice: dict[int, int]) -> float:
        e = sum(float(self.e_self[i][choice[i]]) for i in self.resnums)
        for a in range(len(self.resnums)):
            for b in range(a + 1, len(self.resnums)):
                i, j = self.resnums[a], self.resnums[b]
                e += float(self.e_pair[(i, j)][choice[i], choice[j]])
        return e


def _lj(coords_a, rad_a, eps_a, coords_b, rad_b, eps_b, excl=None) -> float:
    """Lennard-Jones (12-6) energy between two atom groups, with a distance cutoff."""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0.0
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    r = np.sqrt(np.einsum("abk,abk->ab", diff, diff))
    rmin = rad_a[:, None] + rad_b[None, :]
    eps = np.sqrt(eps_a[:, None] * eps_b[None, :])
    # Clamp separation so a hard clash gives a large-but-finite penalty.
    r = np.maximum(r, 0.5 * rmin)
    ratio6 = (rmin / r) ** 6
    e = eps * (ratio6 ** 2 - 2.0 * ratio6)
    mask = r < LJ_CUTOFF
    if excl is not None:
        mask &= ~excl
    return float(e[mask].sum())


def build_energy_matrix(
    peptide: Peptide, max_chi: int = 2
) -> RotamerEnergyMatrix:
    """Precompute the self/pair LJ energies for every rotamer of every residue."""
    mol = peptide.mol
    n_atoms = mol.GetNumAtoms()
    topo = Chem.GetDistanceMatrix(mol)          # topological bond distances
    elements = [_atom_element(a) for a in mol.GetAtoms()]
    radius = np.array([VDW_RADIUS.get(e, DEFAULT_RADIUS) for e in elements])
    epsilon = np.array([VDW_EPS.get(e, DEFAULT_EPS) for e in elements])

    # Partition atoms into flexible side chains vs. the fixed template.
    side_atoms: dict[int, list[int]] = {}
    for atom in mol.GetAtoms():
        resnum, is_side = _residue_of_atom(atom)
        if resnum is None or not is_side:
            continue
        side_atoms.setdefault(resnum, []).append(atom.GetIdx())

    flexible = [res for res in peptide.residues if res.n_chi > 0]
    resnums = [r.number for r in flexible if r.number in side_atoms]
    resnames = {r.number: r.name for r in flexible}

    base_conf = mol.GetConformer()
    base_coords = np.array(
        [list(base_conf.GetAtomPosition(i)) for i in range(n_atoms)]
    )
    side_index_set = {idx for idxs in side_atoms.values() for idx in idxs}
    template_idx = np.array(
        [i for i in range(n_atoms) if i not in side_index_set], dtype=int
    )

    # For each residue+rotamer, capture the side-chain coordinates.
    rotamers: dict[int, list[Rotamer]] = {}
    side_coords: dict[int, list[np.ndarray]] = {}
    for resnum in resnums:
        rots = enumerate_rotamers(resnames[resnum], max_chi=max_chi)
        rotamers[resnum] = rots
        idxs = side_atoms[resnum]
        coords_per_rot = []
        for rot in rots:
            trial = peptide.copy()
            trial.set_rotamer(resnum, rot)
            conf = trial.mol.GetConformer()
            coords_per_rot.append(
                np.array([list(conf.GetAtomPosition(k)) for k in idxs])
            )
        side_coords[resnum] = coords_per_rot

    # Self-energy: side chain vs. template + intra-side-chain (excluding 1-2/1-3).
    e_self: dict[int, np.ndarray] = {}
    for resnum in resnums:
        idxs = np.array(side_atoms[resnum], dtype=int)
        excl_tmpl = topo[np.ix_(idxs, template_idx)] <= 2
        excl_intra = topo[np.ix_(idxs, idxs)] <= 2
        vals = []
        for coords in side_coords[resnum]:
            e = _lj(
                coords, radius[idxs], epsilon[idxs],
                base_coords[template_idx], radius[template_idx], epsilon[template_idx],
                excl=excl_tmpl,
            )
            e += _lj(
                coords, radius[idxs], epsilon[idxs],
                coords, radius[idxs], epsilon[idxs],
                excl=excl_intra,
            ) * 0.5  # each intra pair counted twice
            vals.append(e)
        e_self[resnum] = np.array(vals)

    # Pair-energy: side chain i (rotamer r) vs. side chain j (rotamer s).
    e_pair: dict[tuple[int, int], np.ndarray] = {}
    for a in range(len(resnums)):
        for b in range(a + 1, len(resnums)):
            i, j = resnums[a], resnums[b]
            idx_i = np.array(side_atoms[i], dtype=int)
            idx_j = np.array(side_atoms[j], dtype=int)
            excl = topo[np.ix_(idx_i, idx_j)] <= 2
            mat = np.zeros((len(rotamers[i]), len(rotamers[j])))
            for ri, ci in enumerate(side_coords[i]):
                for rj, cj in enumerate(side_coords[j]):
                    mat[ri, rj] = _lj(
                        ci, radius[idx_i], epsilon[idx_i],
                        cj, radius[idx_j], epsilon[idx_j],
                        excl=excl,
                    )
            e_pair[(i, j)] = mat

    return RotamerEnergyMatrix(
        resnums=resnums, resnames=resnames, rotamers=rotamers,
        e_self=e_self, e_pair=e_pair,
    )


# ----------------------------------------------------------------------
# Dead-End Elimination (Goldstein criterion)
# ----------------------------------------------------------------------
def dead_end_elimination(
    matrix: RotamerEnergyMatrix, verbose: bool = False
) -> dict[int, list[int]]:
    """Prune rotamers that cannot be in the global minimum (Goldstein DEE).

    Rotamer r at residue i is eliminated if some alternative t is always at least
    as good, for every combination of the other residues' (still-allowed) rotamers:

        E_self(i,r) - E_self(i,t)
            + sum_{j != i} min_s [ E_pair(i,r; j,s) - E_pair(i,t; j,s) ]  > 0

    Returns the surviving rotamer indices per residue.
    """
    allowed = {i: list(range(len(matrix.rotamers[i]))) for i in matrix.resnums}

    changed = True
    while changed:
        changed = False
        for i in matrix.resnums:
            if len(allowed[i]) <= 1:
                continue
            survivors = list(allowed[i])
            for r in list(survivors):
                if len(survivors) <= 1:
                    break
                for t in survivors:
                    if t == r:
                        continue
                    delta = matrix.e_self[i][r] - matrix.e_self[i][t]
                    for j in matrix.resnums:
                        if j == i:
                            continue
                        diffs = [
                            matrix.pair(i, r, j, s) - matrix.pair(i, t, j, s)
                            for s in allowed[j]
                        ]
                        delta += min(diffs)
                    if delta > 1e-9:  # r is dominated by t -> eliminate r
                        survivors.remove(r)
                        changed = True
                        if verbose:
                            print(f"    DEE: eliminate {matrix.resnames[i]}{i} "
                                  f"rotamer #{r} (dominated by #{t})")
                        break
            allowed[i] = survivors

    if verbose:
        remaining = {matrix.resnames[i] + str(i): len(allowed[i])
                     for i in matrix.resnums}
        print(f"    DEE survivors per residue: {remaining}")
    return allowed


# ----------------------------------------------------------------------
# Simulated annealing
# ----------------------------------------------------------------------
def simulated_annealing(
    matrix: RotamerEnergyMatrix,
    candidates: dict[int, list[int]] | None = None,
    n_steps: int = 4000,
    t_start: float = 5.0,
    t_end: float = 0.05,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[dict[int, int], float]:
    """Minimize the total decomposable energy by simulated annealing.

    Parameters
    ----------
    candidates : optional
        Allowed rotamer indices per residue (e.g. the DEE survivors). Defaults to
        all rotamers.

    Returns
    -------
    (choice, energy) : the best rotamer assignment and its decomposable energy.
    """
    rng = random.Random(seed)
    if candidates is None:
        candidates = {i: list(range(len(matrix.rotamers[i]))) for i in matrix.resnums}

    choice = {i: rng.choice(candidates[i]) for i in matrix.resnums}
    energy = matrix.total_energy(choice)
    best_choice, best_energy = dict(choice), energy

    cooling = (t_end / t_start) ** (1.0 / max(1, n_steps))
    temp = t_start
    for _ in range(n_steps):
        i = rng.choice(matrix.resnums)
        options = candidates[i]
        if len(options) <= 1:
            temp *= cooling
            continue
        new_r = rng.choice(options)
        if new_r == choice[i]:
            temp *= cooling
            continue
        old_r = choice[i]
        # Energy delta for reassigning residue i only.
        d = matrix.e_self[i][new_r] - matrix.e_self[i][old_r]
        for j in matrix.resnums:
            if j == i:
                continue
            d += matrix.pair(i, new_r, j, choice[j]) - matrix.pair(i, old_r, j, choice[j])
        if d < 0 or rng.random() < math.exp(-d / max(temp, 1e-9)):
            choice[i] = new_r
            energy += d
            if energy < best_energy:
                best_energy, best_choice = energy, dict(choice)
        temp *= cooling

    if verbose:
        print(f"    SA best packing energy: {best_energy:.2f}")
    return best_choice, best_energy


def _greedy_on_candidates(
    matrix: RotamerEnergyMatrix, candidates: dict[int, list[int]]
) -> dict[int, int]:
    """Pick, per residue, the rotamer with lowest self+pair energy (few sweeps)."""
    choice = {i: candidates[i][0] for i in matrix.resnums}
    for _ in range(2):
        for i in matrix.resnums:
            best_r, best_e = choice[i], math.inf
            for r in candidates[i]:
                e = matrix.e_self[i][r]
                for j in matrix.resnums:
                    if j != i:
                        e += matrix.pair(i, r, j, choice[j])
                if e < best_e:
                    best_e, best_r = e, r
            choice[i] = best_r
    return choice


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def solve_rotamers(
    peptide: Peptide,
    method: str = "dee+sa",
    max_chi: int = 2,
    minimize_final: bool = True,
    restrain_backbone: bool = True,
    sa_steps: int = 4000,
    seed: int = 0,
    verbose: bool = False,
) -> SearchResult:
    """Solve side-chain packing with DEE and/or simulated annealing.

    Parameters
    ----------
    method : {"dee+sa", "sa", "dee"}
        Selection strategy. ``dee+sa`` prunes with DEE then anneals over the
        survivors; ``sa`` anneals over all rotamers; ``dee`` prunes then picks
        greedily among survivors.
    """
    if verbose:
        print(f"  building energy matrix (max_chi={max_chi}) ...")
    matrix = build_energy_matrix(peptide, max_chi=max_chi)

    candidates: dict[int, list[int]] | None = None
    if "dee" in method:
        candidates = dead_end_elimination(matrix, verbose=verbose)

    if "sa" in method:
        choice, packing = simulated_annealing(
            matrix, candidates=candidates, n_steps=sa_steps,
            seed=seed, verbose=verbose,
        )
    else:
        cand = candidates or {
            i: list(range(len(matrix.rotamers[i]))) for i in matrix.resnums
        }
        choice = _greedy_on_candidates(matrix, cand)
        packing = matrix.total_energy(choice)

    # Realize the chosen rotamers on a fresh peptide copy.
    work = peptide.copy()
    energy_initial = energy_mod.mmff_energy(work.mol)
    assignments: dict[int, str] = {}
    for i in matrix.resnums:
        rot = matrix.rotamers[i][choice[i]]
        work.set_rotamer(i, rot)
        assignments[i] = rot.name
    energy_constructed = energy_mod.mmff_energy(work.mol)

    energy_minimized = energy_constructed
    if minimize_final:
        energy_minimized = energy_mod.minimize(
            work, restrain_backbone=restrain_backbone
        ).energy

    return SearchResult(
        peptide=work,
        assignments=assignments,
        energy_initial=energy_initial,
        energy_constructed=energy_constructed,
        energy_minimized=energy_minimized,
        method=method,
        packing_energy=packing,
    )
