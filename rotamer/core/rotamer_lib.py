"""A compact, backbone-independent side-chain rotamer library.

Real packing tools (SCWRL4, Dunbrack's libraries) use large statistical tables
of chi means and frequencies that also depend on the backbone (phi, psi). For a
learning module we use the classic **staggered rotamer approximation**: every
rotatable chi angle prefers one of three staggered states around the tetrahedral
bond,

    p  ("plus",  gauche+)  ~  +60 deg
    t  ("trans")           ~  180 deg
    m  ("minus", gauche-)  ~  -60 deg

A rotamer is then a combination of per-chi states, e.g. Lys ``"mt"`` means
chi1 = -60, chi2 = 180. This reproduces the dominant rotamers well; the final
geometry is always relaxed by energy minimization afterwards, which absorbs the
difference between these idealized means and the true statistical means.

The number of combinations grows as ``3**n_chi``, so by default we only enumerate
the first ``max_chi`` angles (chi1, chi2 dominate side-chain identity) and leave the
deeper angles in the trans state.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .residues import n_chi

# The three staggered chi states and their representative angles (degrees).
CHI_STATES: dict[str, float] = {"p": 60.0, "t": 180.0, "m": -60.0}
DEFAULT_STATE = "t"


@dataclass(frozen=True)
class Rotamer:
    """One side-chain rotamer.

    Attributes
    ----------
    name : str
        State string, one letter per enumerated chi (e.g. ``"mt"``).
    chi : tuple[float, ...]
        Full chi angle values in degrees, length == number of chi angles of the
        residue. Angles beyond ``max_chi`` are held at the default (trans) state.
    """

    name: str
    chi: tuple[float, ...]


def enumerate_rotamers(resname: str, max_chi: int = 2) -> list[Rotamer]:
    """Enumerate staggered rotamers for a residue.

    Parameters
    ----------
    resname : str
        3-letter residue name (e.g. ``"LYS"``).
    max_chi : int
        Maximum number of chi angles to vary. Deeper chi angles are fixed at the
        default trans state. Use a large value to enumerate all chi angles.

    Returns
    -------
    list[Rotamer]
        All state combinations for the varied chi angles. Returns an empty list
        for residues with no rotatable side chain (ALA, GLY, PRO, unknown).
    """
    total = n_chi(resname)
    if total == 0:
        return []

    n_vary = min(max_chi, total)
    n_fixed = total - n_vary
    letters = list(CHI_STATES.keys())

    rotamers: list[Rotamer] = []
    for combo in itertools.product(letters, repeat=n_vary):
        chi = [CHI_STATES[s] for s in combo]
        chi.extend(CHI_STATES[DEFAULT_STATE] for _ in range(n_fixed))
        rotamers.append(Rotamer(name="".join(combo), chi=tuple(chi)))
    return rotamers
