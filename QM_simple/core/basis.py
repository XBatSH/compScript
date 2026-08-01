"""
basis.py -- Minimal STO-3G basis set for the teaching Hartree-Fock code.

Each atomic orbital is a *contracted Gaussian* basis function:

    phi(r) = N * (x-Ax)^l (y-Ay)^m (z-Az)^n * sum_k  c_k * exp(-alpha_k |r-A|^2)

STO-3G means: every Slater-type orbital is approximated by 3 Gaussians.
The exponents (alpha) and contraction coefficients (c) below are the
standard published values (Hehre, Stewart & Pople, JCP 51, 2657 (1969),
as tabulated on the Basis Set Exchange).

Angular momentum is stored as Cartesian powers (l, m, n):
    s  -> (0,0,0)
    px -> (1,0,0),  py -> (0,1,0),  pz -> (0,0,1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def factorial2(n):
    """Double factorial n!! with the convention (-1)!! = 1.

    (scipy.special.factorial2 returns 0 for negative n in recent versions,
    which would break the normalization of s orbitals where l = 0.)
    """
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result


# --------------------------------------------------------------------- #
# STO-3G data:  element -> list of shells.
# Each shell = (shell_type, [exponents], {orbital: [coefficients]})
# For "SP" shells the 2s and 2p functions share exponents (a historical
# trick that made integrals cheaper in the 1970s).
# --------------------------------------------------------------------- #
_S_COEFF_1S = [0.1543289673, 0.5353281423, 0.4446345422]
_S_COEFF_2S = [-0.09996722919, 0.3995128261, 0.7001154689]
_P_COEFF_2P = [0.1559162750, 0.6076837186, 0.3919573931]

STO3G = {
    "H":  [("S", [3.425250914, 0.6239137298, 0.1688554040], _S_COEFF_1S)],
    "He": [("S", [6.362421394, 1.158922999, 0.3136497915], _S_COEFF_1S)],
    "Li": [("S",  [16.11957475, 2.936200663, 0.7946504870], _S_COEFF_1S),
           ("SP", [0.6362897469, 0.1478600533, 0.0480886784], None)],
    "Be": [("S",  [30.16787069, 5.495115306, 1.487192653], _S_COEFF_1S),
           ("SP", [1.314833110, 0.3055389383, 0.0993707456], None)],
    "B":  [("S",  [48.79111318, 8.887362172, 2.405267040], _S_COEFF_1S),
           ("SP", [2.236956142, 0.5198204999, 0.1690617600], None)],
    "C":  [("S",  [71.61683735, 13.04509632, 3.530512160], _S_COEFF_1S),
           ("SP", [2.941249355, 0.6834830964, 0.2222899159], None)],
    "N":  [("S",  [99.10616896, 18.05231239, 4.885660238], _S_COEFF_1S),
           ("SP", [3.780455879, 0.8784966449, 0.2857143744], None)],
    "O":  [("S",  [130.7093214, 23.80886605, 6.443608313], _S_COEFF_1S),
           ("SP", [5.033151319, 1.169596125, 0.3803889600], None)],
    "F":  [("S",  [166.6791340, 30.36081233, 8.216820672], _S_COEFF_1S),
           ("SP", [6.464803249, 1.502281245, 0.4885884864], None)],
}


@dataclass
class BasisFunction:
    """One contracted Cartesian Gaussian basis function."""
    center: np.ndarray          # position A (Bohr)
    lmn: tuple                  # Cartesian powers (l, m, n)
    exps: np.ndarray            # primitive exponents alpha_k
    coefs: np.ndarray           # contraction coefficients c_k (normalized)
    label: str = ""             # human-readable label, e.g. "O1 2px"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.exps = np.asarray(self.exps, dtype=float)
        self.coefs = np.asarray(self.coefs, dtype=float)
        self._normalize()

    def _normalize(self):
        """Normalize primitives, then the whole contraction, so <phi|phi> = 1."""
        l, m, n = self.lmn
        L = l + m + n
        # norm of each primitive Gaussian: standard closed-form expression
        # N_k = [ (2a/pi)^{3/2} (4a)^L / ((2l-1)!!(2m-1)!!(2n-1)!!) ]^{1/2}
        prim_norm = np.sqrt(
            (2 * self.exps / np.pi) ** 1.5 * (4 * self.exps) ** L
            / (factorial2(2 * l - 1) * factorial2(2 * m - 1)
               * factorial2(2 * n - 1))
        )
        self.coefs = self.coefs * prim_norm
        # now scale the contraction so the total function is normalized:
        # <phi|phi> = sum_kl c_k c_l <g_k|g_l>
        pref = (np.pi ** 1.5 * factorial2(2 * l - 1) * factorial2(2 * m - 1)
                * factorial2(2 * n - 1) / 2.0 ** L)
        s = 0.0
        for ck, ak in zip(self.coefs, self.exps):
            for cl, al in zip(self.coefs, self.exps):
                s += ck * cl / (ak + al) ** (L + 1.5)
        self.coefs /= np.sqrt(pref * s)


def build_basis(molecule) -> list[BasisFunction]:
    """Assemble the STO-3G basis-function list for a whole molecule."""
    basis = []
    p_dirs = {"px": (1, 0, 0), "py": (0, 1, 0), "pz": (0, 0, 1)}
    for i, atom in enumerate(molecule.atoms):
        tag = f"{atom.symbol}{i + 1}"
        shell_n = 0                                 # 1s, then 2s/2p, ...
        for shell_type, exps, coefs in STO3G[atom.symbol]:
            shell_n += 1
            if shell_type == "S":
                basis.append(BasisFunction(atom.coord, (0, 0, 0), exps, coefs,
                                           label=f"{tag} {shell_n}s"))
            elif shell_type == "SP":                # shared-exponent 2s + 2p
                basis.append(BasisFunction(atom.coord, (0, 0, 0), exps,
                                           _S_COEFF_2S,
                                           label=f"{tag} {shell_n}s"))
                for pname, lmn in p_dirs.items():
                    basis.append(BasisFunction(atom.coord, lmn, exps,
                                               _P_COEFF_2P,
                                               label=f"{tag} {shell_n}{pname}"))
    return basis
