"""
scf.py -- Restricted Hartree-Fock (RHF) self-consistent-field procedure.

The algorithm (Szabo & Ostlund, "Modern Quantum Chemistry", section 3.4.6):

  1. Compute integrals S, T, V, (uv|ls); form the core Hamiltonian
         H_core = T + V
  2. Build the orthogonalizer X = S^{-1/2} (symmetric / Loewdin).
  3. Initial guess: diagonalize H_core (no electron-electron repulsion).
  4. Iterate:
       a. Build density  D_uv = 2 sum_i^{occ} C_ui C_vi
       b. Build Fock     F = H_core + G(D)
          G_uv = sum_ls D_ls [ (uv|ls) - 1/2 (ul|vs) ]   (Coulomb - exchange)
       c. Solve FC = SCe by transforming to F' = X^T F X, diagonalizing,
          and back-transforming C = X C'.
       d. Electronic energy  E = 1/2 sum_uv D_uv (H_core + F)_uv
       e. Check convergence on the energy and density change.
  5. E_total = E_electronic + E_nuclear_repulsion.

Only closed-shell molecules (even electron count) are supported: each
occupied spatial orbital holds exactly 2 electrons.
"""

from __future__ import annotations

import time

import numpy as np

from .basis import build_basis
from .integrals import build_one_electron, build_eri


def rhf(molecule, max_iter=100, e_conv=1e-8, d_conv=1e-6, verbose=True):
    """Run a restricted Hartree-Fock calculation.

    Returns a dict with energies, MO coefficients, orbital energies, etc.
    """
    log = print if verbose else (lambda *a, **k: None)

    n_elec = molecule.n_electrons
    if n_elec % 2 != 0:
        raise ValueError("RHF needs an even number of electrons "
                         f"(got {n_elec}). Use a closed-shell molecule/ion.")
    n_occ = n_elec // 2                       # doubly-occupied orbitals

    log("=" * 64)
    log(str(molecule))
    log("=" * 64)

    # ---- Step 1: integrals -------------------------------------------- #
    basis = build_basis(molecule)
    nbf = len(basis)
    log(f"\nSTO-3G basis: {nbf} basis functions, "
        f"{n_occ} doubly occupied orbitals")

    t0 = time.perf_counter()
    S, T, V = build_one_electron(basis, molecule)
    H_core = T + V
    eri = build_eri(basis, verbose=verbose)
    log(f"  integral evaluation took {time.perf_counter() - t0:.2f} s")

    E_nn = molecule.nuclear_repulsion()
    log(f"\nNuclear repulsion energy: {E_nn:.10f} Hartree")

    # ---- Step 2: orthogonalizer X = S^(-1/2) -------------------------- #
    # Diagonalize S = U s U^T, then X = U s^{-1/2} U^T.
    s_val, U = np.linalg.eigh(S)
    if s_val.min() < 1e-7:
        log("  warning: near-linear dependence in the basis "
            f"(smallest S eigenvalue {s_val.min():.2e})")
    X = U @ np.diag(s_val ** -0.5) @ U.T

    # ---- Step 3: core-Hamiltonian guess (D = 0) ----------------------- #
    def solve_roothaan(F):
        """Solve FC = SCe via the orthogonalized eigenproblem."""
        Fp = X.T @ F @ X                      # F' in orthonormal basis
        eps, Cp = np.linalg.eigh(Fp)          # ordinary eigenproblem
        C = X @ Cp                            # back to the AO basis
        return eps, C

    eps, C = solve_roothaan(H_core)
    D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T   # density matrix

    # ---- Step 4: SCF iterations --------------------------------------- #
    log(f"\n{'iter':>4} {'E(elec)/Ha':>18} {'E(total)/Ha':>18} "
        f"{'dE':>10} {'d(D)':>10}")

    E_old = 0.0
    for it in range(1, max_iter + 1):
        # G(D): Coulomb J minus half exchange K, contracted with density.
        # einsum spells out:  J_uv = (uv|ls) D_ls ,  K_uv = (ul|vs) D_ls
        J = np.einsum("uvls,ls->uv", eri, D)
        K = np.einsum("ulvs,ls->uv", eri, D)
        F = H_core + J - 0.5 * K

        # electronic energy from the CURRENT density and Fock matrix
        E_elec = 0.5 * np.sum(D * (H_core + F))

        eps, C = solve_roothaan(F)
        D_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T

        dE = E_elec - E_old
        dD = np.max(np.abs(D_new - D))
        log(f"{it:4d} {E_elec:18.10f} {E_elec + E_nn:18.10f} "
            f"{dE:10.2e} {dD:10.2e}")

        D = D_new
        E_old = E_elec
        if abs(dE) < e_conv and dD < d_conv:
            log(f"\nSCF converged in {it} iterations!")
            break
    else:
        raise RuntimeError(f"SCF did not converge in {max_iter} iterations")

    E_total = E_elec + E_nn

    # ---- Report -------------------------------------------------------- #
    log(f"\nFinal RHF energy:   {E_total:.10f} Hartree")
    log(f"  electronic part:  {E_elec:.10f}")
    log(f"  nuclear repulsion:{E_nn:.10f}")

    log("\nMolecular orbital energies (Hartree):")
    homo, lumo = None, None
    for i, e in enumerate(eps):
        occ = "occupied (2e)" if i < n_occ else "virtual"
        marker = ""
        if i == n_occ - 1:
            marker, homo = "  <-- HOMO", e
        elif i == n_occ:
            marker, lumo = "  <-- LUMO", e
        log(f"  MO {i + 1:2d}: {e:12.6f}  {occ}{marker}")
    if homo is not None and lumo is not None:
        log(f"HOMO-LUMO gap: {(lumo - homo) * 27.211386:.2f} eV")

    # Mulliken population analysis: q_A = Z_A - sum_{u in A} (D S)_uu
    log("\nMulliken atomic charges:")
    DS_diag = np.diag(D @ S)
    charges = []
    for i, atom in enumerate(molecule.atoms):
        pop = sum(DS_diag[k] for k, bf in enumerate(basis)
                  if bf.label.startswith(f"{atom.symbol}{i + 1} "))
        q = atom.Z - pop
        charges.append(q)
        log(f"  {atom.symbol}{i + 1}: {q:+.4f}")

    return {
        "energy": E_total,
        "energy_elec": E_elec,
        "energy_nuc": E_nn,
        "mo_energies": eps,
        "mo_coefficients": C,
        "density": D,
        "overlap": S,
        "basis": basis,
        "mulliken_charges": np.array(charges),
        "n_iterations": it,
    }
