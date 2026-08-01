# A Hands-On Tutorial: Hartree–Fock from Scratch in Python

*A companion document for the `core` teaching code
(pure Python + numpy + scipy + RDKit).*

This tutorial explains **every step** of a Hartree–Fock (HF) calculation and
shows how each formula maps onto a short piece of Python.  All code snippets
are taken (sometimes lightly shortened) from the actual program in
[`core/`](../core), so you can always jump from the text to the real,
runnable source.

**Contents**

1. [What problem are we solving?](#1-what-problem-are-we-solving)
2. [The Hartree–Fock approximation](#2-the-hartree–fock-approximation)
3. [Basis sets: STO-3G](#3-basis-sets-sto-3g)
4. [Molecular integrals](#4-molecular-integrals)
5. [The SCF procedure](#5-the-scf-procedure)
6. [Reading the results](#6-reading-the-results)
7. [Case study: breaking H₂](#7-case-study-breaking-h)
8. [Exercises](#8-exercises)
9. [References](#9-references)

---

## 1. What problem are we solving?

Chemistry is, at the bottom, the quantum mechanics of electrons moving around
nuclei.  For a molecule with $N$ electrons and $M$ nuclei we want solutions of
the time-independent Schrödinger equation

$$
\hat{H}\,\Psi = E\,\Psi .
$$

Two standard simplifications get us to something computable:

**(a) Born–Oppenheimer approximation.**  Nuclei are thousands of times heavier
than electrons, so we clamp them at fixed positions $\mathbf{R}_A$ and solve
only for the electrons.  The electronic Hamiltonian (in **atomic units**:
$\hbar = m_e = e = 4\pi\varepsilon_0 = 1$; energies in Hartree, lengths in
Bohr) is

$$
\hat{H}_{el}
= \underbrace{-\sum_{i=1}^{N} \tfrac{1}{2}\nabla_i^2}_{\text{kinetic}}
\; \underbrace{-\sum_{i=1}^{N}\sum_{A=1}^{M} \frac{Z_A}{r_{iA}}}_{\text{electron–nucleus attraction}}
\; + \underbrace{\sum_{i<j} \frac{1}{r_{ij}}}_{\text{electron–electron repulsion}} .
$$

The nuclei themselves contribute a trivial constant that we add at the end:

$$
E_{nn} = \sum_{A<B} \frac{Z_A Z_B}{R_{AB}} .
$$

In the code this is three lines
([`molecule.py`](../core/molecule.py)):

```python
def nuclear_repulsion(self) -> float:
    """E_nn = sum_{A<B} Z_A Z_B / |R_A - R_B|   (atomic units)."""
    e_nn = 0.0
    for i, a in enumerate(self.atoms):
        for b in self.atoms[i + 1:]:
            e_nn += a.Z * b.Z / np.linalg.norm(a.coord - b.coord)
    return e_nn
```

**(b) Where do the nuclear positions come from?**  For teaching we either type
textbook geometries by hand, or we let **RDKit** build a reasonable 3D
structure from a SMILES string (distance-geometry embedding + MMFF94 force
field):

```python
mol = Chem.AddHs(Chem.MolFromSmiles(smiles))    # SMILES hides hydrogens!
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())   # generate 3D coordinates
AllChem.MMFFOptimizeMolecule(mol)               # force-field refinement
```

Even after Born–Oppenheimer, the electronic problem is unsolvable exactly for
more than one electron, because the $1/r_{ij}$ term couples all electrons
together.  Hartree–Fock is the classic, systematic way to approximate it.

---

## 2. The Hartree–Fock approximation

### 2.1 One determinant, mean field

HF makes a single structural assumption: the $N$-electron wavefunction is one
**Slater determinant** of orthonormal spin-orbitals $\chi_i$,

$$
\Psi_{HF}(\mathbf{x}_1,\dots,\mathbf{x}_N) =
\frac{1}{\sqrt{N!}}
\begin{vmatrix}
\chi_1(\mathbf{x}_1) & \cdots & \chi_N(\mathbf{x}_1) \\
\vdots               &        & \vdots \\
\chi_1(\mathbf{x}_N) & \cdots & \chi_N(\mathbf{x}_N)
\end{vmatrix},
$$

which automatically satisfies the Pauli principle (swap two rows → sign
flips).  Minimizing $\langle\Psi|\hat H|\Psi\rangle$ with respect to the
orbitals (variational principle) yields the **Fock equations**: each electron
moves in the *average* field of all the others,

$$
\hat{f}\,\chi_i = \varepsilon_i\,\chi_i,
\qquad
\hat{f} = -\tfrac12\nabla^2 - \sum_A \frac{Z_A}{r_A} + \hat{J} - \hat{K},
$$

where $\hat J$ is the classical Coulomb repulsion of the electron cloud and
$\hat K$ is the purely quantum **exchange** operator.  Because $\hat J$ and
$\hat K$ depend on the orbitals we are trying to find, the equations must be
solved **iteratively** — hence "self-consistent field" (SCF).

### 2.2 Restricted HF and the Roothaan equations

For a **closed-shell** molecule (all electrons paired — the only case
`core` handles) every spatial orbital $\psi_i$ holds two electrons.
Expanding each molecular orbital (MO) in a finite set of $K$ known **basis
functions** $\phi_\mu$ (Section 3),

$$
\psi_i(\mathbf r) = \sum_{\mu=1}^{K} C_{\mu i}\,\phi_\mu(\mathbf r),
$$

turns the differential Fock equations into a matrix problem, the
**Roothaan–Hall equations**:

$$
\boxed{\;\mathbf{F}\,\mathbf{C} = \mathbf{S}\,\mathbf{C}\,\boldsymbol{\varepsilon}\;}
$$

- $\mathbf S$: overlap matrix, $S_{\mu\nu} = \langle\phi_\mu|\phi_\nu\rangle$
  (basis functions are *not* orthogonal!),
- $\mathbf F$: Fock matrix (depends on $\mathbf C$ — that is the
  self-consistency),
- $\boldsymbol\varepsilon$: diagonal matrix of orbital energies.

With the **density matrix**

$$
D_{\mu\nu} = 2\sum_{i}^{N/2} C_{\mu i} C_{\nu i}
$$

the Fock matrix takes the form that appears verbatim in the code:

$$
F_{\mu\nu} = H^{core}_{\mu\nu}
 + \underbrace{\sum_{\lambda\sigma} D_{\lambda\sigma}\,(\mu\nu|\lambda\sigma)}_{J,\ \text{Coulomb}}
 - \tfrac12 \underbrace{\sum_{\lambda\sigma} D_{\lambda\sigma}\,(\mu\lambda|\nu\sigma)}_{K,\ \text{exchange}} ,
$$

and the electronic energy is

$$
E_{el} = \tfrac12 \sum_{\mu\nu} D_{\mu\nu}\left(H^{core}_{\mu\nu} + F_{\mu\nu}\right),
\qquad
E_{tot} = E_{el} + E_{nn}.
$$

So the whole calculation reduces to: **compute a handful of integral arrays,
then iterate a small linear-algebra loop.**

---

## 3. Basis sets: STO-3G

### 3.1 Why Gaussians?

Physically, atomic orbitals decay like Slater functions $e^{-\zeta r}$.  But
integrals over Slater functions on *different* centers are extremely hard.
Boys (1950) noticed that **Gaussians** $e^{-\alpha r^2}$ make every integral
analytic.  The price: one Gaussian is a poor imitation of a Slater orbital
(wrong shape at the nucleus and in the tail).  The fix: use a **contraction**,
a fixed linear combination of several Gaussians.

**STO-3G** approximates each Slater orbital by 3 Gaussians:

$$
\phi^{STO\text{-}3G}(\mathbf r)
= \sum_{k=1}^{3} c_k\,N_k\, x^l y^m z^n\, e^{-\alpha_k |\mathbf r - \mathbf A|^2}.
$$

![STO-3G contraction](figures/sto3g_contraction.png)

*Three Gaussians (dashed) sum to the blue curve, which closely tracks the
exact Slater orbital (gray) — except at the nucleus, where a Gaussian has zero
slope but the true orbital has a cusp.*

The Cartesian powers $(l,m,n)$ encode angular momentum: $s = (0,0,0)$,
$p_x = (1,0,0)$, etc.  In [`basis.py`](../core/basis.py) each basis
function is a tiny dataclass:

```python
@dataclass
class BasisFunction:
    center: np.ndarray          # position A (Bohr)
    lmn: tuple                  # Cartesian powers (l, m, n)
    exps: np.ndarray            # primitive exponents alpha_k
    coefs: np.ndarray           # contraction coefficients c_k (normalized)
```

and the published STO-3G parameters are just a table:

```python
STO3G = {
    "H":  [("S", [3.425250914, 0.6239137298, 0.1688554040], _S_COEFF_1S)],
    "O":  [("S",  [130.7093214, 23.80886605, 6.443608313], _S_COEFF_1S),
           ("SP", [5.033151319, 1.169596125, 0.3803889600], None)],
    ...
}
```

("SP" means the 2s and 2p orbitals share exponents — a 1969 trick that made
integral evaluation cheaper.)

### 3.2 Normalization

Each primitive must be normalized, and then the whole contraction rescaled so
that $\langle\phi|\phi\rangle = 1$.  The closed-form primitive norm is

$$
N_k = \left[
\left(\frac{2\alpha_k}{\pi}\right)^{3/2}
\frac{(4\alpha_k)^{l+m+n}}{(2l-1)!!\,(2m-1)!!\,(2n-1)!!}
\right]^{1/2}.
$$

> ⚠️ **A real-world bug we hit:** the double factorial must obey the
> convention $(-1)!! = 1$ for $s$ orbitals ($l=0$).  Recent SciPy versions
> changed `scipy.special.factorial2(-1)` to return **0**, which silently
> produced `NaN` energies.  `core` therefore ships its own 5-line
> `factorial2`.  Moral for students: *never trust library corner cases.*

---

## 4. Molecular integrals

This is the mathematically heaviest part of any quantum chemistry program.
We need four arrays, all computed **once** before the SCF loop:

| symbol | meaning | size |
|---|---|---|
| $S_{\mu\nu}$ | overlap $\langle\mu\vert\nu\rangle$ | $K\times K$ |
| $T_{\mu\nu}$ | kinetic energy $\langle\mu\vert{-\tfrac12\nabla^2}\vert\nu\rangle$ | $K\times K$ |
| $V_{\mu\nu}$ | nuclear attraction $\langle\mu\vert{-\sum_A Z_A/r_A}\vert\nu\rangle$ | $K\times K$ |
| $(\mu\nu\vert\lambda\sigma)$ | electron–electron repulsion (ERI) | $K^4$ |

### 4.1 The Gaussian product theorem

Everything rests on one beautiful fact: *the product of two Gaussians on
different centers is a single Gaussian on an intermediate center.*

$$
e^{-\alpha|\mathbf r-\mathbf A|^2}\, e^{-\beta|\mathbf r-\mathbf B|^2}
= \underbrace{e^{-\frac{\alpha\beta}{p}|\mathbf A-\mathbf B|^2}}_{K_{AB}}
\; e^{-p\,|\mathbf r - \mathbf P|^2},
\qquad
p = \alpha+\beta,\quad
\mathbf P = \frac{\alpha\mathbf A + \beta\mathbf B}{p}.
$$

![Gaussian product theorem](figures/gaussian_product.png)

*The red curve (product of blue and orange) is exactly a Gaussian centered at
the weighted midpoint P — this is what makes 2-, 3-, and 4-center integrals
tractable.*

### 4.2 Hermite expansion (McMurchie–Davidson)

To handle $p$ and $d$ functions systematically, the product of two Cartesian
Gaussians is expanded in **Hermite Gaussians** $\Lambda_t$, one Cartesian
direction at a time:

$$
G_i(x;\alpha,A)\,G_j(x;\beta,B) = \sum_{t=0}^{i+j} E_t^{ij}\,\Lambda_t(x;p,P).
$$

The coefficients $E_t^{ij}$ obey a 3-term recursion, which in
[`integrals.py`](../core/integrals.py) is a direct transcription:

```python
def E(i, j, t, Qx, a, b):
    p, q = a + b, a * b / (a + b)
    if t < 0 or t > i + j:
        return 0.0                       # out of bounds
    if i == j == t == 0:
        return np.exp(-q * Qx * Qx)      # Gaussian product prefactor K_AB
    if j == 0:                           # decrement index i
        return (E(i-1, j, t-1, Qx, a, b) / (2*p)
                - q * Qx / a * E(i-1, j, t, Qx, a, b)
                + (t+1) * E(i-1, j, t+1, Qx, a, b))
    return (E(i, j-1, t-1, Qx, a, b) / (2*p)          # decrement index j
            + q * Qx / b * E(i, j-1, t, Qx, a, b)
            + (t+1) * E(i, j-1, t+1, Qx, a, b))
```

**Overlap** then falls out immediately — only the $t=0$ term survives the
integration:

$$
S_{ab} = E_0^{l_1 l_2} E_0^{m_1 m_2} E_0^{n_1 n_2}
\left(\frac{\pi}{p}\right)^{3/2}.
$$

```python
def _overlap_prim(a, lmn1, A, b, lmn2, B):
    s_x = E(l1, l2, 0, A[0] - B[0], a, b)
    s_y = E(m1, m2, 0, A[1] - B[1], a, b)
    s_z = E(n1, n2, 0, A[2] - B[2], a, b)
    return s_x * s_y * s_z * (np.pi / (a + b)) ** 1.5
```

**Kinetic energy** needs no new machinery: differentiating a Gaussian twice
just shifts its Cartesian power by $\pm 2$, so $T$ is a linear combination of
overlaps:

$$
T_{ab} = \beta\,(2(l_2+m_2+n_2)+3)\,S_{ab}
- 2\beta^2\left(S_{ab}^{(l_2+2)} + S_{ab}^{(m_2+2)} + S_{ab}^{(n_2+2)}\right)
- \tfrac12\left(l_2(l_2-1)S_{ab}^{(l_2-2)} + \dots\right).
$$

### 4.3 Coulomb integrals and the Boys function

Integrals containing $1/r$ have no closed form in elementary functions.  The
standard trick is the identity

$$
\frac{1}{r} = \frac{2}{\sqrt{\pi}} \int_0^\infty e^{-r^2 u^2}\, du ,
$$

which converts the Coulomb kernel into *yet another Gaussian*.  After the dust
settles, every Coulomb-type integral reduces to the **Boys function**

$$
F_n(x) = \int_0^1 t^{2n}\, e^{-x t^2}\, dt ,
$$

which scipy evaluates for us through the confluent hypergeometric function:

$$
F_n(x) = \frac{{}_1F_1\!\left(n+\tfrac12;\, n+\tfrac32;\, -x\right)}{2n+1}
\qquad\Longrightarrow\qquad
\texttt{hyp1f1(n + 0.5, n + 1.5, -x) / (2 * n + 1)}
$$

![Boys function](figures/boys_function.png)

*$F_0(0) = 1$ and $F_0(x)\to\frac12\sqrt{\pi/x}$ for large $x$ — the
long-range Coulomb tail.*

Derivatives of $F_n$ needed for higher angular momenta are organized in the
**Hermite Coulomb integrals** $R_{tuv}$, again a short recursion in the code
(`R(t, u, v, n, ...)`).  With these, nuclear attraction is

$$
V_{ab}^{(C)} = \frac{2\pi}{p} \sum_{tuv}
E_t^{l_1l_2} E_u^{m_1m_2} E_v^{n_1n_2}\; R_{tuv}(p, \mathbf P - \mathbf C),
$$

and the 4-center ERI couples *two* Hermite expansions (bra pair and ket pair):

$$
(ab|cd) = \frac{2\pi^{5/2}}{pq\sqrt{p+q}}
\sum_{tuv}\sum_{\tau\nu\varphi}
E^{ab}_{tuv}\, E^{cd}_{\tau\nu\varphi}\, (-1)^{\tau+\nu+\varphi}\,
R_{t+\tau,\,u+\nu,\,v+\varphi}\!\left(\frac{pq}{p+q},\, \mathbf P - \mathbf Q\right).
$$

### 4.4 Permutation symmetry — the first "real" optimization

The ERI tensor has an 8-fold symmetry:

$$
(\mu\nu|\lambda\sigma) = (\nu\mu|\lambda\sigma) = (\mu\nu|\sigma\lambda)
= (\lambda\sigma|\mu\nu) = \dots
$$

so we compute only unique quadruplets and copy the value 8 times:

```python
for i in range(n):
    for j in range(i + 1):
        ij = i * (i + 1) // 2 + j          # compound pair index
        for k in range(n):
            for l in range(k + 1):
                kl = k * (k + 1) // 2 + l
                if ij < kl:
                    continue               # filled later by symmetry
                val = electron_repulsion(basis[i], basis[j],
                                         basis[k], basis[l])
```

For water (7 basis functions) this means **406 integrals instead of 2401** —
students see immediately why production codes obsess over integral screening.

---

## 5. The SCF procedure

Now the physics is done; what remains is a compact linear-algebra loop
([`scf.py`](../core/scf.py)), summarized in one picture:

![SCF flowchart](figures/scf_flowchart.png)

### 5.1 Orthogonalization: $X = S^{-1/2}$

$\mathbf F\mathbf C = \mathbf S\mathbf C\boldsymbol\varepsilon$ is a
*generalized* eigenvalue problem because the basis is not orthogonal.  Löwdin
symmetric orthogonalization fixes that: diagonalize
$\mathbf S = \mathbf U\,\mathbf s\,\mathbf U^{T}$ and form

$$
\mathbf X = \mathbf U\, \mathbf s^{-1/2}\, \mathbf U^{T},
\qquad
\mathbf F' = \mathbf X^{T}\mathbf F\mathbf X,
\qquad
\mathbf F'\mathbf C' = \mathbf C'\boldsymbol\varepsilon,
\qquad
\mathbf C = \mathbf X\mathbf C' .
$$

```python
s_val, U = np.linalg.eigh(S)
X = U @ np.diag(s_val ** -0.5) @ U.T

def solve_roothaan(F):
    Fp = X.T @ F @ X                      # F' in orthonormal basis
    eps, Cp = np.linalg.eigh(Fp)          # ordinary eigenproblem
    return eps, X @ Cp                    # back to the AO basis
```

### 5.2 The iteration

The initial guess simply ignores electron–electron repulsion (diagonalize
$H^{core}$).  Then each cycle is four lines of numpy — note how literally
`einsum` spells out the tensor contractions from Section 2.2:

```python
J = np.einsum("uvls,ls->uv", eri, D)      # J_uv = (uv|ls) D_ls
K = np.einsum("ulvs,ls->uv", eri, D)      # K_uv = (ul|vs) D_ls
F = H_core + J - 0.5 * K

E_elec = 0.5 * np.sum(D * (H_core + F))   # current electronic energy

eps, C = solve_roothaan(F)
D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T   # new density
```

Convergence is declared when both the energy change and the largest density
matrix change are tiny ($|\Delta E| < 10^{-8}$ Ha, $|\Delta D| < 10^{-6}$).
Watching the iterations is instructive — the energy drops *monotonically* and
roughly geometrically (each cycle gains about one digit):

```
iter         E(elec)/Ha        E(total)/Ha         dE       d(D)
   1     -82.4207855367     -73.2327803510  -8.24e+01   1.75e+00
   2     -84.1337551682     -74.9457499825  -1.71e+00   1.48e-01
   3     -84.1502000344     -74.9621948487  -1.64e-02   4.38e-02
   ...
  16     -84.1510578511     -74.9630526653  -8.81e-13   4.63e-07
SCF converged in 16 iterations!
```

### 5.3 Validation — always check against the literature

| system | `core` | literature (RHF/STO-3G) |
|---|---|---|
| H₂, $R = 1.4$ Bohr | $-1.11671$ Ha | $-1.117$ (Szabo & Ostlund, Table 3.5) |
| H₂O, exp. geometry | $-74.96305$ Ha | $-74.963$ |
| CH₄, RDKit geometry | $-39.72658$ Ha | $\approx -39.727$ |

---

## 6. Reading the results

### 6.1 Orbital energies and Koopmans' theorem

The converged $\varepsilon_i$ are the MO energies.  For water:

![H2O MO diagram](figures/h2o_mo_diagram.png)

Points worth making in class:

- The O 1s core orbital sits at $-20.24$ Ha ($\approx -551$ eV) — chemically
  inert, hence "core".
- **Koopmans' theorem**: $-\varepsilon_{HOMO}$ approximates the ionization
  energy.  Here $-\varepsilon_{HOMO} = 0.391\ \text{Ha} = 10.6$ eV vs. the
  experimental 12.6 eV — the right order of magnitude from a minimal basis.
- The HOMO ($1b_1$) is the pure oxygen lone pair perpendicular to the
  molecular plane — the orbital that makes water a Lewis base.

### 6.2 Mulliken population analysis

Partial charges from the density and overlap matrices:

$$
q_A = Z_A - \sum_{\mu \in A} (\mathbf{D}\mathbf{S})_{\mu\mu}.
$$

```python
DS_diag = np.diag(D @ S)
q = atom.Z - sum(DS_diag[k] for k, bf in enumerate(basis)
                 if bf belongs to this atom)
```

For water: $q_O = -0.37$, $q_H = +0.18$ — the familiar bond polarity, obtained
from first principles.  (Warn students: Mulliken charges are notoriously
basis-set dependent; they are a *bookkeeping device*, not an observable.)

---

## 7. Case study: breaking H₂

A one-loop script scans the H–H distance and calls `rhf` at each point —
a complete potential energy surface in ~20 lines:

```python
from core import Molecule, rhf

for d in np.linspace(0.8, 5.0, 25):                    # distances in Bohr
    mol = Molecule.from_atoms([("H", (0, 0, 0)), ("H", (0, 0, d))],
                              unit="bohr")
    energies.append(rhf(mol, verbose=False)["energy"])
```

![H2 dissociation](figures/h2_dissociation.png)

Two lessons in one figure:

1. **Near equilibrium HF is good.**  The minimum at $R \approx 1.35$ Bohr is
   close to the experimental 1.40 Bohr.
2. **RHF fails at dissociation.**  Two isolated H atoms have exactly
   $E = -1$ Ha, but the RHF curve levels off far above it.  Why?  The
   restricted determinant forces both electrons into the *same* spatial
   orbital $\sigma_g$, so even at infinite separation the wavefunction
   contains 50 % ionic terms $\mathrm{H^+\!\cdots H^-}$.  This is the
   textbook motivation for electron **correlation** methods (CI, CASSCF,
   coupled cluster) — a perfect cliffhanger for the next lecture.

---

## 8. Exercises

1. **Basis-set play.** Change one STO-3G exponent of hydrogen by ±20 % and
   recompute H₂.  Which direction raises the energy?  Why can the energy only
   go *up* when you de-optimize (variational principle)?
2. **Bond angle of water.**  Loop over the H–O–H angle from 90° to 120° and
   plot $E(\theta)$.  Compare your optimal angle with experiment (104.5°).
3. **Koopmans across a series.**  Compute $-\varepsilon_{HOMO}$ for CH₄, NH₃,
   H₂O, HF (SMILES: `C`, `N`, `O`, `F`) and compare the trend with tabulated
   ionization energies.
4. **HeH⁺.**  The simplest heteronuclear system:
   `Molecule.from_atoms([("He",(0,0,0)),("H",(0,0,1.4632))], charge=1,
   unit="bohr")`.  Reproduce Szabo & Ostlund's $E = -2.860$ Ha.
5. **Count the cost.**  Time `build_eri` for H₂O, NH₃, CH₄ and verify the
   $\mathcal{O}(K^4)$ scaling by log-log fitting.
6. **(Harder) DIIS.**  Implement Pulay's DIIS convergence accelerator using
   the error matrix $\mathbf e = \mathbf{FDS} - \mathbf{SDF}$ and show it cuts
   the number of iterations for H₂O roughly in half.

---

## 9. References

- A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, Dover (1996) — the
  classic; our SCF section follows its §3.4.6 algorithm literally.
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure
  Theory*, Wiley (2000), ch. 9 — the definitive treatment of Gaussian
  integrals and the McMurchie–Davidson scheme.
- L. E. McMurchie, E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978) —
  the original Hermite-expansion paper.
- W. J. Hehre, R. F. Stewart, J. A. Pople, *J. Chem. Phys.* **51**, 2657
  (1969) — the STO-3G basis set.
- S. F. Boys, *Proc. R. Soc. London A* **200**, 542 (1950) — Gaussians enter
  quantum chemistry.

*All figures in this tutorial are generated by
[`docs/make_figures.py`](make_figures.py) — several of them by running
`core` itself. Regenerate with:* `python docs/make_figures.py`
