# Protein Loop Closure as Robot Inverse Kinematics

A hands-on tutorial for the `kinematics_loop/` module. It explains **what** the
loop-closure problem is, **why** it is the same problem a robot arm solves, and
**how** we close a loop with Cyclic Coordinate Descent (CCD) and then pick the
*good* conformations by energy — with formulas, diagrams, and the essential code.

---

## 1. Motivation: the loop-closure problem

Most of a protein folds into rigid secondary structure (helices, sheets). The
pieces that connect them are **loops**: flexible stretches of backbone whose two
ends are pinned to the rigid framework, but whose middle can adopt many shapes.

The task this module solves:

> *"I know a loop's **sequence** and the backbone positions of the two residues
> that **flank** it. What backbone conformation(s) of the loop connect those two
> fixed ends?"*

This shows up constantly: rebuilding missing loops in a crystal structure,
grafting a loop in antibody design, or sampling loop motion. The two flanking
residues are **anchors** — one fixed end (the N-anchor) and one target the loop
must reach (the C-anchor).

The key realization is that a polypeptide backbone is a **serial kinematic
chain** — exactly like a robot arm — so loop closure *is* the robotics **inverse
kinematics** problem:

```
robot arm                     protein loop
---------                     ------------
fixed base            <->     N-anchor (first residues, fixed)
joint angles          <->     phi/psi backbone torsions
rigid links           <->     bonds of fixed length/angle
end-effector          <->     last atoms of the loop
target pose           <->     C-anchor position to reach
```

The pipeline:

```
sequence + two anchors ──> build chain (forward kinematics) ──> CCD closes the end
                            (phi/psi -> 3D)                     (analytic torsion steps)
                       ──> many random starts ──> rank by energy ──> good conformations
```

---

## 2. The backbone as a kinematic chain

### 2.1 What moves and what stays rigid

Bond **lengths** and bond **angles** in a peptide barely vary, so we hold them at
ideal values (Engh & Huber). The peptide bond is planar, so **ω ≈ 180°** is fixed
too. That leaves only two rotatable torsions per residue — **φ** (about the N–CA
bond) and **ψ** (about the CA–C bond). These are the chain's only degrees of
freedom, the protein equivalent of a robot's joint angles.

We model just the main-chain trace `N, CA, C` per residue, plus one fixed
carbonyl `C0` in front so that φ of the first residue is well defined. Atoms are
stored in build order:

```
C0, N1, CA1, C1, N2, CA2, C2, ..., N_L, CA_L, C_L
```

with index formulas (residue `i`, 1-based): `N_i = 3i-2`, `CA_i = 3i-1`,
`C_i = 3i`.

### 2.2 Forward kinematics: placing an atom (NeRF)

Given three placed atoms `A-B-C` and a bond length, bond angle, and torsion, the
next atom `D` is fully determined. This is the **Natural Extension Reference
Frame** (NeRF) construction: build a local orthonormal frame on `C` and drop `D`
into it.

$$
\hat{\mathbf{bc}}=\frac{\mathbf{c}-\mathbf{b}}{\lVert\mathbf{c}-\mathbf{b}\rVert},\qquad
\hat{\mathbf{n}}=\frac{(\mathbf{b}-\mathbf{a})\times\hat{\mathbf{bc}}}{\lVert\cdots\rVert},\qquad
\mathbf{M}=[\,\hat{\mathbf{bc}}\;\;\hat{\mathbf{n}}\times\hat{\mathbf{bc}}\;\;\hat{\mathbf{n}}\,]
$$

$$
\mathbf{d}_\text{local}=\big(-L\cos\theta,\;\;L\sin\theta\cos\tau,\;\;L\sin\theta\sin\tau\big),\qquad
\mathbf{D}=\mathbf{c}+\mathbf{M}\,\mathbf{d}_\text{local}
$$

where `L` is the bond length, `θ` the bond angle, and `τ` the torsion. Sweeping
this atom-by-atom down the chain turns a list of (φ, ψ) into 3D coordinates —
**forward kinematics**.

---

## 3. Cyclic Coordinate Descent (CCD)

### 3.1 The idea

Closing the loop is *inverse* kinematics: find torsions that move the chain's end
onto the C-anchor. CCD (Canutescu & Dunbrack, 2003) does this with a beautifully
simple loop:

> Walk over the torsions one at a time. For the current torsion, rotate everything
> downstream by the single angle that brings the moving end **as close as possible**
> to its target. Repeat, sweeping over all torsions, until the end reaches the
> target.

The "end" here is the loop's last three atoms `(N_L, CA_L, C_L)` — the
**end-effector** — and the target is the C-anchor's three atoms.

### 3.2 The analytic per-torsion step

The magic is that each torsion's optimal angle has a **closed-form solution** — no
line search. Rotating the moving atoms `M_j` by θ about a bond axis, we want to
minimize the squared distance to their targets `F_j`. Only one part of that
objective depends on θ, and it has the form

$$
f(\theta)=b\cos\theta + c\sin\theta \;+\; \text{const}
$$

which is maximized (distance minimized) at

$$
\theta^* = \operatorname{atan2}(c,\, b),\qquad
b=\sum_j \mathbf{r}_j^{\perp}\!\cdot\mathbf{t}_j,\quad
c=\sum_j (\hat{\mathbf{k}}\times\mathbf{r}_j)\cdot\mathbf{t}_j
$$

where `k̂` is the unit rotation axis, `r_j = M_j − origin`, `r_j^⊥` is `r_j` with
its component along the axis removed, and `t_j = F_j − origin`. One `atan2` per
torsion, and the end-effector moves optimally.

### 3.3 Rotating the downstream block

Applying θ means rigidly rotating every atom downstream of the bond about that
axis, done with **Rodrigues' rotation formula**:

$$
\mathbf{v}_\text{rot}=\mathbf{v}\cos\theta + (\hat{\mathbf{k}}\times\mathbf{v})\sin\theta + \hat{\mathbf{k}}\,(\hat{\mathbf{k}}\cdot\mathbf{v})(1-\cos\theta)
$$

applied to `v = point − origin`. One sweep = one such rotation per torsion; a few
hundred sweeps drive the end-effector RMSD below the tolerance (0.08 Å).

---

## 4. Loop closure is under-determined

A loop has `2L` torsions but only needs to satisfy 6 constraints (the position and
orientation of one end). For any loop longer than ~3 residues there are **infinitely
many** conformations that close. CCD from a given start finds *one* of them; the one
it finds depends entirely on where it started.

That is both a problem and an opportunity: to explore the real conformational
freedom of the loop we **restart CCD from many random torsion sets** and collect
the distinct closures. But "it closes" is a low bar — a geometrically closed loop
can still be full of clashes or sit in forbidden Ramachandran regions. We need a
way to say which closures are *good*.

---

## 5. Ranking closures by energy

CCD only makes the ends *meet*. To pick physically sensible closures we score each
with a coarse backbone **energy** and keep the lowest:

$$
E = E_\text{vdW} + w_\text{rama}\, E_\text{rama}
$$

### 5.1 van der Waals term

A **Lennard-Jones (12-6)** energy over non-adjacent backbone atom pairs. Steric
clashes cost a large positive energy; comfortable spacing is mildly negative:

$$
E_\text{vdW} = \sum_{\text{pairs}} \varepsilon\left[\left(\frac{R_\text{min}}{r}\right)^{12} - 2\left(\frac{R_\text{min}}{r}\right)^{6}\right],
\qquad R_\text{min}=R_i+R_j,\quad \varepsilon=\sqrt{\varepsilon_i\varepsilon_j}
$$

Bonded (1-2) and 1-3 neighbours are excluded, and the separation is clamped at
`0.5·R_min` so a hard overlap gives a large but finite cost.

### 5.2 Ramachandran term

A smooth pseudo-energy that rewards each residue's (φ, ψ) for lying near a favored
basin (α-helix, β-sheet, PPII, left-handed α). Each residue pays its squared
angular distance to the nearest basin centre, normalized by the basin width `σ`:

$$
E_\text{rama} = \sum_i \frac{1}{2\sigma^2}\min_{\text{basins}}\Big(\Delta\varphi_i^2 + \Delta\psi_i^2\Big)
$$

so a backbone sitting deep in an allowed region contributes ≈ 0 and an outlier
contributes a growing penalty.

Because we model only the N-CA-C trace (no side chains, no hydrogen bonds), this
energy is deliberately coarse — it is meant for **ranking** closures, not for
reporting absolute stabilities.

---

## 6. The multi-start solver

`LoopProblem.solve` ties it together:

1. Restart CCD from random (φ, ψ). Keep every closure CCD completes into a
   **candidate pool** (default `max(5·n_solutions, 25)`).
2. Score each candidate with `backbone_energy`.
3. Sort by energy and return the lowest `n_solutions`.

The essential point: CCD provides *feasibility* (the ends meet); the energy
provides *selection* (which feasible loop is good). Ranking only the first few
closures that happened to converge would make the choice arbitrary — so we
deliberately over-sample into a pool, then let the energy decide.

---

## 7. The essential code

Three snippets from `core/` capture the whole idea.

### 7.1 Forward kinematics: place one atom (NeRF)

```python
def place_atom(a, b, c, bond, angle, torsion):
    bc = normalize(c - b)
    n = normalize(np.cross(b - a, bc))
    m = np.stack([bc, np.cross(n, bc), n], axis=1)   # local -> world frame
    d_local = np.array([
        -bond * np.cos(angle),
        bond * np.sin(angle) * np.cos(torsion),
        bond * np.sin(angle) * np.sin(torsion),
    ])
    return c + m @ d_local
```

### 7.2 The analytic CCD step

```python
def optimal_angle(moving, targets, origin, axis_unit):
    b = c = 0.0
    for m, f in zip(moving, targets):
        r = m - origin
        r_perp = r - np.dot(r, axis_unit) * axis_unit
        s = np.cross(axis_unit, r)          # the "sin" direction
        t = f - origin
        b += float(np.dot(r_perp, t))
        c += float(np.dot(s, t))
    return float(np.arctan2(c, b))          # closed form, no line search
```

### 7.3 One CCD sweep, closing the loop

```python
for it in range(max_iter):
    for kind, i in backbone.rotatable_axes():        # each phi/psi torsion
        a, b = backbone.axis_atoms(kind, i)
        origin = backbone.coords[b]
        axis = normalize(backbone.coords[b] - backbone.coords[a])
        theta = optimal_angle(backbone.end_effector(), targets, origin, axis)
        backbone.apply_rotation(kind, i, theta)      # rotate downstream block
    if rmsd(backbone.end_effector(), targets) < tol:
        break
```

The unifying idea: **one `atan2`** analytically closes each torsion, and sweeping
drives the whole end onto its target.

---

## 8. Worked example

Run from the `kinematics_loop/` directory:

```bash
python examples/example_loop.py
```

It takes an 8-residue loop `GSDGKTPN`, synthesizes the two anchors from a known
reference loop (so we have a ground truth), then closes the loop. First a single
CCD run from a fully extended start:

```
Single CCD closure from an extended start:
  converged=True  iterations=369  final RMSD=0.0791 A
```

An extended chain whose end starts ~15 Å from the target is folded onto it to
sub-0.1 Å in a few hundred sweeps, using nothing but per-torsion `atan2` steps.

Then the multi-start solver samples 25 closures and returns the 5 lowest-energy:

| # | energy | rmsd | clashes | rama_bad |
|---|---|---|---|---|
| 1 | 13.60 | 0.080 | 0 | 5 |
| 2 | 15.79 | 0.079 | 0 | 5 |
| 3 | 17.24 | 0.080 | 0 | 6 |
| 4 | 20.12 | 0.080 | 0 | 6 |
| 5 | 20.43 | 0.080 | 0 | 6 |

All five *close* equally well (RMSD ≈ 0.08 Å) — so RMSD cannot tell them apart.
The **energy** is what ranks them, and note the lowest-energy picks also carry the
fewest Ramachandran outliers: the energy is genuinely steering toward better
backbones. Two PDB files (`loop_reference.pdb`, `loop_solution_best.pdb`) are
written to `examples/output/` for PyMOL/VMD.

---

## 9. Inputs, outputs, and parameters

### `LoopBackbone`

| Method | Meaning |
|---|---|
| `LoopBackbone.from_torsions(seq, phi, psi, seed=None)` | forward-kinematics build (radians) |
| `rotatable_axes()` | list of `("phi"|"psi", i)` torsions (ψ of last residue omitted) |
| `apply_rotation(kind, i, theta)` | rotate the downstream block about a bond |
| `end_effector()` | last three atoms `(N_L, CA_L, C_L)` |
| `torsions()` | recover (φ, ψ) in degrees from coordinates |
| `write_pdb(path)` | export the N-CA-C trace |

### `close_loop(backbone, targets, max_iter=5000, tol=0.08)`

Runs CCD in place; returns `ClosureResult(converged, iterations, rmsd, history)`.

### `LoopProblem`

| Member | Meaning |
|---|---|
| `LoopProblem(sequence, seed, targets)` | a loop between an N-anchor (`seed`) and C-anchor (`targets`) |
| `LoopProblem.from_reference(seq, phi_deg, psi_deg)` | build a self-consistent test case + reference backbone |
| `solve(...)` | multi-start CCD ranked by energy → list of `Solution` |

### `LoopProblem.solve(...)`

| Parameter | Meaning | Default |
|---|---|---|
| `n_solutions` | how many best conformations to return | 5 |
| `max_tries` | random restarts to attempt | 200 |
| `tol` | end-effector RMSD (Å) that counts as closed | 0.08 |
| `w_rama` | weight of the Ramachandran term in the energy | 1.0 |
| `candidate_pool` | closures to score before ranking | `max(5·n_solutions, 25)` |
| `seed` | RNG seed for reproducible restarts | 0 |

Each `Solution` carries `backbone`, `phi`, `psi` (degrees), `rmsd`, `iterations`,
`clashes`, `rama_bad`, and `energy` (the ranking score; lower is better).

### `backbone_energy(backbone, w_rama=1.0)`

Coarse ranking energy `vdw_energy(coords) + w_rama · rama_energy(phi, psi)`.

---

## 10. Things to explore

1. **Loop length**: try longer loops. Closure gets *easier* (more freedom) but the
   space of good conformations grows — does a bigger `candidate_pool` help?
2. **`w_rama`**: crank it up. Do the top conformations shift toward canonical
   secondary-structure torsions at the cost of packing?
3. **Anchor span**: move the C-anchor. What happens as the required span approaches
   the fully-extended length of the loop (closure becomes impossible)?
4. **Convergence**: plot `ClosureResult.history` (RMSD per sweep). How does the
   start (extended vs random) change the number of sweeps?
5. **Rama-biased restarts**: draw random starts from favored basins instead of
   uniform torsions, and see how many fewer tries you need for good closures.
6. **Analytic KIC**: replace CCD with the exact Kinematic Closure (KIC) that solves
   the last three torsions in closed form; compare completeness and speed.
7. **Real anchors**: instead of `from_reference`, read the two flanking residues
   from a PDB and rebuild a missing loop between them.

---

## 11. References

- Canutescu, A. A.; Dunbrack, R. L. *Cyclic coordinate descent: A robotics
  algorithm for protein loop closure.* Protein Science 2003, 12, 963-972.
- Coutsias, E. A.; Seok, C.; Jacobson, M. P.; Dill, K. A. *A kinematic view of loop
  closure.* J. Comput. Chem. 2004, 25, 510-528.
- Parsons, J.; Holmes, J. B.; Rojas, J. M.; Tsai, J.; Strauss, C. E. M. *Practical
  conversion from torsion space to Cartesian space for in silico protein synthesis
  (NeRF).* J. Comput. Chem. 2005, 26, 1063-1068.
- Engh, R. A.; Huber, R. *Accurate bond and angle parameters for X-ray protein
  structure refinement.* Acta Cryst. A 1991, 47, 392-400.
- Ramachandran, G. N.; Ramakrishnan, C.; Sasisekharan, V. *Stereochemistry of
  polypeptide chain configurations.* J. Mol. Biol. 1963, 7, 95-99.
