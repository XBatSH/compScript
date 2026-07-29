# Peptide Side-Chain Rotamers: Construction and Energy Minimization

A hands-on tutorial for the `rotamer/` module. It explains **what** a rotamer is,
**how** we place side chains from a rotamer library, and **how** we find a
low-energy conformation by minimization — with formulas, diagrams, and the
essential code.

---

## 1. Motivation: what is a rotamer, and why do we care?

A protein backbone is a chain of `N-CA-C` atoms. Hanging off each `CA` is a
**side chain** whose shape is controlled by a handful of rotatable single bonds.
The torsion angles about those bonds are called **chi (χ) angles**: χ1, χ2, ….

Side chains do **not** rotate freely — steric strain forces each χ angle into one
of a few preferred wells (near ±60° and 180°). Each such combination of preferred
χ values is a **rotamer**. Predicting side-chain conformations is exactly the
problem of choosing, for every residue, the right rotamer that packs well without
clashing. This is the core of tools like SCWRL and of protein design.

The module answers a two-step question:

> *"Given a backbone, which rotamers should the side chains adopt, and what is the
> resulting low-energy 3D structure?"*

The pipeline:

```
sequence ──> 3D backbone ──> place rotamers (library) ──> score (MMFF) ──> minimize ──> low-energy conformation
             (RDKit embed)     (set χ angles)            (greedy pick)     (relax)
```

---

## 2. Chi angles: the coordinates of a side chain

### 2.1 A χ angle is a dihedral of four atoms

Each χ angle is the **dihedral** defined by four consecutively bonded atoms
`i-j-k-l`. For χ1 of most residues that is `N-CA-CB-CG`. Given the four positions,
the dihedral is computed from the two planes they span:

$$
\mathbf{b}_1=\mathbf{r}_j-\mathbf{r}_i,\quad
\mathbf{b}_2=\mathbf{r}_k-\mathbf{r}_j,\quad
\mathbf{b}_3=\mathbf{r}_l-\mathbf{r}_k
$$

$$
\mathbf{n}_1=\mathbf{b}_1\times\mathbf{b}_2,\qquad
\mathbf{n}_2=\mathbf{b}_2\times\mathbf{b}_3
$$

$$
\chi=\operatorname{atan2}\!\Big((\mathbf{n}_1\times\mathbf{n}_2)\cdot\hat{\mathbf{b}}_2,\;\;\mathbf{n}_1\cdot\mathbf{n}_2\Big)
$$

We never implement this by hand — RDKit's `rdMolTransforms` both **reads**
(`GetDihedralDeg`) and **sets** (`SetDihedralDeg`) a dihedral, rotating the whole
downstream fragment rigidly. That is exactly what "placing a rotamer" means.

### 2.2 How many χ angles per residue?

The number of χ angles is a property of the residue type. The module stores the
four-atom names of every χ for the 18 flexible residues in
`core/residues.py`:

```
ALA, GLY : 0 chi   (no rotatable side chain)
SER, VAL, ... : 1 chi
LEU, PHE, ... : 2 chi
MET, GLN, GLU : 3 chi
LYS, ARG      : 4 chi
```

`PRO` is treated as rigid because its χ angles are locked inside a ring.

---

## 3. The rotamer library

### 3.1 Staggered states

Because each χ sits near a staggered position of a tetrahedral bond, we use three
canonical states:

```
        p  (gauche+)   ~  +60 deg
        t  (trans)     ~ 180 deg
        m  (gauche-)   ~  -60 deg
```

A rotamer is then just a string of state letters, one per χ. For example Lys
`"mt"` means χ1 = -60°, χ2 = 180° (deeper angles held at trans by default).

This is the classic **backbone-independent staggered approximation**. Production
tools (Dunbrack, SCWRL) refine these means and attach backbone-dependent
frequencies; here the idealized angles are always relaxed later by minimization,
which absorbs the difference.

### 3.2 Enumeration and combinatorics

Varying `n` χ angles gives `3^n` rotamers. Since χ1/χ2 dominate a side chain's
identity, `enumerate_rotamers(resname, max_chi=2)` varies only the first two χ
angles by default (so Lys/Arg give 9 rotamers instead of 81) and keeps deeper
angles trans.

---

## 4. Building a peptide and setting rotamers

`Peptide.from_sequence` builds the molecule, adds hydrogens, embeds a 3D
conformer (ETKDGv3), and does a short MMFF relaxation to get a sensible backbone.
Crucially, `Chem.MolFromSequence` tags every atom with its **PDB name**
(`N, CA, CB, CG, …`) and residue number, which is how we later locate each χ by
name.

```python
from core import Peptide

pep = Peptide.from_sequence("KLVFF")   # Lys-Leu-Val-Phe-Phe
for res in pep.residues:
    print(res.name, res.number, "n_chi =", res.n_chi)

pep.set_chi(1, 1, -60.0)     # set Lys1 chi1 to -60 deg
print(pep.get_all_chi(1))    # read back all chi of Lys1
```

Setting a whole rotamer just loops over its χ values **from χ1 outward**, so
rotating an inner bond does not disturb an already-placed outer angle.

---

## 5. Energy and minimization

### 5.1 The energy model

We reuse RDKit's **MMFF94** force field as the scoring function. It sums bonded
terms (bonds, angles, torsions) and non-bonded terms (van der Waals, electrostatics):

$$
E = E_{\text{bond}} + E_{\text{angle}} + E_{\text{torsion}} + E_{\text{vdW}} + E_{\text{elec}}
$$

The van der Waals term is what penalizes **steric clashes** — the dominant signal
when choosing rotamers. `mmff_energy(mol)` returns this single-point energy.

### 5.2 Backbone-restrained minimization

After placing rotamers we relax the structure, but we usually want to keep the
backbone fixed and let only the side chains move (SCWRL-style packing). `minimize`
does this by adding a fixed-point constraint to every backbone atom (`N, CA, C, O`):

```python
ff = _force_field(peptide.mol)
if restrain_backbone:
    for idx in _backbone_indices(peptide):
        ff.AddFixedPoint(idx)     # freeze backbone; side chains relax
ff.Minimize(maxIts=max_iters)
```

---

## 6. The search algorithm

Choosing the globally best set of rotamers is combinatorial. We use a simple,
effective **greedy** scheme (a light version of dead-end-elimination / SCWRL):

1. Start from the embedded peptide; record its energy.
2. For each flexible residue in turn, try **all** its library rotamers, evaluate
   the **whole-molecule** MMFF energy for each, and commit the lowest.
   (Because the energy is whole-molecule, this automatically accounts for clashes
   with the backbone and with already-placed neighbours.)
3. Repeat for a few passes so residues can re-optimize against updated neighbours.
4. Run a final backbone-restrained minimization to relax the idealized angles.

The result bundles the optimized peptide, the chosen rotamer per residue, and the
energy at each stage.

---

## 7. Global optimization: Dead-End Elimination and simulated annealing

The greedy sweep is fast but *myopic*: once it commits a residue it never revisits
that decision in light of later ones, so it can settle into a local minimum.
Choosing the globally best rotamer set is a genuine **combinatorial optimization**
problem — with `R` rotamers on `L` residues there are `R^L` combinations.

### 7.1 A decomposable energy

The trick that makes the problem tractable is a **decomposable** (pairwise) energy:
the total is a sum of terms that each involve at most two residues,

$$
E(\text{choice}) = \sum_i E_\text{self}(i, r_i) + \sum_{i<j} E_\text{pair}(i, r_i;\, j, r_j)
$$

- **self-energy** $E_\text{self}(i, r)$ — side chain `i` in rotamer `r` interacting
  with the fixed template (backbone + rigid residues) plus its own internal strain;
- **pair-energy** $E_\text{pair}(i, r; j, s)$ — the interaction between two flexible
  side chains.

Because every term touches at most two residues, we **precompute** them once into a
self-energy vector and a pair-energy matrix (`build_energy_matrix`); after that the
solvers never look at 3D coordinates again — they just add up numbers. For the
interaction we use a **Lennard-Jones (12-6)** van der Waals energy, the dominant
cleanly-decomposable signal for steric packing:

$$
E_\text{LJ}(r) = \varepsilon\left[\left(\frac{R_\text{min}}{r}\right)^{12} - 2\left(\frac{R_\text{min}}{r}\right)^{6}\right],
\qquad R_\text{min}=R_i+R_j,\quad \varepsilon=\sqrt{\varepsilon_i\varepsilon_j}
$$

with bonded (1-2) and 1-3 pairs excluded and a distance cutoff for speed.

### 7.2 Dead-End Elimination (DEE)

DEE *provably* discards rotamers that cannot appear in the global minimum. By the
**Goldstein criterion**, rotamer `r` at residue `i` is eliminated if some
alternative `t` lowers the energy for *every* possible choice of the other residues:

$$
E_\text{self}(i,r) - E_\text{self}(i,t) + \sum_{j\neq i}\min_s\big[E_\text{pair}(i,r;j,s) - E_\text{pair}(i,t;j,s)\big] > 0
$$

Each elimination shrinks the search space; iterating often collapses many residues
to a single rotamer (sometimes solving the problem outright).

```python
delta = e_self[i][r] - e_self[i][t]
for j in other_residues:
    delta += min(pair(i, r, j, s) - pair(i, t, j, s) for s in allowed[j])
if delta > 0:
    eliminate(r)          # r is dominated by t, for every neighbour choice
```

### 7.3 Simulated annealing (SA)

Over whatever rotamers survive DEE, simulated annealing runs a stochastic search.
It proposes changing one residue's rotamer at a time and accepts the move with the
**Metropolis** rule, cooling a temperature `T` geometrically from hot (explore) to
cold (exploit):

$$
P(\text{accept}) = \begin{cases}1 & \Delta E \le 0\\[2pt] e^{-\Delta E / T} & \Delta E > 0\end{cases}
$$

Because the energy is decomposable, each move's $\Delta E$ is an **O(L) update**,
not a full recompute, so thousands of steps are cheap.

### 7.4 Using the solvers

```python
from core import solve_rotamers

res = solve_rotamers(pep, method="dee+sa", max_chi=2)
print(res.assignments, res.packing_energy, res.energy_minimized)
```

`method` may be `"dee"` (prune, then pick greedily among survivors), `"sa"` (anneal
over all rotamers), or `"dee+sa"` (prune, then anneal — the default). The returned
`SearchResult` adds `method` and `packing_energy` (the decomposable LJ energy of the
chosen set); the geometry is still finished with the same backbone-restrained MMFF
minimization.

> **DEE/SA select on the LJ packing energy, then MMFF is the final judge.** The
> decomposability requirement is exactly why selection uses LJ rather than
> whole-molecule MMFF. On small, un-crowded peptides all methods usually agree; the
> DEE/SA advantage shows on large, tightly packed systems where greedy gets trapped.

---

## 8. The essential code

The module is small; three snippets from `core/` capture the whole idea.

### 8.1 Placing a rotamer (set χ dihedrals by name)

```python
def set_chi(self, resnum, chi_index, angle_deg):
    names = chi_atom_names(self.residue(resnum).name)   # e.g. ("N","CA","CB","CG")
    idxs = [self._atom_index(resnum, nm) for nm in names[chi_index - 1]]
    rmt.SetDihedralDeg(self.mol.GetConformer(), *idxs, float(angle_deg))

def set_rotamer(self, resnum, rotamer):
    for i, angle in enumerate(rotamer.chi):   # chi1 outward
        self.set_chi(resnum, i + 1, angle)
```

### 8.2 Enumerating rotamers from the library

```python
CHI_STATES = {"p": 60.0, "t": 180.0, "m": -60.0}

for combo in itertools.product(CHI_STATES, repeat=n_vary):     # 3^n_vary
    chi = [CHI_STATES[s] for s in combo]
    chi.extend(CHI_STATES["t"] for _ in range(n_fixed))        # deeper chi = trans
    rotamers.append(Rotamer(name="".join(combo), chi=tuple(chi)))
```

### 8.3 Greedy construction + minimization

```python
for sweep in range(n_passes):
    for res in flexible:
        best = None
        for rot in enumerate_rotamers(res.name, max_chi=max_chi):
            work.set_rotamer(res.number, rot)
            e = mmff_energy(work.mol)                # whole-molecule energy
            if best is None or e < best.energy:
                best = RotamerScore(res.number, res.name, rot, e)
        work.set_rotamer(res.number, best.rotamer)   # commit lowest-energy rotamer

minimize(work, restrain_backbone=True)               # final relaxation
```

The unifying idea: **one MMFF energy call** both scores each candidate rotamer and
drives the final relaxation.

---

## 9. Worked example

Run from the `rotamer/` directory:

```bash
python examples/example_rotamer.py
```

It builds `KLVFF`, scans Lys1's rotamers, greedily constructs a conformation, and
minimizes it. Typical energy summary (kcal/mol):

| Stage | Energy |
|---|---|
| embedded start | 115.13 |
| after rotamer placement | 122.06 |
| after minimization | **113.58** |

Note the shape of the curve: placing idealized staggered angles first **raises**
the energy, then minimization relaxes it **below** the starting structure. That is
the whole point — discrete construction gives a good starting pose, continuous
minimization refines it. Two PDB files (`peptide_start.pdb`,
`peptide_optimized.pdb`) are written to `examples/output/` for PyMOL/VMD.

The example then runs the combinatorial solvers on the same peptide. All three
agree on the global optimum of the LJ packing energy — and DEE alone prunes every
residue to a single rotamer — confirming greedy already found a good basin here:

| method | packing energy | after minimization |
|---|---|---|
| greedy | – | 113.58 |
| sa | -15.28 | 113.69 |
| dee | -15.28 | 113.69 |
| dee+sa | -15.28 | 113.69 |

(On this small, un-crowded peptide the methods tie; the difference from greedy's
113.58 is just LJ-vs-MMFF selection. The DEE/SA advantage appears on larger,
tightly packed systems.)

---

## 10. Inputs, outputs, and parameters

### `Peptide`

| Method | Meaning |
|---|---|
| `Peptide.from_sequence(seq, seed=...)` | build + embed a 3D peptide |
| `residues` | list of `ResidueInfo(number, name, n_chi)` |
| `get_chi(resnum, i)` / `set_chi(resnum, i, deg)` | read / set one χ angle (1-based) |
| `get_all_chi(resnum)` | all χ angles of a residue |
| `set_rotamer(resnum, rotamer)` | apply every χ of a rotamer |
| `write_pdb(path)` | export the structure |

### `enumerate_rotamers(resname, max_chi=2)`

| Parameter | Meaning |
|---|---|
| `resname` | 3-letter residue name |
| `max_chi` | how many χ angles to vary (`3**max_chi` rotamers); deeper χ = trans |

### `build_low_energy_conformation(peptide, ...)`

| Parameter | Meaning | Default |
|---|---|---|
| `max_chi` | χ angles varied per residue | 2 |
| `n_passes` | greedy sweeps over flexible residues | 2 |
| `minimize_final` | run final MMFF minimization | True |
| `restrain_backbone` | keep backbone fixed while side chains relax | True |

Returns a `SearchResult` with `peptide`, `assignments` (resnum → rotamer name),
and `energy_initial` / `energy_constructed` / `energy_minimized`.

### `solve_rotamers(peptide, method="dee+sa", ...)`

| Parameter | Meaning | Default |
|---|---|---|
| `method` | `"dee"`, `"sa"`, or `"dee+sa"` | `"dee+sa"` |
| `max_chi` | χ angles varied per residue | 2 |
| `sa_steps` | simulated-annealing moves | 4000 |
| `minimize_final` | run final MMFF minimization | True |
| `restrain_backbone` | keep backbone fixed while side chains relax | True |

Returns a `SearchResult` with the extra `method` and `packing_energy` fields.

### `minimize(peptide, max_iters=1000, restrain_backbone=True)`

Relaxes the peptide in place; returns `MinimizeResult(energy, converged)`.

---

## 11. Things to explore

1. **`max_chi`**: enumerate all χ of Lys/Arg (`max_chi=4`, 81 rotamers). Does the
   greedy choice change? How much slower is it?
2. **`restrain_backbone=False`**: let the backbone move too. How does the final
   energy and structure change?
3. **More passes**: does `n_passes=3` improve tightly packed sequences?
4. **Bigger peptides / real backbones**: load a backbone from a PDB instead of
   embedding from sequence, and repack the side chains.
5. **Better library**: replace the staggered means with real Dunbrack
   backbone-dependent rotamer means and frequencies, and weight the score by
   `-log(frequency)`.
6. **Global search**: the `dee` / `sa` / `dee+sa` solvers are built in — try a
   larger, tightly packed sequence where greedy and DEE+SA disagree, and compare
   the minima found and the number of rotamers DEE eliminates.

---

## 12. References

- Ponder, J. W.; Richards, F. M. *Tertiary templates for proteins: use of packing
  criteria in the enumeration of allowed sequences.* J. Mol. Biol. 1987, 193, 775-791.
- Dunbrack, R. L. *Rotamer libraries in the 21st century.* Curr. Opin. Struct.
  Biol. 2002, 12, 431-440.
- Krivov, G. G.; Shapovalov, M. V.; Dunbrack, R. L. *Improved prediction of protein
  side-chain conformations with SCWRL4.* Proteins 2009, 77, 778-795.
- Desmet, J.; De Maeyer, M.; Hazes, B.; Lasters, I. *The dead-end elimination
  theorem and its use in protein side-chain positioning.* Nature 1992, 356, 539-542.
- Goldstein, R. F. *Efficient rotamer elimination applied to protein side-chains
  and related spin glasses.* Biophys. J. 1994, 66, 1335-1340.
- Kirkpatrick, S.; Gelatt, C. D.; Vecchi, M. P. *Optimization by simulated
  annealing.* Science 1983, 220, 671-680.
- Halgren, T. A. *Merck molecular force field (MMFF94).* J. Comput. Chem. 1996, 17, 490-519.
