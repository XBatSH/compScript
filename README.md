# compScript

Hands-on implementations of computational chemistry / structural biology algorithms,
written from scratch in Python to **learn how they actually work**. Every module
pairs runnable code with a full tutorial (English + Chinese) that derives the maths,
shows the essential code, and reports real output from the demo.

The emphasis is on *understanding*, not on production performance: each algorithm is
implemented in the simplest form that still does the real thing.

---

## Modules

### 1. `GaussianShape/` — molecular shape as a sum of Gaussians

Represents a molecule's occupied volume as a sum of atom-centred Gaussians, then
compares two molecules by **shape overlap** (Tanimoto) and **aligns** one onto
another by maximizing that overlap. This is the idea behind shape-based virtual
screening (e.g. ROCS).

- Gaussian volume representation, analytic pairwise overlap integrals
- Shape Tanimoto similarity
- Rigid-body alignment by overlap maximization
- Tutorial: [English](GaussianShape/GaussianShape_tutorial.md) ·
  [中文](GaussianShape/GaussianShape_tutorial_zh.md) ·
  [Notebook](GaussianShape/GaussianShape_tutorial.ipynb)

```bash
cd GaussianShape
python examples/example_shape.py
python examples/example_align_sdf.py
```

### 2. `rotamer/` — side-chain rotamers and packing

Builds a peptide, places side chains from a rotamer library by setting **χ (chi)
dihedrals**, and searches for a low-energy conformation. Includes both a greedy
sweep and the two classic combinatorial optimizers.

- Backbone-independent staggered rotamer library; χ angles set via RDKit
- MMFF94 scoring and backbone-restrained minimization
- Decomposable (pairwise) Lennard-Jones packing energy
- **Dead-End Elimination** (Goldstein criterion) and **simulated annealing**
- Tutorial: [English](rotamer/Rotamer_tutorial.md) ·
  [中文](rotamer/Rotamer_tutorial_zh.md) ·
  [Notebook](rotamer/Rotamer_tutorial.ipynb)

```bash
cd rotamer
python examples/example_rotamer.py
```

### 3. `kinematics_loop/` — protein loop closure as robot inverse kinematics

Treats the backbone as a **serial kinematic chain** (φ/ψ are the joint angles) and
closes a loop between two fixed anchors with **Cyclic Coordinate Descent** — the
robotics inverse-kinematics algorithm. Because closure is under-determined, many
conformations fit the same ends, so candidates are ranked by energy.

- NeRF forward kinematics; Rodrigues rotations; signed dihedrals
- CCD with the analytic per-torsion step `θ* = atan2(c, b)` (no line search)
- Multi-start closure ranked by a coarse backbone energy (Lennard-Jones +
  Ramachandran)
- Tutorial: [English](kinematics_loop/Kinematics_loop_tutorial.md) ·
  [中文](kinematics_loop/Kinematics_loop_tutorial_zh.md) ·
  [Notebook](kinematics_loop/Kinematics_loop_tutorial.ipynb)

```bash
cd kinematics_loop
python examples/example_loop.py
```

### 4. `internal2cartesian/` — Cartesian ↔ internal coordinate conversion

Implements the complete, lossless round-trip between Cartesian coordinates and
internal coordinates (Z-matrix). Builds peptide backbones from ideal geometry
plus user-specified φ/ψ torsions, propagates torsion changes via Rodrigues
rotation, and extracts the full Z-matrix from Cartesian coordinates.

- `Internal → Cartesian`: builds coordinates in 4 stages (origin, +z, xz-plane, NeRF)
- `Cartesian → Internal`: extracts Z-matrix with intelligent reference-atom selection
- Rodrigues' rotation formula for O(k) torsion propagation on the downstream slice
- IUPAC ↔ NeRF sign convention and exact round-trip (RMSD = 0)
- Peptide backbone building from φ/ψ with ideal Engh-Huber geometry
- Tutorial: [English](internal2cartesian/TorsionPropagation_tutorial.md) ·
  [中文](internal2cartesian/TorsionPropagation_tutorial_zh.md) ·
  [Notebook](internal2cartesian/TorsionPropagation_tutorial.ipynb)

```bash
cd internal2cartesian
python examples/example_peptide.py
python examples/example_torsion_propagation.py
```

### 5. `QM_simple/` — Hartree–Fock from scratch

A minimal restricted Hartree–Fock (RHF) implementation in pure Python. Given a
molecule (from SMILES or from explicit atoms), it builds the STO-3G basis, computes
every molecular integral analytically, and self-consistently solves the Roothaan
equations for the SCF orbitals. No quantum-chemistry library is used — every formula
in the tutorial maps onto a short piece of real code.

- STO-3G basis from contraction of Gaussian primitives
- Boys function, McMurchie–Davidson / Obara–Saika integrals
- Löwdin orthogonalisation, DIIS-accelerated SCF
- Mulliken charges and a canonical MO diagram
- Tutorial: [English](QM_simple/docs/TUTORIAL.md) ·
  [中文](QM_simple/docs/TUTORIAL_zh.md) ·
  [Notebook](QM_simple/docs/QM_simple_tutorial.ipynb)

```bash
cd QM_simple
python examples/example_hf.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (the code uses `X | None` type syntax). RDKit is needed by
`GaussianShape/`, `rotamer/`, and `QM_simple/` (for 3D geometry generation);
`kinematics_loop/` and `internal2cartesian/` need only NumPy, plus Matplotlib for their notebooks.

Each module is self-contained — run its examples from inside that module's
directory, since the demos add their own parent directory to `sys.path`.

Demo results are written to each module's `examples/output/` directory, which is
gitignored: run the examples to regenerate them.

---

## Layout

```
compScript/
├── GaussianShape/
│   ├── core/            # gaussian_shape.py: volumes, overlap, alignment
│   ├── visualize/       # plotting helpers
│   ├── docs/images/     # tutorial figures
│   └── examples/
├── rotamer/
│   ├── core/            # peptide, residues, rotamer_lib, energy, search, optimize
│   └── examples/
├── kinematics_loop/
│   ├── core/            # geometry, backbone, ccd, energy
│   └── examples/
├── internal2cartesian/
│   ├── core/            # convert.py: internal_to_cartesian, cartesian_to_internal, place_atom
│   └── examples/
└── QM_simple/
    ├── core/            # molecule, basis, integrals, scf
    └── examples/
```

---

## References

Each tutorial ends with its own reference list. The central papers:

- Grant, J. A.; Pickett, S. D. *A Gaussian description of molecular shape.*
  J. Phys. Chem. 1995, 99, 3503-3510.
- Desmet, J. et al. *The dead-end elimination theorem and its use in protein
  side-chain positioning.* Nature 1992, 356, 539-542.
- Canutescu, A. A.; Dunbrack, R. L. *Cyclic coordinate descent: A robotics
  algorithm for protein loop closure.* Protein Science 2003, 12, 963-972.
- Hehre, W. J. et al. *Ab initio molecular-orbital theory.* Prog. Phys. Chem.
  1970 (the STO-3G basis set).
- Szabo, A.; Ostlund, N. *Modern Quantum Chemistry*, 1996 — the standard
  reference for restricted Hartree–Fock.
