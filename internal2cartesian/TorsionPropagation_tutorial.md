# Torsion Propagation & Internal ↔ Cartesian Conversion

A hands-on tutorial for the `internal2cartesian/` module. It explains the **full pipeline**
of converting between internal coordinates (Z-matrix) and Cartesian coordinates, **why**
changing a single torsion angle displaces every atom downstream of the rotation axis,
**how** the two equivalent methods (Rodrigues rotation and Z-matrix rebuild) implement
this, and **where** this principle appears across the rest of the codebase — with
formulas, diagrams, and the essential code.

---

## 1. Motivation: one torsion, many atoms

You change the backbone $\psi_2$ angle by +30°, or flip a side-chain $\chi_1$ from -60°
to +60°. Intuitively you touched **one** degree of freedom, yet every atom from the
rotation point onward moves — often by several angstroms. Why?

The answer is that a molecule described by internal coordinates is a **serial kinematic
chain**, and each torsion is a **revolute joint**. Rotating joint $k$ rigidly rotates
every link from $k+1$ to the end of the chain. Understanding *which* atoms move and *by
how much* is essential for everything downstream:

| Module | How it uses torsion propagation |
|---|---|
| `kinematics_loop/` | CCD closes loops by rotating the "downstream slice" one torsion at a time |
| `rotamer/` | Setting a $\chi$ angle rotates the rest of the side chain outward |
| `internal2cartesian/` | Forward kinematics: changing a Z-matrix dihedral shifts all later atoms |

---

## 2. The physical picture: a torsion is a revolute joint

### 2.1 Atoms A, B are stationary; C is the origin; everything after C rotates

For any dihedral $A-B-C-D$:

- Atoms **A** and **B** are **fixed** — they sit before the rotation axis.
- Atom **C** is the **origin** — it does not change position, but it anchors the axis.
- The bond **$B \rightarrow C$** is the **rotation axis**.
- Atom **D** and every atom reachable from D through bonds **rotate as a rigid block**
  about the axis $B \rightarrow C$.

This is forward kinematics: atom D's position is defined by

$$
D = f(A, B, C,\; \text{bond},\; \text{angle},\; \tau)
$$

where $\tau$ is the torsion. Change $\tau$ and D moves. Any atom E that was defined
using D as one of its reference atoms (A, B, or C) will also move — even though E's
*own* bond length, bond angle, and dihedral are unchanged — because its reference frame
has been rotated.

### 2.2 The backbone as a kinematic chain

A peptide backbone with atoms $[N_1, CA_1, C_1, N_2, CA_2, C_2, \ldots, N_L, CA_L, C_L]$
has two rotatable torsions per residue:

$$
\begin{aligned}
\phi_i &= C_{i-1} - N_i - CA_i - C_i \quad &\text{rotates } C_i \text{ and everything after} \\
\psi_i &= N_i - CA_i - C_i - N_{i+1} \quad &\text{rotates } N_{i+1} \text{ and everything after}
\end{aligned}
$$

The **downstream slice** of a torsion is the list of atom indices that move when that
torsion changes:

$$
\begin{aligned}
\text{slice}(\phi_i) &= [\text{idx}(C_i),\; \ldots,\; \text{end}] \\
\text{slice}(\psi_i) &= [\text{idx}(N_{i+1}),\; \ldots,\; \text{end}]
\end{aligned}
$$

For a 5-residue backbone (15 atoms), rotating $\psi_2$ moves the 9 atoms from $N_3$
to $C_5$; rotating $\phi_3$ moves the 7 atoms from $C_3$ to $C_5$.

### 2.3 Side chains: the same physics

Consider a lysine side chain: $CA - CB - CG - CD - CE - NZ$. Each $\chi$ angle is
a revolute joint:

$$
\begin{aligned}
\chi_1 &= N-CA-CB-CG \quad &\text{rotates } CG, CD, CE, NZ \\
\chi_2 &= CA-CB-CG-CD \quad &\text{rotates } CD, CE, NZ \\
\chi_3 &= CB-CG-CD-CE \quad &\text{rotates } CE, NZ \\
\chi_4 &= CG-CD-CE-NZ \quad &\text{rotates } NZ \text{ only}
\end{aligned}
$$

Changing $\chi_1$ moves 4 atoms; changing $\chi_4$ moves only 1. This is why rotamer
libraries store all $\chi$ angles — the cumulative effect of changing $\chi_1$ is
much larger than changing $\chi_4$, and rotamers are set from $\chi_1$ outward so that
inner rotations do not disturb already-placed outer angles.

---

## 3. Internal ↔ Cartesian: the full conversion pipeline

The `internal2cartesian/` module implements a complete, lossless round-trip between the
two representations. This section walks through the algorithms in detail.

### 3.1 The data structures

**`ZMatrixEntry`** — one line of a Z-matrix. Stores a single atom's placement
parameters relative to three already-placed reference atoms (1-based indexing):

| Field | Meaning |
|---|---|
| `symbol` | element symbol (e.g. `"N"`, `"CA"`) |
| `index` | 1-based index of *this* atom |
| `bond_to` | 1-based index of the atom this one is bonded to |
| `bond_length` | distance to `bond_to` in Angstroms |
| `angle_with` | 1-based index; angle(B, C, D) = `angle` |
| `angle` | bond angle in radians |
| `dihedral_with` | 1-based index; dihedral(A, B, C, D) = `dihedral` |
| `dihedral` | dihedral angle in radians (NeRF convention) |

The first three atoms have `None` for fields they cannot define:
- Atom 1: all reference fields are `None` (placed at origin).
- Atom 2: only `bond_to` and `bond_length` defined (placed along +z axis).
- Atom 3: `bond_to`, `bond_length`, `angle_with`, `angle` defined (placed in xz-plane).

**`InternalCoords`** — a flat, query-friendly representation with three lists:

```python
@dataclass
class InternalCoords:
    bonds:     list[tuple[int, int, float]]              # (i, j, distance)
    angles:    list[tuple[int, int, int, float]]          # (i, j, k, angle)
    dihedrals: list[tuple[int, int, int, int, float]]     # (i, j, k, l, dihedral)
```

All indices are 0-based, all angles are in radians. This is the format returned by
`extract_backbone_internal`.

### 3.2 Internal → Cartesian: `internal_to_cartesian(entries)`

Given a list of `ZMatrixEntry` in order (atom 1 to atom $n$), build Cartesian
coordinates. The algorithm proceeds in three stages:

**Stage 1 — Atom 1 (origin):**

$$
\mathbf{r}_1 = (0, 0, 0)
$$

**Stage 2 — Atom 2 (along +z):**

$$
\mathbf{r}_2 = (0, 0, d_{12}), \quad d_{12} = \text{bond\_length of entry 1}
$$

**Stage 3 — Atom 3 (in xz-plane):**

Given bond length $d$ (entry 2's `bond_length`) and angle $\alpha$ (entry 2's `angle`),
atom 3 is placed so that the 1–2–3 bond angle equals $\alpha$:

$$
\mathbf{r}_3 = \big(d \sin\alpha,\; 0,\; d_{12} - d \cos\alpha\big)
$$

**Stage 4 — Atoms 4 through $n$ (NeRF):**

For each remaining atom $k$, look up its three reference atoms:

```python
a = coords[entry.dihedral_with - 1]   # A (1-based → 0-based)
b = coords[entry.angle_with - 1]      # B
c = coords[entry.bond_to - 1]         # C
```

Then place atom $k$ via the NeRF construction (see Section 5.1). The full code
(`convert.py` lines 150–195):

```python
def internal_to_cartesian(entries):
    n = len(entries)
    coords = np.zeros((n, 3))
    coords[0] = [0.0, 0.0, 0.0]                           # atom 1
    coords[1] = [0.0, 0.0, entries[1].bond_length]        # atom 2
    b_prev = entries[1].bond_length
    bond = entries[2].bond_length
    ang = entries[2].angle
    coords[2] = [bond * np.sin(ang), 0.0,                 # atom 3
                 b_prev - bond * np.cos(ang)]
    for k in range(3, n):                                  # atoms 4+
        e = entries[k]
        a, b, c = coords[e.dihedral_with - 1], coords[e.angle_with - 1], coords[e.bond_to - 1]
        coords[k] = place_atom(a, b, c, e.bond_length, e.angle, e.dihedral)
    return coords
```

The key constraint: every reference atom (`bond_to`, `angle_with`, `dihedral_with`) must
have a *smaller* index than the current atom. This guarantees a valid build order.

### 3.3 Cartesian → Internal: `cartesian_to_internal(atoms, bonds=None)`

Given a list of `(symbol, x, y, z)` tuples and optionally a bond graph, extract the
Z-matrix. This is the *inverse* of `internal_to_cartesian`:

1. For each atom $i$ (in order), determine its three **reference atoms** among the
   already-processed atoms $j < i$.
2. Measure the geometric quantities from the Cartesian coordinates: bond length to
   `bond_to`, bond angle with `angle_with`, and dihedral with `dihedral_with`.

**Reference atom selection** (`_pick_references`, lines 270–301):

```
For atom i:
  1. Prefer BONDED atoms that precede i (closest first).
  2. If fewer than 3 bonded predecessors, fill remaining slots from the
     NEAREST preceding atoms not yet chosen.
  3. Return (bond_to, angle_with, dihedral_with) as 1-based indices.
```

This prioritises chemically meaningful references (bonds) while gracefully falling back
to geometric proximity — essential for compact structures like helices where nearest-atom
heuristics alone would pick the wrong partners.

The geometric extraction for atom $i \ge 4$:

```python
b_idx, a_idx, d_idx = _pick_references(coords, i, bonded_preceding[i])
bond = ||coords[i] - coords[b_idx - 1]||
ang  = bond_angle(coords[a_idx - 1], coords[b_idx - 1], coords[i])
dih  = dihedral(coords[d_idx - 1], coords[a_idx - 1], coords[b_idx - 1], coords[i])
# Store with NEGATED dihedral for NeRF consistency (see Section 3.5).
entries.append(ZMatrixEntry(sym, i + 1,
    bond_to=b_idx, bond_length=bond,
    angle_with=a_idx, angle=ang,
    dihedral_with=d_idx, dihedral=-dih))
```

### 3.4 The round-trip: why it is exact

The pipeline satisfies:

```
Cartesian ──[cartesian_to_internal]──> Z-matrix ──[internal_to_cartesian]──> Cartesian'
```

with RMSD(Cartesian, Cartesian') = 0 to floating-point precision. This works because:

1. The Z-matrix stores *exactly* the bond lengths, bond angles, and dihedrals measured
   from the Cartesian coordinates — no approximation.
2. The NeRF construction (`place_atom`) deterministically reconstructs the same
   Cartesian position from those exact parameters.
3. The sign convention is handled consistently: the IUPAC dihedral (measured by
   `dihedral()`) is negated on entry to match the NeRF convention, and NeRF
   reconstructs the original geometry regardless of sign choice.

### 3.5 The IUPAC ↔ NeRF sign convention

This is the single most subtle point of the conversion:

- **IUPAC convention** (used by PDB, the `dihedral()` function): looking down the
  B→C bond, a *positive* dihedral rotates D **clockwise** relative to A.
- **NeRF convention** (used by `place_atom`, `build_peptide_from_internal`): the
  local frame has its `n` vector = `(b - a) × bc`, which points in the *opposite*
  direction from the IUPAC normal. Therefore the NeRF torsion has the **opposite** sign.

Consequences in the code (`convert.py`):

```python
# In build_peptide_from_internal (lines 416, 443):
coords[idx_N] = place_atom(..., -psi[i - 1], ...)   # negate ψ
coords[idx_C] = place_atom(..., -phi[i - 1], ...)   # negate φ

# In cartesian_to_internal (line 264):
dihedral_with=d_idx, dihedral=-dih                   # negate measured dihedral
```

Practical rule: **when you go from a PDB torsion to a NeRF rotation, negate it; when
you read a torsion back from the rotated structure, compare magnitudes.** Both Rodrigues
rotation and Z-matrix rebuild produce identical final geometries as long as the sign is
handled consistently.

---

## 4. Method 1: Direct Cartesian rotation (Rodrigues)

When you already have Cartesian coordinates and only want to change one torsion,
Rodrigues rotation is the most efficient approach — $O(k)$ where $k$ is the number of
downstream atoms, with no rebuild of the rest of the molecule.

### 4.1 The formula

Given a rotation axis $\hat{\mathbf{k}}$ (unit vector), an origin point $\mathbf{o}$,
and an angle $\theta$, Rodrigues' rotation formula rigidly rotates any point
$\mathbf{r}$ to $\mathbf{r}'$:

$$
\mathbf{r}' = \mathbf{o} + (\mathbf{r} - \mathbf{o})\cos\theta
            + \big[\hat{\mathbf{k}} \times (\mathbf{r} - \mathbf{o})\big]\sin\theta
            + \hat{\mathbf{k}}\big[\hat{\mathbf{k}} \cdot (\mathbf{r} - \mathbf{o})\big](1 - \cos\theta)
$$

### 4.2 Applying a torsion change on the backbone

To change $\psi_i$ by $\Delta\theta$:

1. Identify the rotation axis: the bond $CA_i \rightarrow C_i$.
2. The origin is atom $C_i$.
3. The downstream slice starts at $N_{i+1}$.
4. Apply Rodrigues rotation to `coords[slice_start:]`.

```python
def apply_rotation(coords, axis_atom_a, axis_atom_b, slice_start, delta_rad):
    origin = coords[axis_atom_b]               # C_i
    axis = coords[axis_atom_b] - coords[axis_atom_a]  # C_i - CA_i
    k = normalize(axis)
    v = coords[slice_start:] - origin
    cos_t, sin_t = np.cos(delta_rad), np.sin(delta_rad)
    rotated = v * cos_t + np.cross(k, v) * sin_t + np.outer(v @ k, k) * (1.0 - cos_t)
    coords[slice_start:] = origin + rotated
```

This is exactly what `kinematics_loop/core/backbone.py:apply_rotation` does, and what
RDKit's `SetDihedralDeg` does internally.

### 4.3 For side chains

To change $\chi_1$ of a lysine:

```python
# χ₁ = N–CA–CB–CG
# axis: CA → CB,  origin: CB,  downstream starts at CG
apply_rotation(coords, idx_CA, idx_CB, slice_start=idx_CG, delta_rad)
```

The downstream atoms are the subtree rooted at the fourth atom of the dihedral, found
by depth-first traversal of the molecular graph from that atom, skipping the branch back
toward the backbone.

---

## 5. Method 2: Change a torsion in the Z-matrix and rebuild (NeRF)

### 5.1 The NeRF construction

Given three placed atoms $A, B, C$ and the geometric parameters for atom $D$, the
**Natural Extension Reference Frame** (NeRF) algorithm places D in one step:

$$
\begin{aligned}
\hat{\mathbf{bc}} &= \frac{\mathbf{c} - \mathbf{b}}{\lVert\mathbf{c} - \mathbf{b}\rVert}, \qquad
\hat{\mathbf{n}} = \frac{(\mathbf{b} - \mathbf{a}) \times \hat{\mathbf{bc}}}{\lVert\cdots\rVert}, \qquad
\mathbf{M} = \big[\,\hat{\mathbf{bc}} \;\; \hat{\mathbf{n}} \times \hat{\mathbf{bc}} \;\; \hat{\mathbf{n}}\,\big] \\[6pt]
\mathbf{d}_\text{local} &= \big(-\ell\cos\theta,\;\; \ell\sin\theta\cos\tau,\;\; \ell\sin\theta\sin\tau\big), \qquad
\mathbf{D} = \mathbf{c} + \mathbf{M}\,\mathbf{d}_\text{local}
\end{aligned}
$$

where $\ell = |C-D|$, $\theta = \angle(B, C, D)$, and $\tau$ is the dihedral
$A-B-C-D$. The matrix $\mathbf{M}$ is an orthonormal frame: $\hat{\mathbf{bc}}$
points along the bond, $\hat{\mathbf{n}}$ is the plane normal from A-B-C, and
$\hat{\mathbf{n}} \times \hat{\mathbf{bc}}$ completes the right-handed frame.

### 5.2 Changing a torsion in the Z-matrix

1. Find the Z-matrix entry of the atom whose dihedral you want to change.
2. Update its `dihedral` field by $\Delta\theta$.
3. Rebuild Cartesian coordinates from that entry onward using `internal_to_cartesian`.

```python
# Change the dihedral of atom 8, then rebuild from atom 8:
entries[7].dihedral = np.deg2rad(new_dihedral_deg)
coords = internal_to_cartesian(entries)
# Atoms 0–7 are rebuilt identically; atoms 8..N shift.
```

All atoms with index $\geq$ the changed entry get new positions because each
subsequent `place_atom` call sees rotated reference atoms. The result is mathematically
identical to Method 1 — NeRF and Rodrigues are equivalent operations expressed in
different coordinate systems.

---

## 6. When to use each method

| Scenario | Recommended method |
|---|---|
| You have Cartesian coords; tweak one torsion | Method 1 (Rodrigues) — $O(k)$ per change, no rebuild |
| You're doing inverse kinematics (CCD loop closure) | Method 1 — the `kinematics_loop` module uses this |
| You're building geometry from scratch with ideal bond lengths/angles | Method 2 (Z-matrix) — ensures ideal geometry |
| You want to extract internal coordinates from a PDB | `cartesian_to_internal` (Section 3.3) |
| You want to rebuild after changing multiple parameters atom-by-atom | `internal_to_cartesian` (Section 3.2) |
| You want to set side-chain rotamers on an RDKit molecule | Use `rdMolTransforms.SetDihedralDeg` (internally Method 1) |
| You want to verify that *only* the target torsion changed | Method 1 — then extract dihedrals with `extract_backbone_internal` |

---

## 7. Connection to the rest of the codebase

### 7.1 `kinematics_loop/` — CCD loop closure

The `LoopBackbone.apply_rotation(kind, i, theta)` method is a direct implementation of
Method 1. During one CCD sweep, every torsion is visited and the downstream slice is
rotated by the analytic optimal angle $\theta^* = \operatorname{atan2}(c, b)$. The
`downstream_slice(kind, i)` property returns exactly the slice described in Section 2.2.

### 7.2 `rotamer/` — side-chain packing

`Peptide.set_chi(resnum, chi_index, angle_deg)` calls RDKit's
`SetDihedralDeg(conf, a, b, c, d, angle)`, which internally finds all atoms downstream
of atom `c` via bond traversal and rotates them by $\Delta\theta$ using Rodrigues'
formula. The `set_rotamer` method sets chi angles from $\chi_1$ outward precisely
because of the propagation logic described in Section 2.3.

### 7.3 `internal2cartesian/` — conversion and peptide building

- `build_peptide_from_internal` walks down the backbone using NeRF, with $\phi, \psi$
  as the only variable parameters and ideal bond lengths/angles as constants.
- `cartesian_to_internal` extracts the Z-matrix from any molecule given its atom order
  and bond graph.
- `internal_to_cartesian` rebuilds Cartesian coordinates from any valid Z-matrix.
- `extract_backbone_internal` is a peptide-specific shortcut that returns all bond
  lengths, angles, and dihedrals ($\phi, \psi, \omega$) in a structured format.

---

## 8. The essential code

### 8.1 NeRF: place one atom from three predecessors

```python
def place_atom(a, b, c, bond, angle, torsion):
    """Place atom D: |C-D|=bond, angle(B,C,D)=angle, dihedral(A,B,C,D)=torsion."""
    bc = normalize(c - b)
    n = normalize(np.cross(b - a, bc))
    m = np.stack([bc, np.cross(n, bc), n], axis=1)
    d_local = np.array([
        -bond * np.cos(angle),
         bond * np.sin(angle) * np.cos(torsion),
         bond * np.sin(angle) * np.sin(torsion),
    ])
    return c + m @ d_local
```

### 8.2 Rodrigues rotation of a downstream slice

```python
def rotate_points(points, origin, axis, theta):
    """Rotate points by theta (rad) about the line (origin, axis)."""
    k = normalize(axis)
    v = points - origin
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return origin + (v * cos_t + np.cross(k, v) * sin_t
                     + np.outer(v @ k, k) * (1.0 - cos_t))
```

### 8.3 Cartesian → Z-matrix, one atom at a time

```python
# For atom i (i >= 3), with chosen reference atoms bond_to, angle_with, dihedral_with:
bond = np.linalg.norm(coords[i] - coords[bond_to])
ang  = bond_angle(coords[angle_with], coords[bond_to], coords[i])
dih  = dihedral(coords[dihedral_with], coords[angle_with], coords[bond_to], coords[i])
entries.append(ZMatrixEntry(sym, i + 1,
    bond_to=bond_to + 1, bond_length=bond,
    angle_with=angle_with + 1, angle=ang,
    dihedral_with=dihedral_with + 1, dihedral=-dih))   # negate: IUPAC → NeRF
```

### 8.4 Identifying the downstream slice for backbone torsions

```python
# For φ_i: axis = N_i → CA_i,  slice starts at C_i
phi_slice = idx_C(i)      # downstream = C_i, N_{i+1}, CA_{i+1}, ...

# For ψ_i: axis = CA_i → C_i,  slice starts at N_{i+1}
psi_slice = idx_N(i + 1)  # downstream = N_{i+1}, CA_{i+1}, C_{i+1}, ...
```

The unifying idea: **a torsion change is always a rigid-body rotation of a downstream
block about a bond axis.** Whether you express it as Rodrigues rotation on Cartesian
coordinates or as a dihedral value in a Z-matrix rebuilt with NeRF, the physics is
identical.

---

## 9. Worked examples

### 9.1 Torsion propagation demo

Run from the `internal2cartesian/` directory:

```bash
python examples/example_torsion_propagation.py
```

It builds a penta-alanine ($\alpha$-helix, $\phi=-57^\circ, \psi=-47^\circ$), then:

1. **Rotates $\psi_2$ by +30°** — shows that atoms 0–5 ($N_1 \ldots C_2$) are
   stationary while atoms 6–14 ($N_3 \ldots C_5$) all move by up to several angstroms.
2. **Rotates $\phi_3$ by +30°** — shows a smaller downstream block (atoms 8–14, i.e.
   $C_3 \ldots C_5$) moves while the entire N-terminal half is frozen.
3. **Verifies torsion invariance**: extracts all dihedrals before and after, confirming
   that only the target torsion changed.

Typical output (displacement magnitudes for $\psi_2$ rotation):

| atom | index | displacement (Å) | moved? |
|---|---|---|---|
| N1 | 0 | 0.0000 | |
| CA1 | 1 | 0.0000 | |
| C1 | 2 | 0.0000 | |
| N2 | 3 | 0.0000 | |
| CA2 | 4 | 0.0000 | |
| C2 | 5 | 0.0000 | |
| N3 | 6 | 0.7625 | <<< |
| CA3 | 7 | 1.8642 | <<< |
| ... | ... | ... | <<< |

The transition from zero to finite displacement at the slice boundary is instantaneous —
exactly one atom index apart. This is the "downstream slice" concept made visible.

### 9.2 Full conversion pipeline demo

```bash
python examples/example_peptide.py
```

This demonstrates the complete round-trip:

1. Builds an $\alpha$-helix from $\phi, \psi$ angles (`internal → Cartesian`).
2. Extracts the Z-matrix from Cartesian coordinates (`Cartesian → internal`).
3. Rebuilds Cartesian from the Z-matrix (`internal → Cartesian`), verifying RMSD = 0.
4. Extracts all internal coordinates (bonds, angles, $\phi/\psi/\omega$) and compares
   the helix with an extended strand.
5. Confirms that input torsions exactly match extracted torsions.

---

## 10. API reference

### Core conversion functions

| Function | Signature | Description |
|---|---|---|
| `place_atom` | `(a, b, c, bond, angle, torsion) → D` | NeRF: place atom D from A-B-C |
| `internal_to_cartesian` | `(entries: list[ZMatrixEntry]) → (n, 3)` | Z-matrix → Cartesian |
| `cartesian_to_internal` | `(atoms, bonds=None) → list[ZMatrixEntry]` | Cartesian → Z-matrix |
| `dihedral` | `(p0, p1, p2, p3) → float` | Signed IUPAC dihedral (rad) |
| `bond_angle` | `(a, b, c) → float` | Angle A-B-C (rad) |
| `normalize` | `(v) → v/||v||` | Unit vector |

### Peptide-specific functions

| Function | Signature | Description |
|---|---|---|
| `peptide_ideal_geometry` | `() → dict` | Standard backbone bond lengths & angles |
| `build_peptide_from_internal` | `(seq, phi_deg, psi_deg, omega_deg=180) → (coords, names)` | Build backbone from torsions |
| `extract_backbone_internal` | `(coords) → InternalCoords` | Extract $\phi, \psi, \omega$ from backbone |

### Torsion propagation (from example)

| Function | Signature | Description |
|---|---|---|
| `rotate_points` | `(points, origin, axis, theta) → rotated` | Rodrigues rotation |
| `apply_torsion_change` | `(coords, a, b, c, slice_start, delta_deg) → new_coords` | Rotate downstream slice |

### Data structures

| Class | Fields |
|---|---|
| `ZMatrixEntry` | `symbol, index, bond_to, bond_length, angle_with, angle, dihedral_with, dihedral` |
| `InternalCoords` | `bonds: list[(i,j,d)], angles: list[(i,j,k,θ)], dihedrals: list[(i,j,k,l,τ)]` |

---

## 11. Things to explore

1. **Round-trip on an arbitrary molecule**: take the SDF of ethanol, build a Z-matrix
   with `cartesian_to_internal`, change one dihedral, rebuild with
   `internal_to_cartesian`. Verify that only the downstream atoms moved.
2. **Larger $\Delta\theta$**: rotate $\psi_2$ by 180°. Plot displacement vs atom
   index — the curve traces a circle whose radius is the distance from the rotation axis.
3. **Cumulative effects**: change $\psi_2$ by +30° *and* $\phi_3$ by -30° on the same
   structure. Does the final displacement equal the sum of the individual displacements?
   (No — rotations in 3D do not commute.)
4. **Reference atom selection**: run `cartesian_to_internal` on a helix with and without
   providing the bond graph. How does the Z-matrix differ? Which reference atoms are
   chosen in each case?
5. **Omega perturbation**: rotate the peptide bond ($\omega$) by 30°. How many atoms
   move? Why is $\omega$ usually held at 180°?
6. **Rodrigues vs NeRF equivalence**: change a torsion via Method 1, then independently
   change the same torsion in the Z-matrix and rebuild via Method 2. Verify RMSD = 0
   to within floating-point precision.
7. **Cross-module connection**: run `kinematics_loop/examples/example_loop.py` and trace
   one CCD step. How does `backbone.py:apply_rotation` relate to the Rodrigues rotation
   in this demo?

---

## 12. References

- Parsons, J.; Holmes, J. B.; Rojas, J. M.; Tsai, J.; Strauss, C. E. M. *Practical
  conversion from torsion space to Cartesian space for in silico protein synthesis
  (NeRF).* J. Comput. Chem. 2005, 26, 1063-1068.
- Canutescu, A. A.; Dunbrack, R. L. *Cyclic coordinate descent: A robotics algorithm
  for protein loop closure.* Protein Science 2003, 12, 963-972.
- Coutsias, E. A.; Seok, C.; Jacobson, M. P.; Dill, K. A. *A kinematic view of loop
  closure.* J. Comput. Chem. 2004, 25, 510-528.
- Engh, R. A.; Huber, R. *Accurate bond and angle parameters for X-ray protein
  structure refinement.* Acta Cryst. A 1991, 47, 392-400.
- Rodrigues, O. *Des lois géométriques qui régissent les déplacements d'un système
  solide dans l'espace.* J. Math. Pures Appl. 1840, 5, 380-440.
