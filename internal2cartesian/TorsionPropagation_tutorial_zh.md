# 扭转角传播与内坐标 ↔ 笛卡尔坐标转换

这是 `internal2cartesian/` 模块的实践教程。它解释内坐标（Z-matrix）与笛卡尔坐标之间的
**完整转换流程**，**为什么**改变单个扭转角会位移旋转轴下游的每一个原子，以及**如何**
用两种等价方法（Rodrigues 旋转和 Z-matrix 重建）实现——包含公式、示意图和核心代码。

---

## 1. 动机：一个扭转角，许多原子

你把主链 \(\psi_2\) 角改变 +30°，或者把侧链 \(\chi_1\) 从 -60° 翻转到 +60°。直觉上你只
动了一个自由度，但从旋转点开始，每个原子都移动了——通常达数埃。为什么？

答案是：由内坐标描述的分子是一个**串联运动链**，每个扭转角是一个**旋转关节**。转动
关节 \(k\) 会刚性地旋转从 \(k+1\) 到链末端的每一个链节：

| 模块 | 如何使用扭转传播 |
|---|---|
| `kinematics_loop/` | CCD 通过逐次旋转"下游切片"来闭合环区 |
| `rotamer/` | 设置 χ 角会旋转侧链向外延伸的其余部分 |
| `internal2cartesian/` | 正向运动学：改变 Z-matrix 二面角会移动所有后续原子 |

---

## 2. 物理图像：扭转角即旋转关节

### 2.1 原子 A、B 固定；C 是原点；C 之后的一切都旋转

对于任意二面角 \(A-B-C-D\)：

- 原子 **A** 和 **B** 是**固定的**——它们位于旋转轴之前。
- 原子 **C** 是**原点**——其位置不变，但锚定了旋转轴。
- 键 **\(B \rightarrow C\)** 是**旋转轴**。
- 原子 **D** 以及从 D 出发通过键可达的所有原子，作为一个**刚性块**绕轴旋转。

这是正向运动学：原子 D 的位置由下式定义

\[
D = f(A, B, C,\; \text{bond},\; \text{angle},\; \tau)
\]

改变 τ，D 就移动。任何将 D 用作其参考原子之一的原子 E 也会移动——即使 E *自身*的
键长、键角和二面角都未改变——因为其参考框架被旋转了。

### 2.2 主链作为运动链

多肽主链 \([N_1, CA_1, C_1, N_2, CA_2, C_2, \ldots, N_L, CA_L, C_L]\) 每个残基有
两个可旋转扭转角：

\[
\begin{aligned}
\phi_i &= C_{i-1} - N_i - CA_i - C_i \quad &\text{旋转 } C_i \text{ 及其后所有原子} \\
\psi_i &= N_i - CA_i - C_i - N_{i+1} \quad &\text{旋转 } N_{i+1} \text{ 及其后所有原子}
\end{aligned}
\]

**下游切片**是该扭转角改变时发生移动的原子索引列表：

\[
\begin{aligned}
\text{slice}(\phi_i) &= [\text{idx}(C_i),\; \ldots,\; \text{末端}] \\
\text{slice}(\psi_i) &= [\text{idx}(N_{i+1}),\; \ldots,\; \text{末端}]
\end{aligned}
\]

5 残基主链（15 原子）中，旋转 \(\psi_2\) 移动 9 个原子（\(N_3\) 到 \(C_5\)）；
旋转 \(\phi_3\) 移动 7 个原子（\(C_3\) 到 \(C_5\)）。

### 2.3 侧链同理

赖氨酸侧链 \(CA - CB - CG - CD - CE - NZ\)：每个 χ 角都是一个旋转关节。

\[
\begin{aligned}
\chi_1 &= N-CA-CB-CG \quad &\text{旋转 } CG, CD, CE, NZ \\
\chi_2 &= CA-CB-CG-CD \quad &\text{旋转 } CD, CE, NZ \\
\chi_3 &= CB-CG-CD-CE \quad &\text{旋转 } CE, NZ \\
\chi_4 &= CG-CD-CE-NZ \quad &\text{仅旋转 } NZ
\end{aligned}
\]

改变 \(\chi_1\) 移动 4 个原子，改变 \(\chi_4\) 仅移动 1 个。因此 rotamer 从 \(\chi_1\)
向外设置，使内层旋转不扰动已放置的外层角。

---

## 3. 内坐标 ↔ 笛卡尔坐标：完整转换流程

`internal2cartesian/` 模块实现了两种表示之间完整、无损的往返转换。本节详细讲解

### 3.1 数据结构

**`ZMatrixEntry`** — Z-matrix 的一行。存储单个原子相对于三个已放置参考原子的放置参数
（1 起始索引）：

| 字段 | 含义 |
|---|---|
| `symbol` | 元素符号（如 `"N"`, `"CA"`） |
| `index` | 本原子的 1 起始索引 |
| `bond_to` | 与之成键的原子（1 起始索引） |
| `bond_length` | 到 `bond_to` 的距离（埃） |
| `angle_with` | 1 起始索引；angle(B, C, D) = `angle` |
| `angle` | 键角（弧度） |
| `dihedral_with` | 1 起始索引；dihedral(A, B, C, D) = `dihedral` |
| `dihedral` | 二面角（弧度，NeRF 约定） |

前三个原子无法定义的字段为 `None`：
- 原子 1：所有参考字段为 `None`（置于原点）。
- 原子 2：仅定义了 `bond_to` 和 `bond_length`（沿 +z 轴放置）。
- 原子 3：定义了 `bond_to`, `bond_length`, `angle_with`, `angle`（置于 xz 平面）。

**`InternalCoords`** — 扁平、易于查询的表示：

```python
@dataclass
class InternalCoords:
    bonds:     list[tuple[int, int, float]]              # (i, j, 距离)
    angles:    list[tuple[int, int, int, float]]          # (i, j, k, 角度)
    dihedrals: list[tuple[int, int, int, int, float]]     # (i, j, k, l, 二面角)
```

所有索引均为 0 起始，所有角度为弧度。这是 `extract_backbone_internal` 返回的格式。

### 3.2 内坐标 → 笛卡尔：`internal_to_cartesian(entries)`

给定按顺序排列的 `ZMatrixEntry` 列表（原子 1 到 \(n\)），构建笛卡尔坐标。算法分四步：

**第 1 步 — 原子 1（原点）：**

\[
\mathbf{r}_1 = (0, 0, 0)
\]

**第 2 步 — 原子 2（沿 +z 轴）：**

\[
\mathbf{r}_2 = (0, 0, d_{12}), \quad d_{12} = \text{entry 1 的 bond\_length}
\]

**第 3 步 — 原子 3（在 xz 平面内）：**

给定键长 \(d\)（entry 2 的 `bond_length`）和键角 \(\alpha\)（entry 2 的 `angle`），
放置原子 3 使 1–2–3 键角等于 \(\alpha\)：

\[
\mathbf{r}_3 = \big(d \sin\alpha,\; 0,\; d_{12} - d \cos\alpha\big)
\]

**第 4 步 — 原子 4 至 \(n\)（NeRF）：**

对每个剩余原子 \(k\)，查找其三个参考原子：

```python
a = coords[entry.dihedral_with - 1]   # A（1 起始 → 0 起始）
b = coords[entry.angle_with - 1]      # B
c = coords[entry.bond_to - 1]         # C
```

然后通过 NeRF 构造放置原子 \(k\)（见第 5.1 节）。完整代码
（`convert.py` 第 150–195 行）：

```python
def internal_to_cartesian(entries):
    n = len(entries)
    coords = np.zeros((n, 3))
    coords[0] = [0.0, 0.0, 0.0]                           # 原子 1
    coords[1] = [0.0, 0.0, entries[1].bond_length]        # 原子 2
    b_prev = entries[1].bond_length
    bond = entries[2].bond_length
    ang = entries[2].angle
    coords[2] = [bond * np.sin(ang), 0.0,                 # 原子 3
                 b_prev - bond * np.cos(ang)]
    for k in range(3, n):                                  # 原子 4+
        e = entries[k]
        a, b, c = coords[e.dihedral_with - 1], coords[e.angle_with - 1], coords[e.bond_to - 1]
        coords[k] = place_atom(a, b, c, e.bond_length, e.angle, e.dihedral)
    return coords
```

关键约束：每个参考原子（`bond_to`, `angle_with`, `dihedral_with`）的索引必须**小于**
当前原子的索引。这保证了有效的构建顺序。

### 3.3 笛卡尔 → 内坐标：`cartesian_to_internal(atoms, bonds=None)`

给定 `(symbol, x, y, z)` 元组列表以及（可选的）键图，提取 Z-matrix。这是
`internal_to_cartesian` 的**逆运算**：

1. 对每个原子 \(i\)（按顺序），在已处理原子 \(j < i\) 中选择三个**参考原子**。
2. 从笛卡尔坐标测量几何量：到 `bond_to` 的键长，与 `angle_with` 的键角，以及与
   `dihedral_with` 的二面角。

**参考原子选择**（`_pick_references`，第 270–301 行）：

```
对原子 i：
  1. 优先选择在其之前已**成键**的原子（按最近优先排序）。
  2. 若成键的前驱原子少于 3 个，则从未被选择且在其之前的**最近**原子中填充剩余名额。
  3. 返回 (bond_to, angle_with, dihedral_with)，均为 1 起始索引。
```

这优先选择有化学意义的参考原子（键），同时优雅地退回到几何邻近性——对螺旋等紧凑
结构至关重要，因为仅用最近原子启发式会选择错误的配对。

原子 \(i \ge 4\) 的几何提取：

```python
b_idx, a_idx, d_idx = _pick_references(coords, i, bonded_preceding[i])
bond = ||coords[i] - coords[b_idx - 1]||
ang  = bond_angle(coords[a_idx - 1], coords[b_idx - 1], coords[i])
dih  = dihedral(coords[d_idx - 1], coords[a_idx - 1], coords[b_idx - 1], coords[i])
# 存储时对二面角取反以保证 NeRF 一致性（见第 3.5 节）。
entries.append(ZMatrixEntry(sym, i + 1,
    bond_to=b_idx, bond_length=bond,
    angle_with=a_idx, angle=ang,
    dihedral_with=d_idx, dihedral=-dih))
```

### 3.4 往返：为何它是精确的

该流程满足：

```
笛卡尔 ──[cartesian_to_internal]──> Z-matrix ──[internal_to_cartesian]──> 笛卡尔'
```

且 RMSD(笛卡尔, 笛卡尔') 在浮点精度内为 0。原因：

1. Z-matrix 存储了从笛卡尔坐标*精确*测量的键长、键角和二面角——没有近似。
2. NeRF 构造（`place_atom`）从这些精确参数确定性地重建相同的笛卡尔位置。
3. 符号约定处理一致：IUPAC 二面角（由 `dihedral()` 测量）在录入时取反以匹配 NeRF
   约定，而 NeRF 无论符号选择如何都重建原始几何。

### 3.5 IUPAC ↔ NeRF 符号约定

这是转换中最微妙的一点：

- **IUPAC 约定**（PDB、`dihedral()` 函数所用）：沿 B→C 键方向看，二面角为*正*时 D
  相对于 A **顺时针**旋转。
- **NeRF 约定**（`place_atom`、`build_peptide_from_internal` 所用）：局部框架的
  `n` 向量 = `(b - a) × bc`，指向 IUPAC 法向量的**相反**方向。因此 NeRF 二面角符号**相反**。

代码中的体现（`convert.py`）：

```python
# build_peptide_from_internal 中（第 416, 443 行）：
coords[idx_N] = place_atom(..., -psi[i - 1], ...)   # ψ 取反
coords[idx_C] = place_atom(..., -phi[i - 1], ...)   # φ 取反

# cartesian_to_internal 中（第 264 行）：
dihedral_with=d_idx, dihedral=-dih                   # 测得的二面角取反
```

实用规则：**从 PDB 扭转角转到 NeRF 旋转时取反；从旋转后结构回读扭转角时比较绝对值。**
只要符号一致，Rodrigues 旋转和 Z-matrix 重建产生完全相同的最终几何结构。

---

## 4. 方法一：直接笛卡尔旋转（Rodrigues）

当你已有笛卡尔坐标且只想改变一个扭转角时，Rodrigues 旋转是最高效的方法——\(O(k)\)，
其中 \(k\) 是下游原子数，无需重建分子其余部分。

### 4.1 公式

给定旋转轴 \(\hat{\mathbf{k}}\)（单位向量）、原点 \(\mathbf{o}\) 和角度 \(\theta\)：

\[
\mathbf{r}' = \mathbf{o} + (\mathbf{r} - \mathbf{o})\cos\theta
            + \big[\hat{\mathbf{k}} \times (\mathbf{r} - \mathbf{o})\big]\sin\theta
            + \hat{\mathbf{k}}\big[\hat{\mathbf{k}} \cdot (\mathbf{r} - \mathbf{o})\big](1 - \cos\theta)
\]

### 4.2 对主链施加扭转变化

改变 \(\psi_i\) 即 \(\Delta\theta\)：

1. 旋转轴：键 \(CA_i \rightarrow C_i\)
2. 原点：原子 \(C_i\)
3. 下游切片从 \(N_{i+1}\) 开始
4. 对 `coords[slice_start:]` 应用 Rodrigues 旋转

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

这正是 `kinematics_loop/core/backbone.py:apply_rotation` 所做的，也是 RDKit 的
`SetDihedralDeg` 内部所做的。

### 4.3 侧链同理

赖氨酸 \(\chi_1\)：

```python
# χ₁ = N–CA–CB–CG
# 轴: CA → CB,  原点: CB,  下游从 CG 开始
apply_rotation(coords, idx_CA, idx_CB, slice_start=idx_CG, delta_rad)
```

下游原子恰是以二面角第四个原子为根的子树，可通过分子图深度优先遍历找到。

---

## 5. 方法二：在 Z-matrix 中改变扭转角并重建（NeRF）

### 5.1 NeRF 构造

给定三个已放置原子 \(A, B, C\) 和原子 \(D\) 的几何参数，**自然延伸参考系**（NeRF）
一步放置 D：

\[
\begin{aligned}
\hat{\mathbf{bc}} &= \frac{\mathbf{c} - \mathbf{b}}{\lVert\mathbf{c} - \mathbf{b}\rVert}, \qquad
\hat{\mathbf{n}} = \frac{(\mathbf{b} - \mathbf{a}) \times \hat{\mathbf{bc}}}{\lVert\cdots\rVert}, \qquad
\mathbf{M} = \big[\,\hat{\mathbf{bc}} \;\; \hat{\mathbf{n}} \times \hat{\mathbf{bc}} \;\; \hat{\mathbf{n}}\,\big] \\[6pt]
\mathbf{d}_\text{local} &= \big(-\ell\cos\theta,\;\; \ell\sin\theta\cos\tau,\;\; \ell\sin\theta\sin\tau\big), \qquad
\mathbf{D} = \mathbf{c} + \mathbf{M}\,\mathbf{d}_\text{local}
\end{aligned}
\]

其中 \(\ell = |C-D|\)，\(\theta = \angle(B, C, D)\)，\(\tau\) 为二面角 \(A-B-C-D\)。
矩阵 \(\mathbf{M}\) 是正交框架：\(\hat{\mathbf{bc}}\) 沿键方向，\(\hat{\mathbf{n}}\) 是
A-B-C 平面法向量，\(\hat{\mathbf{n}} \times \hat{\mathbf{bc}}\) 构成右手系。

### 5.2 在 Z-matrix 中改变一个扭转角

1. 找到你想改变其二面角的原子对应的 Z-matrix 条目。
2. 将其 `dihedral` 字段更新 \(\Delta\theta\)。
3. 用 `internal_to_cartesian` 从该条目开始重建笛卡尔坐标。

```python
# 改变原子 8 的二面角，然后从原子 8 开始重建：
entries[7].dihedral = np.deg2rad(new_dihedral_deg)
coords = internal_to_cartesian(entries)
# 原子 0–7 完全一致地重建；原子 8..N 发生偏移。
```

所有索引 ≥ 被改变条目的原子都会得到新位置——每个后续的 `place_atom` 调用看到的
参考原子都已被旋转。结果在数学上与方法一等价。

---

## 6. 何时使用哪种方法

| 场景 | 推荐方法 |
|---|---|
| 已有笛卡尔坐标；微调一个扭转角 | 方法一（Rodrigues）——每次改变 \(O(k)\) |
| 做逆运动学（CCD 环区闭合） | 方法一——`kinematics_loop` 模块即用此法 |
| 用理想键长/键角从头构建几何结构 | 方法二（Z-matrix）——确保理想几何 |
| 从 PDB 提取内坐标 | `cartesian_to_internal`（第 3.3 节） |
| 逐原子更改多个参数后重建 | `internal_to_cartesian`（第 3.2 节） |
| 对 RDKit 分子设置侧链 rotamer | 用 `rdMolTransforms.SetDihedralDeg`（内部即方法一） |
| 验证*只有*目标扭转角发生了变化 | 方法一——然后用 `extract_backbone_internal` 提取二面角确认 |

---

## 7. 与代码库其他部分的关联

### 7.1 `kinematics_loop/` — CCD 环区闭合

`LoopBackbone.apply_rotation(kind, i, theta)` 是方法一的直接实现。一次 CCD 扫描中
每个扭转角被访问，下游切片被 \(\theta^* = \operatorname{atan2}(c, b)\) 旋转。

### 7.2 `rotamer/` — 侧链堆积

`Peptide.set_chi(resnum, chi_index, angle_deg)` 调用 `SetDihedralDeg`，
内部通过键遍历找到下游原子并用 Rodrigues 公式旋转。`set_rotamer` 从 \(\chi_1\) 向外
设置正是基于第 2.3 节的传播逻辑。

### 7.3 `internal2cartesian/` — 转换与多肽构建

- `build_peptide_from_internal`：用 NeRF 沿主链推进，\(\phi, \psi\) 为唯一可变参数。
- `cartesian_to_internal`：从任意分子（给定原子顺序和键图）提取 Z-matrix。
- `internal_to_cartesian`：从任意有效 Z-matrix 重建笛卡尔坐标。
- `extract_backbone_internal`：多肽专用快捷函数，返回结构化格式的所有 \(\phi, \psi, \omega\)。

---

## 8. 核心代码

### 8.1 NeRF：从前三个原子放置一个原子

```python
def place_atom(a, b, c, bond, angle, torsion):
    """放置原子 D: |C-D|=bond, angle(B,C,D)=angle, dihedral(A,B,C,D)=torsion。"""
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

### 8.2 Rodrigues 旋转下游切片

```python
def rotate_points(points, origin, axis, theta):
    """绕通过 origin、方向为 axis 的直线将 points 旋转 theta（弧度）。"""
    k = normalize(axis)
    v = points - origin
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return origin + (v * cos_t + np.cross(k, v) * sin_t
                     + np.outer(v @ k, k) * (1.0 - cos_t))
```

### 8.3 笛卡尔 → Z-matrix，逐原子进行

```python
# 对原子 i (i >= 3)，使用选定的参考原子 bond_to, angle_with, dihedral_with：
bond = np.linalg.norm(coords[i] - coords[bond_to])
ang  = bond_angle(coords[angle_with], coords[bond_to], coords[i])
dih  = dihedral(coords[dihedral_with], coords[angle_with], coords[bond_to], coords[i])
entries.append(ZMatrixEntry(sym, i + 1,
    bond_to=bond_to + 1, bond_length=bond,
    angle_with=angle_with + 1, angle=ang,
    dihedral_with=dihedral_with + 1, dihedral=-dih))   # 取反：IUPAC → NeRF
```

### 8.4 识别主链下游切片

```python
# φ_i: 轴 = N_i → CA_i,  切片从 C_i 开始
phi_slice = idx_C(i)      # 下游 = C_i, N_{i+1}, CA_{i+1}, ...

# ψ_i: 轴 = CA_i → C_i,  切片从 N_{i+1} 开始
psi_slice = idx_N(i + 1)  # 下游 = N_{i+1}, CA_{i+1}, C_{i+1}, ...
```

统一思想：**扭转角变化总是下游块绕键轴的刚体旋转。** 无论你用 Rodrigues 旋转表达在
笛卡尔坐标上，还是用 NeRF 重建表达在 Z-matrix 的二面角值中，物理本质相同。

---

## 9. 完整示例

### 9.1 扭转传播演示

在 `internal2cartesian/` 目录下运行：

```bash
python examples/example_torsion_propagation.py
```

构建五丙氨酸 α-螺旋链（\(\phi=-57^\circ, \psi=-47^\circ\)），然后：

1. **将 \(\psi_2\) 旋转 +30°**——展示原子 0–5 不动，原子 6–14 全部移动。
2. **将 \(\phi_3\) 旋转 +30°**——展示更小的下游块移动，N 端半部冻结。
3. **验证扭转不变性**：提取变化前后的二面角，仅目标扭转角改变。

\(\psi_2\) 旋转的典型输出：

| 原子 | 索引 | 位移 (Å) | 移动? |
|---|---|---|---|
| N1 | 0 | 0.0000 | |
| CA1 | 1 | 0.0000 | |
| C1 | 2 | 0.0000 | |
| N2 | 3 | 0.0000 | |
| CA2 | 4 | 0.0000 | |
| C2 | 5 | 0.0000 | |
| N3 | 6 | 0.7625 | <<< |
| CA3 | 7 | 1.8642 | <<< |

### 9.2 完整转换流程演示

```bash
python examples/example_peptide.py
```

演示完整往返：
1. 从 \(\phi, \psi\) 构建 α-螺旋（内坐标 → 笛卡尔）。
2. 从笛卡尔提取 Z-matrix（笛卡尔 → 内坐标）。
3. 从 Z-matrix 重建笛卡尔（内坐标 → 笛卡尔），验证 RMSD = 0。
4. 提取所有内坐标并对比螺旋与伸展链。
5. 确认输入扭转角与提取的扭转角完全一致。

---

## 10. API 参考

### 核心转换函数

| 函数 | 签名 | 描述 |
|---|---|---|
| `place_atom` | `(a, b, c, bond, angle, torsion) → D` | NeRF：从 A-B-C 放置原子 D |
| `internal_to_cartesian` | `(entries) → (n, 3)` | Z-matrix → 笛卡尔 |
| `cartesian_to_internal` | `(atoms, bonds=None) → list[ZMatrixEntry]` | 笛卡尔 → Z-matrix |
| `dihedral` | `(p0, p1, p2, p3) → float` | 有符号 IUPAC 二面角（弧度） |
| `bond_angle` | `(a, b, c) → float` | 角 A-B-C（弧度） |
| `normalize` | `(v) → v/||v||` | 单位向量 |

### 多肽专用函数

| 函数 | 签名 | 描述 |
|---|---|---|
| `peptide_ideal_geometry` | `() → dict` | 标准主链键长和键角 |
| `build_peptide_from_internal` | `(seq, φ, ψ, ω=180°) → (coords, names)` | 从扭转角构建主链 |
| `extract_backbone_internal` | `(coords) → InternalCoords` | 提取主链 φ, ψ, ω |

### 扭转传播（示例中）

| 函数 | 签名 | 描述 |
|---|---|---|
| `rotate_points` | `(points, origin, axis, theta) → rotated` | Rodrigues 旋转 |
| `apply_torsion_change` | `(coords, a, b, c, start, Δ°) → new_coords` | 旋转下游切片 |

### 数据结构

| 类 | 字段 |
|---|---|
| `ZMatrixEntry` | `symbol, index, bond_to, bond_length, angle_with, angle, dihedral_with, dihedral` |
| `InternalCoords` | `bonds: list[...], angles: list[...], dihedrals: list[...]` |

---

## 11. 可以探索的方向

1. **任意分子的往返**：取乙醇的 SDF，用 `cartesian_to_internal` 构建 Z-matrix，改变
   一个二面角，用 `internal_to_cartesian` 重建。验证仅下游原子移动。
2. **更大的 Δθ**：将 \(\psi_2\) 旋转 180°。绘制位移对原子索引的图——曲线是什么形状？
3. **累积效应**：将 \(\psi_2\) 改变 +30° *并*将 \(\phi_3\) 改变 -30°。最终位移等于各自
   位移之和吗？（否——三维旋转不可交换。）
4. **参考原子选择**：对螺旋分别提供和不提供键图运行 `cartesian_to_internal`。
   Z-matrix 有何不同？各选了什么参考原子？
5. **Omega 扰动**：将肽键（\(\omega\)）旋转 30°。多少原子移动？为何 \(\omega\) 通常
   保持 180°？
6. **Rodrigues vs NeRF 等价性**：用方法一改变一个扭转角，然后独立地在 Z-matrix 中
   改变同一扭转角并用方法二重建。验证 RMSD = 0。
7. **跨模块连接**：运行 `kinematics_loop/examples/example_loop.py`，追踪一个 CCD 步骤。
   `backbone.py:apply_rotation` 与本演示中的 Rodrigues 旋转有何关联？

---

## 12. 参考文献

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
