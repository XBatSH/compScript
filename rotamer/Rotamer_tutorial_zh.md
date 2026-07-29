# 多肽侧链 Rotamer：构建与能量最小化

这是 `rotamer/` 模块的实践教程。它通过公式、示意图和核心代码，解释 rotamer
**是什么**、我们**如何**从 rotamer 库放置侧链，以及**如何**通过最小化找到低能量构象。

---

## 1. 动机：rotamer 是什么，我们为何关注它？

蛋白质主链是一条 `N-CA-C` 原子链。每个 `CA` 上都挂着一条**侧链**，其形状由若干
可旋转单键控制。围绕这些键的扭转角称为 **chi（χ）角**：χ1、χ2……

侧链**并非**自由旋转——空间位阻迫使每个 χ 角落入少数几个优势势阱（约 ±60° 和
180°）。每一组这样的优势 χ 值组合就是一个 **rotamer**（旋转异构体）。预测侧链
构象，本质上就是为每个残基选择合适的、既堆积良好又不产生冲突的 rotamer。这正是
SCWRL 等工具和蛋白质设计的核心。

本模块回答一个两步问题：

> *"给定主链，侧链应采用哪些 rotamer，由此得到的低能量 3D 结构是什么？"*

流程如下：

```
序列 ──> 3D 主链 ──> 放置 rotamer（库） ──> 打分（MMFF） ──> 最小化 ──> 低能量构象
        (RDKit 嵌入)   (设置 χ 角)          (贪心挑选)       (弛豫)
```

---

## 2. Chi 角：侧链的坐标

### 2.1 一个 χ 角是四个原子的二面角

每个 χ 角是由四个依次成键的原子 `i-j-k-l` 定义的**二面角**。多数残基的 χ1 就是
`N-CA-CB-CG`。给定这四个坐标，二面角由它们张成的两个平面计算：

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

我们无需手写这些——RDKit 的 `rdMolTransforms` 既能**读取**（`GetDihedralDeg`）
也能**设置**（`SetDihedralDeg`）二面角，并刚性地旋转其下游整个片段。这正是
"放置一个 rotamer"的含义。

### 2.2 每个残基有多少个 χ 角？

χ 角的数量是残基类型的属性。模块在 `core/residues.py` 中存储了 18 种柔性残基每个
χ 的四原子名称：

```
ALA, GLY : 0 个 chi   （无可旋转侧链）
SER, VAL, ... : 1 个 chi
LEU, PHE, ... : 2 个 chi
MET, GLN, GLU : 3 个 chi
LYS, ARG      : 4 个 chi
```

`PRO` 被视为刚性，因为它的 χ 角被锁在一个环内。

---

## 3. Rotamer 库

### 3.1 交错态（staggered states）

由于每个 χ 都位于四面体键的交错位置附近，我们采用三个标准态：

```
        p  (gauche+)   ~  +60 度
        t  (trans)     ~ 180 度
        m  (gauche-)   ~  -60 度
```

于是一个 rotamer 就是一串状态字母，每个 χ 一个。例如 Lys 的 `"mt"` 表示
χ1 = -60°、χ2 = 180°（更深的角默认保持 trans）。

这就是经典的**主链无关交错近似**。生产级工具（Dunbrack、SCWRL）会细化这些均值
并附上主链依赖的频率；这里理想化的角度随后总会被最小化弛豫，从而吸收其中的差异。

### 3.2 枚举与组合数

变动 `n` 个 χ 角会得到 `3^n` 个 rotamer。由于 χ1/χ2 主导侧链的构象特征，
`enumerate_rotamers(resname, max_chi=2)` 默认只变动前两个 χ 角（因此 Lys/Arg
给出 9 个而非 81 个 rotamer），更深的角保持 trans。

---

## 4. 构建多肽并设置 rotamer

`Peptide.from_sequence` 会构建分子、加氢、嵌入一个 3D 构象（ETKDGv3），并做一次
简短的 MMFF 弛豫以得到合理的主链。关键在于 `Chem.MolFromSequence` 会为每个原子
标注其 **PDB 名称**（`N, CA, CB, CG, …`）和残基编号，这正是我们后续按名称定位每个
χ 的依据。

```python
from core import Peptide

pep = Peptide.from_sequence("KLVFF")   # Lys-Leu-Val-Phe-Phe
for res in pep.residues:
    print(res.name, res.number, "n_chi =", res.n_chi)

pep.set_chi(1, 1, -60.0)     # 将 Lys1 的 chi1 设为 -60 度
print(pep.get_all_chi(1))    # 读回 Lys1 的所有 chi
```

设置整个 rotamer 只是**从 χ1 向外**逐个遍历其 χ 值，这样旋转内侧键就不会扰动
已经放置好的外侧角。

---

## 5. 能量与最小化

### 5.1 能量模型

我们复用 RDKit 的 **MMFF94** 力场作为打分函数。它对成键项（键长、键角、扭转）和
非键项（范德华、静电）求和：

$$
E = E_{\text{bond}} + E_{\text{angle}} + E_{\text{torsion}} + E_{\text{vdW}} + E_{\text{elec}}
$$

范德华项正是惩罚**空间冲突**的部分——这是选择 rotamer 时的主导信号。
`mmff_energy(mol)` 返回这个单点能量。

### 5.2 固定主链的最小化

放置 rotamer 之后我们弛豫结构，但通常希望保持主链固定、只让侧链移动（SCWRL 式的
packing）。`minimize` 通过给每个主链原子（`N, CA, C, O`）添加固定点约束来实现：

```python
ff = _force_field(peptide.mol)
if restrain_backbone:
    for idx in _backbone_indices(peptide):
        ff.AddFixedPoint(idx)     # 冻结主链；侧链弛豫
ff.Minimize(maxIts=max_iters)
```

---

## 6. 搜索算法

选择全局最优的 rotamer 组合是组合爆炸问题。我们采用一个简单而有效的**贪心**方案
（dead-end-elimination / SCWRL 的轻量版）：

1. 从嵌入好的多肽出发，记录其能量。
2. 依次对每个柔性残基，尝试它在库中的**所有** rotamer，对每一个计算**整分子**的
   MMFF 能量，并提交能量最低者。
   （由于能量是整分子的，这自动考虑了与主链以及已放置邻居的冲突。）
3. 重复若干轮，让残基能针对更新后的邻居重新优化。
4. 最后做一次固定主链的最小化，弛豫理想化的角度。

结果打包了优化后的多肽、每个残基所选的 rotamer，以及各阶段的能量。

---

## 7. 全局优化：Dead-End Elimination 与模拟退火

贪心扫描很快，但**目光短浅**：一旦提交某个残基，就不会再结合后续残基重新考量该
决策，因此可能陷入局部极小。选择全局最优的 rotamer 组合是一个真正的**组合优化**
问题——`L` 个残基各有 `R` 个 rotamer，就有 `R^L` 种组合。

### 7.1 可分解的能量

让问题变得可解的关键是一个**可分解**（成对）的能量：总能量是一系列每项至多涉及
两个残基的项之和，

$$
E(\text{choice}) = \sum_i E_\text{self}(i, r_i) + \sum_{i<j} E_\text{pair}(i, r_i;\, j, r_j)
$$

- **自能** $E_\text{self}(i, r)$ —— 残基 `i` 处于 rotamer `r` 时，与固定模板（主链
  + 刚性残基）的相互作用，加上其自身的内部张力；
- **成对能** $E_\text{pair}(i, r; j, s)$ —— 两条柔性侧链之间的相互作用。

由于每一项至多涉及两个残基，我们只需**预先计算**一次，存入一个自能向量和一个
成对能矩阵（`build_energy_matrix`）；此后求解器不再接触 3D 坐标——只是把数字相加。
相互作用采用 **Lennard-Jones（12-6）** 范德华能量，这是空间堆积中主导且可干净
分解的信号：

$$
E_\text{LJ}(r) = \varepsilon\left[\left(\frac{R_\text{min}}{r}\right)^{12} - 2\left(\frac{R_\text{min}}{r}\right)^{6}\right],
\qquad R_\text{min}=R_i+R_j,\quad \varepsilon=\sqrt{\varepsilon_i\varepsilon_j}
$$

其中排除成键（1-2）与 1-3 原子对，并设置距离截断以加速。

### 7.2 Dead-End Elimination（DEE，死端消除）

DEE 能**可证明地**剔除不可能出现在全局最小值中的 rotamer。根据 **Goldstein 判据**，
若存在另一个 rotamer `t`，对其他残基的**每一种**可能选择都能降低能量，则残基 `i`
的 rotamer `r` 被消除：

$$
E_\text{self}(i,r) - E_\text{self}(i,t) + \sum_{j\neq i}\min_s\big[E_\text{pair}(i,r;j,s) - E_\text{pair}(i,t;j,s)\big] > 0
$$

每次消除都会缩小搜索空间；迭代往往能把许多残基收缩到唯一的 rotamer（有时直接解出
问题）。

```python
delta = e_self[i][r] - e_self[i][t]
for j in other_residues:
    delta += min(pair(i, r, j, s) - pair(i, t, j, s) for s in allowed[j])
if delta > 0:
    eliminate(r)          # 对邻居的每种选择，r 都被 t 支配
```

### 7.3 模拟退火（SA）

在 DEE 幸存下来的 rotamer 上，模拟退火进行随机搜索。它每次提议改变一个残基的
rotamer，并按 **Metropolis** 准则接受该移动，同时把温度 `T` 从高（探索）几何式
冷却到低（利用）：

$$
P(\text{accept}) = \begin{cases}1 & \Delta E \le 0\\[2pt] e^{-\Delta E / T} & \Delta E > 0\end{cases}
$$

由于能量可分解，每次移动的 $\Delta E$ 只是 **O(L) 的增量更新**，无需完整重算，
因此上千步都很廉价。

### 7.4 使用求解器

```python
from core import solve_rotamers

res = solve_rotamers(pep, method="dee+sa", max_chi=2)
print(res.assignments, res.packing_energy, res.energy_minimized)
```

`method` 可为 `"dee"`（先剪枝，再在幸存者中贪心挑选）、`"sa"`（在所有 rotamer 上
退火）或 `"dee+sa"`（先剪枝再退火——默认）。返回的 `SearchResult` 增加了 `method`
和 `packing_energy`（所选组合的可分解 LJ 能量）；几何结构仍由同样的固定主链 MMFF
最小化收尾。

> **DEE/SA 在 LJ 堆积能量上做选择，最终由 MMFF 裁决。** 可分解性的要求正是选择阶段
> 使用 LJ 而非整分子 MMFF 的原因。在小而不拥挤的多肽上各方法通常一致；DEE/SA 的
> 优势体现在大而堆积紧密、贪心容易陷入困境的体系上。

---

## 8. 核心代码

模块很小；来自 `core/` 的三段代码就概括了全部思想。

### 8.1 放置一个 rotamer（按名称设置 χ 二面角）

```python
def set_chi(self, resnum, chi_index, angle_deg):
    names = chi_atom_names(self.residue(resnum).name)   # 例如 ("N","CA","CB","CG")
    idxs = [self._atom_index(resnum, nm) for nm in names[chi_index - 1]]
    rmt.SetDihedralDeg(self.mol.GetConformer(), *idxs, float(angle_deg))

def set_rotamer(self, resnum, rotamer):
    for i, angle in enumerate(rotamer.chi):   # 从 chi1 向外
        self.set_chi(resnum, i + 1, angle)
```

### 8.2 从库中枚举 rotamer

```python
CHI_STATES = {"p": 60.0, "t": 180.0, "m": -60.0}

for combo in itertools.product(CHI_STATES, repeat=n_vary):     # 3^n_vary
    chi = [CHI_STATES[s] for s in combo]
    chi.extend(CHI_STATES["t"] for _ in range(n_fixed))        # 更深的 chi = trans
    rotamers.append(Rotamer(name="".join(combo), chi=tuple(chi)))
```

### 8.3 贪心构建 + 最小化

```python
for sweep in range(n_passes):
    for res in flexible:
        best = None
        for rot in enumerate_rotamers(res.name, max_chi=max_chi):
            work.set_rotamer(res.number, rot)
            e = mmff_energy(work.mol)                # 整分子能量
            if best is None or e < best.energy:
                best = RotamerScore(res.number, res.name, rot, e)
        work.set_rotamer(res.number, best.rotamer)   # 提交能量最低的 rotamer

minimize(work, restrain_backbone=True)               # 最终弛豫
```

统一的思想：**同一个 MMFF 能量调用**既为每个候选 rotamer 打分，也驱动最终的弛豫。

---

## 9. 完整示例

在 `rotamer/` 目录下运行：

```bash
python examples/example_rotamer.py
```

它会构建 `KLVFF`、扫描 Lys1 的 rotamer、贪心地构建构象并最小化。典型的能量汇总
（kcal/mol）：

| 阶段 | 能量 |
|---|---|
| 嵌入起始 | 115.13 |
| 放置 rotamer 后 | 122.06 |
| 最小化后 | **113.58** |

注意这条曲线的形状：先放置理想化的交错角度会**抬高**能量，随后最小化把它弛豫到
**低于**起始结构。这正是关键所在——离散构建给出一个良好的起始位姿，连续最小化再
精修它。两个 PDB 文件（`peptide_start.pdb`、`peptide_optimized.pdb`）会写入
`examples/output/`，供 PyMOL/VMD 查看。

随后，示例会在同一条多肽上运行各组合优化求解器。三者在 LJ 堆积能量的全局最优上达成
一致——而且 DEE 单独就把每个残基剪到唯一的 rotamer——说明此处贪心已经找到了良好的
能量盆地：

| 方法 | 堆积能量 | 最小化后 |
|---|---|---|
| greedy | – | 113.58 |
| sa | -15.28 | 113.69 |
| dee | -15.28 | 113.69 |
| dee+sa | -15.28 | 113.69 |

（在这条小而不拥挤的多肽上各方法打平；与贪心的 113.58 之差仅源于 LJ 与 MMFF 的
选择差异。DEE/SA 的优势会在更大、堆积更紧密的体系上显现。）

---

## 10. 输入、输出与参数

### `Peptide`

| 方法 | 含义 |
|---|---|
| `Peptide.from_sequence(seq, seed=...)` | 构建并嵌入一个 3D 多肽 |
| `residues` | `ResidueInfo(number, name, n_chi)` 列表 |
| `get_chi(resnum, i)` / `set_chi(resnum, i, deg)` | 读取/设置一个 χ 角（1 起始） |
| `get_all_chi(resnum)` | 一个残基的所有 χ 角 |
| `set_rotamer(resnum, rotamer)` | 应用一个 rotamer 的所有 χ |
| `write_pdb(path)` | 导出结构 |

### `enumerate_rotamers(resname, max_chi=2)`

| 参数 | 含义 |
|---|---|
| `resname` | 3 字母残基名 |
| `max_chi` | 变动多少个 χ 角（`3**max_chi` 个 rotamer）；更深的 χ = trans |

### `build_low_energy_conformation(peptide, ...)`

| 参数 | 含义 | 默认值 |
|---|---|---|
| `max_chi` | 每个残基变动的 χ 角数 | 2 |
| `n_passes` | 对柔性残基的贪心扫描轮数 | 2 |
| `minimize_final` | 运行最终的 MMFF 最小化 | True |
| `restrain_backbone` | 侧链弛豫时保持主链固定 | True |

返回一个 `SearchResult`，包含 `peptide`、`assignments`（resnum → rotamer 名），
以及 `energy_initial` / `energy_constructed` / `energy_minimized`。

### `solve_rotamers(peptide, method="dee+sa", ...)`

| 参数 | 含义 | 默认值 |
|---|---|---|
| `method` | `"dee"`、`"sa"` 或 `"dee+sa"` | `"dee+sa"` |
| `max_chi` | 每个残基变动的 χ 角数 | 2 |
| `sa_steps` | 模拟退火的移动步数 | 4000 |
| `minimize_final` | 运行最终的 MMFF 最小化 | True |
| `restrain_backbone` | 侧链弛豫时保持主链固定 | True |

返回一个 `SearchResult`，额外带有 `method` 和 `packing_energy` 字段。

### `minimize(peptide, max_iters=1000, restrain_backbone=True)`

就地弛豫多肽；返回 `MinimizeResult(energy, converged)`。

---

## 11. 可以探索的方向

1. **`max_chi`**：枚举 Lys/Arg 的全部 χ（`max_chi=4`，81 个 rotamer）。贪心选择会
   改变吗？慢多少？
2. **`restrain_backbone=False`**：让主链也可移动。最终能量和结构如何变化？
3. **更多轮数**：对堆积紧密的序列，`n_passes=3` 是否更好？
4. **更大的多肽/真实主链**：从 PDB 载入主链而非从序列嵌入，再重新 pack 侧链。
5. **更好的库**：用真实的 Dunbrack 主链依赖 rotamer 均值和频率替换交错均值，并用
   `-log(frequency)` 加权打分。
6. **全局搜索**：`dee` / `sa` / `dee+sa` 求解器已内置——试试更大、堆积更紧密、
   贪心与 DEE+SA 会给出不同结果的序列，比较找到的极小值以及 DEE 消除了多少 rotamer。

---

## 12. 参考文献

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
