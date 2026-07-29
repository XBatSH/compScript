# 蛋白质环闭合：将机器人逆运动学用于肽链

`kinematics_loop/` 模块的实践教程。它讲清楚环闭合问题**是什么**、**为什么**它
与机械臂求解的是同一个问题，以及我们**如何**用循环坐标下降（CCD）闭合一个环，
再用能量挑出*好*的构象——配有公式、示意图和核心代码。

---

## 1. 动机：环闭合问题

蛋白质的大部分会折叠成刚性的二级结构（螺旋、折叠片）。连接它们的片段就是
**环（loop）**：一段柔性的主链，两端被固定在刚性骨架上，而中间可以呈现多种形状。

本模块要解决的任务：

> *"我知道一个环的**序列**，也知道它两侧那两个残基的主链坐标。这个环有哪些主链
> 构象能把这两个固定端连起来？"*

这类需求随处可见：补全晶体结构里缺失的环、抗体设计中的环移植、或对环的运动做
采样。两侧的残基就是**锚点（anchor）**——一端固定（N 端锚点），另一端是环必须
够到的目标（C 端锚点）。

关键的洞见是：多肽主链是一条**串联运动链**——和机械臂完全一样——所以环闭合*就是*
机器人学里的**逆运动学**问题：

```
机械臂                        蛋白质环
------                        --------
固定基座            <->        N 端锚点（前几个原子，固定）
关节角              <->        phi/psi 主链二面角
刚性连杆            <->        固定长度/角度的键
末端执行器          <->        环的最后几个原子
目标位姿            <->        要够到的 C 端锚点位置
```

整体流程：

```
序列 + 两个锚点 ──> 建链（正运动学） ──> CCD 把末端闭合
                    (phi/psi -> 三维)     (解析式二面角步长)
             ──> 多次随机起点 ──> 按能量排序 ──> 好的构象
```

---

## 2. 把主链看作运动链

### 2.1 什么在动，什么保持刚性

肽链里的键**长**和键**角**几乎不变，所以我们把它们固定为理想值（Engh & Huber）。
肽键是平面的，因此 **ω ≈ 180°** 也固定。这样每个残基只剩两个可旋转的二面角——
**φ**（绕 N–CA 键）和 **ψ**（绕 CA–C 键）。它们是这条链仅有的自由度，相当于机械
臂的关节角。

我们只建模主链骨架 `N, CA, C`（每残基三个原子），外加最前面一个固定的羰基 `C0`，
这样第一个残基的 φ 才有定义。原子按构建顺序存储：

```
C0, N1, CA1, C1, N2, CA2, C2, ..., N_L, CA_L, C_L
```

下标公式（残基 `i`，从 1 开始）：`N_i = 3i-2`，`CA_i = 3i-1`，`C_i = 3i`。

### 2.2 正运动学：放置一个原子（NeRF）

给定三个已放置的原子 `A-B-C`，再加一个键长、一个键角和一个二面角，下一个原子 `D`
就被完全确定了。这就是**自然延伸参考系**（NeRF）构造：在 `C` 上建一个局部正交系，
把 `D` 放进去。

$$
\hat{\mathbf{bc}}=\frac{\mathbf{c}-\mathbf{b}}{\lVert\mathbf{c}-\mathbf{b}\rVert},\qquad
\hat{\mathbf{n}}=\frac{(\mathbf{b}-\mathbf{a})\times\hat{\mathbf{bc}}}{\lVert\cdots\rVert},\qquad
\mathbf{M}=[\,\hat{\mathbf{bc}}\;\;\hat{\mathbf{n}}\times\hat{\mathbf{bc}}\;\;\hat{\mathbf{n}}\,]
$$

$$
\mathbf{d}_\text{local}=\big(-L\cos\theta,\;\;L\sin\theta\cos\tau,\;\;L\sin\theta\sin\tau\big),\qquad
\mathbf{D}=\mathbf{c}+\mathbf{M}\,\mathbf{d}_\text{local}
$$

其中 `L` 是键长，`θ` 是键角，`τ` 是二面角。沿着链逐个原子这样放下去，就把一串
(φ, ψ) 变成了三维坐标——这就是**正运动学**。

---

## 3. 循环坐标下降（CCD）

### 3.1 核心思想

闭合环是*逆*运动学：找一组二面角，把链的末端移到 C 端锚点上。CCD（Canutescu &
Dunbrack, 2003）用一个极其简洁的循环做到这点：

> 逐个遍历二面角。对当前这个二面角，把它下游的所有原子旋转一个角度，使移动的末端
> **尽可能靠近**目标。反复遍历所有二面角，直到末端到达目标。

这里的"末端"是环最后三个原子 `(N_L, CA_L, C_L)`——即**末端执行器**——目标则是 C
端锚点的三个原子。

### 3.2 每个二面角的解析步长

精妙之处在于：每个二面角的最优角度都有**闭式解**——不需要线搜索。把移动原子 `M_j`
绕某个键轴旋转 θ，我们想最小化它们到目标 `F_j` 的平方距离。这个目标函数里只有一
部分依赖 θ，且形如

$$
f(\theta)=b\cos\theta + c\sin\theta \;+\; \text{常数}
$$

它在下式处取最大（距离最小）：

$$
\theta^* = \operatorname{atan2}(c,\, b),\qquad
b=\sum_j \mathbf{r}_j^{\perp}\!\cdot\mathbf{t}_j,\quad
c=\sum_j (\hat{\mathbf{k}}\times\mathbf{r}_j)\cdot\mathbf{t}_j
$$

其中 `k̂` 是单位旋转轴，`r_j = M_j − origin`，`r_j^⊥` 是 `r_j` 去掉沿轴分量后的
部分，`t_j = F_j − origin`。每个二面角一次 `atan2`，末端执行器就做了最优移动。

### 3.3 旋转下游原子块

施加 θ 意味着把该键下游的所有原子绕这条轴刚性旋转，用**罗德里格斯旋转公式**完成：

$$
\mathbf{v}_\text{rot}=\mathbf{v}\cos\theta + (\hat{\mathbf{k}}\times\mathbf{v})\sin\theta + \hat{\mathbf{k}}\,(\hat{\mathbf{k}}\cdot\mathbf{v})(1-\cos\theta)
$$

作用在 `v = point − origin` 上。一趟扫描 = 每个二面角做一次这样的旋转；几百趟扫描
就能把末端执行器的 RMSD 压到容差（0.08 Å）以下。

---

## 4. 环闭合是欠定的

一个环有 `2L` 个二面角，但只需满足 6 个约束（一端的位置和朝向）。对任何长于约 3
个残基的环，能闭合的构象有**无穷多个**。从某个起点出发，CCD 只找到其中*一个*；找
到哪一个完全取决于从哪里起步。

这既是问题也是机会：为了探索环真正的构象自由度，我们**从许多随机二面角出发反复重启
CCD**，收集不同的闭合结果。但"能闭合"是个很低的门槛——一个几何上闭合的环仍可能布满
冲突，或落在拉氏图的禁区里。我们需要一个标准来判断哪些闭合是*好*的。

---

## 5. 用能量给闭合排序

CCD 只让两端*相遇*。为了挑出物理上合理的闭合，我们用一个粗粒度的主链**能量**给每个
闭合打分，保留最低的：

$$
E = E_\text{vdW} + w_\text{rama}\, E_\text{rama}
$$

### 5.1 范德华项

对非相邻主链原子对做的**兰纳-琼斯（12-6）**能量。空间冲突要付出很大的正能量；间距
舒适则是轻微负值：

$$
E_\text{vdW} = \sum_{\text{pairs}} \varepsilon\left[\left(\frac{R_\text{min}}{r}\right)^{12} - 2\left(\frac{R_\text{min}}{r}\right)^{6}\right],
\qquad R_\text{min}=R_i+R_j,\quad \varepsilon=\sqrt{\varepsilon_i\varepsilon_j}
$$

成键（1-2）和 1-3 邻居被排除，间距被钳制在 `0.5·R_min`，使硬重叠给出一个很大但有
限的代价。

### 5.2 拉氏图项

一个平滑的伪能量，奖励每个残基的 (φ, ψ) 靠近某个偏好盆地（α 螺旋、β 折叠、PPII、
左手 α）。每个残基付出它到最近盆地中心的平方角距离，再用盆地宽度 `σ` 归一化：

$$
E_\text{rama} = \sum_i \frac{1}{2\sigma^2}\min_{\text{basins}}\Big(\Delta\varphi_i^2 + \Delta\psi_i^2\Big)
$$

于是深处允许区的主链贡献 ≈ 0，而离群点贡献一个逐渐增大的惩罚。

因为我们只建模 N-CA-C 骨架（没有侧链、没有氢键），这个能量刻意做得很粗糙——它是用来
给闭合**排序**的，而不是报告绝对稳定性。

---

## 6. 多起点求解器

`LoopProblem.solve` 把这些串起来：

1. 从随机 (φ, ψ) 重启 CCD。把每个成功闭合的结果放进一个**候选池**（默认
   `max(5·n_solutions, 25)`）。
2. 用 `backbone_energy` 给每个候选打分。
3. 按能量排序，返回最低的 `n_solutions` 个。

要点在于：CCD 提供*可行性*（两端相遇）；能量提供*选择性*（哪个可行的环是好的）。
如果只对最先收敛的那几个闭合排序，选择就变得随意——所以我们刻意过采样进一个池子，
再让能量来决定。

---

## 7. 核心代码

`core/` 里三段代码就抓住了全部思想。

### 7.1 正运动学：放置一个原子（NeRF）

```python
def place_atom(a, b, c, bond, angle, torsion):
    bc = normalize(c - b)
    n = normalize(np.cross(b - a, bc))
    m = np.stack([bc, np.cross(n, bc), n], axis=1)   # 局部 -> 世界坐标系
    d_local = np.array([
        -bond * np.cos(angle),
        bond * np.sin(angle) * np.cos(torsion),
        bond * np.sin(angle) * np.sin(torsion),
    ])
    return c + m @ d_local
```

### 7.2 解析式 CCD 步长

```python
def optimal_angle(moving, targets, origin, axis_unit):
    b = c = 0.0
    for m, f in zip(moving, targets):
        r = m - origin
        r_perp = r - np.dot(r, axis_unit) * axis_unit
        s = np.cross(axis_unit, r)          # “sin” 方向
        t = f - origin
        b += float(np.dot(r_perp, t))
        c += float(np.dot(s, t))
    return float(np.arctan2(c, b))          # 闭式解，无需线搜索
```

### 7.3 一趟 CCD 扫描，闭合环

```python
for it in range(max_iter):
    for kind, i in backbone.rotatable_axes():        # 每个 phi/psi 二面角
        a, b = backbone.axis_atoms(kind, i)
        origin = backbone.coords[b]
        axis = normalize(backbone.coords[b] - backbone.coords[a])
        theta = optimal_angle(backbone.end_effector(), targets, origin, axis)
        backbone.apply_rotation(kind, i, theta)      # 旋转下游原子块
    if rmsd(backbone.end_effector(), targets) < tol:
        break
```

统一的思想是：**一次 `atan2`** 就解析地闭合每个二面角，反复扫描把整个末端驱动到
目标上。

---

## 8. 完整示例

在 `kinematics_loop/` 目录下运行：

```bash
python examples/example_loop.py
```

它取一个 8 残基的环 `GSDGKTPN`，从一个已知的参考环合成出两个锚点（这样就有了真值），
然后闭合这个环。先是从完全伸展的起点做一次 CCD：

```
Single CCD closure from an extended start:
  converged=True  iterations=369  final RMSD=0.0791 A
```

一条伸展的链，其末端起初离目标约 15 Å，仅靠每个二面角的 `atan2` 步长，几百趟扫描
就把它折叠到目标处、达到亚 0.1 Å。

接着多起点求解器采样 25 个闭合，返回能量最低的 5 个：

| # | energy | rmsd | clashes | rama_bad |
|---|---|---|---|---|
| 1 | 13.60 | 0.080 | 0 | 5 |
| 2 | 15.79 | 0.079 | 0 | 5 |
| 3 | 17.24 | 0.080 | 0 | 6 |
| 4 | 20.12 | 0.080 | 0 | 6 |
| 5 | 20.43 | 0.080 | 0 | 6 |

五个都同样*闭合*得很好（RMSD ≈ 0.08 Å）——所以 RMSD 无法区分它们。真正给它们排序
的是**能量**，而且注意：能量最低的几个同时拥有最少的拉氏图离群残基——能量确实在把
选择引向更好的主链。两个 PDB 文件（`loop_reference.pdb`、`loop_solution_best.pdb`）
被写入 `examples/output/`，供 PyMOL/VMD 查看。

---

## 9. 输入、输出与参数

### `LoopBackbone`

| 方法 | 含义 |
|---|---|
| `LoopBackbone.from_torsions(seq, phi, psi, seed=None)` | 正运动学建链（弧度） |
| `rotatable_axes()` | `("phi"|"psi", i)` 二面角列表（省略最后残基的 ψ） |
| `apply_rotation(kind, i, theta)` | 绕某个键旋转下游原子块 |
| `end_effector()` | 最后三个原子 `(N_L, CA_L, C_L)` |
| `torsions()` | 从坐标反算出 (φ, ψ)（度） |
| `write_pdb(path)` | 导出 N-CA-C 骨架 |

### `close_loop(backbone, targets, max_iter=5000, tol=0.08)`

就地运行 CCD；返回 `ClosureResult(converged, iterations, rmsd, history)`。

### `LoopProblem`

| 成员 | 含义 |
|---|---|
| `LoopProblem(sequence, seed, targets)` | 位于 N 端锚点（`seed`）和 C 端锚点（`targets`）之间的环 |
| `LoopProblem.from_reference(seq, phi_deg, psi_deg)` | 构建自洽的测试用例 + 参考主链 |
| `solve(...)` | 按能量排序的多起点 CCD → `Solution` 列表 |

### `LoopProblem.solve(...)`

| 参数 | 含义 | 默认值 |
|---|---|---|
| `n_solutions` | 返回多少个最优构象 | 5 |
| `max_tries` | 尝试的随机重启次数 | 200 |
| `tol` | 判定闭合的末端 RMSD（Å） | 0.08 |
| `w_rama` | 能量中拉氏图项的权重 | 1.0 |
| `candidate_pool` | 排序前打分的闭合数量 | `max(5·n_solutions, 25)` |
| `seed` | 可复现重启的随机种子 | 0 |

每个 `Solution` 携带 `backbone`、`phi`、`psi`（度）、`rmsd`、`iterations`、
`clashes`、`rama_bad`，以及 `energy`（排序分数，越低越好）。

### `backbone_energy(backbone, w_rama=1.0)`

粗粒度排序能量 `vdw_energy(coords) + w_rama · rama_energy(phi, psi)`。

---

## 10. 可以探索的方向

1. **环的长度**：试试更长的环。闭合会*更容易*（自由度更多），但好构象的空间也随之
   增大——更大的 `candidate_pool` 有帮助吗？
2. **`w_rama`**：把它调大。顶端构象会不会以牺牲堆积为代价，向经典二级结构二面角靠拢？
3. **锚点跨距**：移动 C 端锚点。当所需跨距接近环的完全伸展长度（闭合变得不可能）时会
   发生什么？
4. **收敛过程**：画出 `ClosureResult.history`（每趟扫描的 RMSD）。起点（伸展 vs 随机）
   如何改变扫描趟数？
5. **拉氏图偏置重启**：从偏好盆地而非均匀分布采样随机起点，看看好闭合需要的尝试次数
   少了多少。
6. **解析式 KIC**：用精确的运动学闭合（KIC）替换 CCD——它以闭式解求解最后三个二面角；
   比较完备性和速度。
7. **真实锚点**：不用 `from_reference`，而是从 PDB 里读取两侧残基，在它们之间重建一个
   缺失的环。

---

## 11. 参考文献

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
