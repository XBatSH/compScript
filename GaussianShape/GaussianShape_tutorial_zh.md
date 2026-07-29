# 高斯分子形状：表征、比较与对齐

这是 `core/gaussian_shape.py` 模块的实践教程。它通过公式和图示，解释该算法
**计算什么**、**为什么有效**，以及**如何使用**。

---

## 1. 动机：为什么用高斯函数来表征形状？

比较两个 3D 分子最直接的方法是 **RMSD**（原子坐标的均方根偏差）。但 RMSD 有明显的局限：

- 它需要一一对应的**原子映射**（相同的原子、相同的数量、相同的顺序）。
- 它对微小扰动很敏感，且完全不描述分子所**占据的体积**。
- 它无法比较两个恰好形状相似的**不同**分子。

**高斯形状**方法转而回答一个不同的问题：

> *"这两个分子占据空间的方式是否相似？"*

这正是药物发现中基于形状的虚拟筛选（如 OpenEye 的 ROCS）的基础。整个流程如下：

![pipeline](docs/images/pipeline.png)

---

## 2. 形状表征

### 2.1 一个原子 = 一个高斯函数

位于 `r_i` 的每个原子 `i` 被建模为一个各向同性的 3D 高斯函数：

$$
g_i(\mathbf{r}) = \exp\!\left(-\frac{\lVert \mathbf{r} - \mathbf{r}_i \rVert^2}{2\sigma_i^2}\right)
$$

宽度 `sigma_i` 控制原子有多"胖"。在代码中，它由原子的**范德华半径**导出：

$$
\sigma_i = \texttt{radius\_scale} \times R^{\text{vdW}}_i
$$

![gaussian_1d](docs/images/gaussian_1d.png)

*`sigma` 越大，原子越宽、越"柔和"。*

### 2.2 整个分子 = 高斯函数之和

分子形状密度就是对所有原子求和：

$$
\rho(\mathbf{r}) = \sum_{i=1}^{N} g_i(\mathbf{r})
$$

![two_atom_density](docs/images/two_atom_density.png)

*两个原子高斯（虚线）叠加成一条光滑的分子密度曲线（实线）。*

在代码中对应 `GaussianShape.density(points)`。

---

## 3. 度量相似性

### 3.1 重叠积分（核心技巧）

两个形状的相似性就是它们密度的**重叠**，即 `∫ rho_A(r) * rho_B(r) dr`。由于两者
都是高斯函数之和，而两个高斯函数的乘积/积分是解析的，整个积分有**闭式解**——
无需数值格点。

对于分别以 `a`、`b` 为中心的一对高斯：

$$
S_{ab} = \int g_a(\mathbf{r})\, g_b(\mathbf{r})\, d\mathbf{r}
= \left(\frac{2\pi\,\sigma_a^2\,\sigma_b^2}{\sigma_a^2+\sigma_b^2}\right)^{3/2}
\exp\!\left(-\frac{\lVert \mathbf{a}-\mathbf{b}\rVert^2}{2(\sigma_a^2+\sigma_b^2)}\right)
$$

分子 A 与 B 的总重叠是对**所有原子对**求和：

$$
S_{AB} = \sum_{i \in A}\sum_{j \in B} S_{ij}
$$

![overlap_vs_distance](docs/images/overlap_vs_distance.png)

*重叠随中心间距按高斯衰减：相距很远的原子几乎无贡献，彼此重合的原子贡献最大。*

在代码中：`shape_a.overlap(shape_b)`，用 NumPy 完全向量化。

### 3.2 自重叠与 Tanimoto 系数

**自重叠** `S_AA = overlap(A, A)` 衡量一个分子的"形状质量"。形状相似性以
**类 Tanimoto 系数**给出，归一化到 `[0, 1]`：

$$
T = \frac{S_{AB}}{S_{AA} + S_{BB} - S_{AB}}
$$

- `T = 1`：两个形状完全重叠。
- `T = 0`：完全没有重叠。

在代码中：`shape_a.tanimoto(shape_b)`。

---

## 4. 对齐：在同一坐标系下比较形状

Tanimoto 取决于两个分子的**相对位姿**。如果 B 位于很远处，即便形状完全相同，
`T` 也约等于 0。因此在比较之前，必须找到*待对齐*分子的刚体变换（旋转 `R` +
平移 `t`），使其与*参考*分子的重叠**最大化**。

### 4.1 一个有用的简化

在待对齐分子做刚体运动时，它的自重叠 `S_AA` 和参考的 `S_BB` 都是**常数**。由于

$$
T = \frac{S_{AB}}{C - S_{AB}}, \qquad C = S_{AA} + S_{BB} = \text{常数},
$$

且 `dT/dS_AB = C / (C - S_AB)^2 > 0`，所以最大化 `T` **等价于**最大化原始重叠
`S_AB`。这让梯度保持简单。

### 4.2 带解析梯度的 BFGS

我们优化 6 个参数：3 个欧拉角 + 3 个平移，`p = [rz, ry, rx, tx, ty, tz]`。
待对齐原子按 `a' = R(angles) a + t` 变换，我们最小化 `-S_AB(p)`。

由于重叠是光滑的，我们可以给 BFGS 提供**精确梯度**。记
`d_ij = a'_i - b_j`、`s_ij^2 = sigma_i^2 + sigma_j^2`：

$$
\frac{\partial S}{\partial \mathbf{d}_{ij}} = S_{ij}\left(-\frac{\mathbf{d}_{ij}}{s_{ij}^2}\right),
\qquad
\mathbf{G}_i = \sum_j \frac{\partial S}{\partial \mathbf{d}_{ij}}
$$

$$
\frac{\partial S}{\partial \mathbf{t}} = \sum_i \mathbf{G}_i,
\qquad
\frac{\partial S}{\partial \theta_k} = \sum_i \mathbf{G}_i \cdot \left(\frac{\partial R}{\partial \theta_k}\mathbf{a}_i\right)
$$

重叠曲面是**非凸的**，因此优化器会从若干随机旋转（`n_starts`）重启，并保留最优结果。

在代码中：`align_shapes_bfgs(mobile, reference, n_starts, seed)`。此外还提供了一个
更简单、仅旋转的 Powell 基线 `align_shapes(...)`。

### 4.3 变换作用于原始坐标

优化器在质心居中的坐标系下工作，但返回的变换会重新组合到**原始**待对齐坐标上，
因此可以直接应用：

$$
\mathbf{r}' = R(\mathbf{r} - \mathbf{c}_{mob}) + (\mathbf{t}_{opt} + \mathbf{c}_{ref})
            = R\,\mathbf{r} + \underbrace{(-R\,\mathbf{c}_{mob} + \mathbf{t}_{opt} + \mathbf{c}_{ref})}_{\mathbf{t}}
$$

`apply_transform_to_mol(mol, R, t)` 会就地改写 RDKit 构象的坐标，使真实的 3D 结构
移动到对齐后的坐标系。

---

## 5. 输入、输出与参数

### `GaussianShape`

| 项 | 含义 |
|---|---|
| `centers` (N,3) | 原子坐标（单位：埃 Angstrom） |
| `sigmas` (N,) | 每个原子的高斯宽度（单位：埃） |
| `elements` | 元素符号（用于记录/绘图） |

构造方法：

- `GaussianShape.from_coordinates(coords, elements, radius_scale=0.8)`
- `GaussianShape.from_rdkit_mol(mol, conf_id=-1, radius_scale=0.8, include_hydrogens=True)`

| 参数 | 含义 | 典型值 |
|---|---|---|
| `radius_scale` | 乘以范德华半径以设定 `sigma`。越小=原子越紧致、对细节越敏感；越大=越柔和、越宽容。 | 0.7 - 0.9 |
| `include_hydrogens` | 是否包含 H 原子，或仅比较重原子形状 | `True` |
| `conf_id` | 使用哪个 RDKit 构象 | `-1`（默认） |

方法：

- `density(points) -> (M,)`：计算 `rho(r)`（用于可视化）。
- `overlap(other) -> float`：解析的 `S_AB`。
- `self_overlap() -> float`：`S_AA`。
- `tanimoto(other) -> float`：`[0, 1]` 区间的相似度。
- `centroid()`、`translated(v)`、`transformed(R, t)`：几何辅助方法。

### `align_shapes_bfgs(mobile, reference, n_starts=8, seed=0)`

| 参数 | 含义 |
|---|---|
| `mobile` | 需要移动的形状 |
| `reference` | 固定的目标形状 |
| `n_starts` | 随机旋转重启的次数（越多=越稳健，越慢） |
| `seed` | 随机数种子，保证重启可复现 |

返回一个 **`AlignmentResult`**：

| 字段 | 含义 |
|---|---|
| `aligned` | 变换后的 `GaussianShape` |
| `tanimoto` | 对齐后的相似度 |
| `rotation` (3,3) | 作用于原始待对齐坐标的旋转 `R` |
| `translation` (3,) | 作用于原始待对齐坐标的平移 `t` |
| `.matrix` | 4x4 齐次变换矩阵 |
| `.apply_to_coords(coords)` | 将变换应用到任意 `(N,3)` 数组 |

### `apply_transform_to_mol(mol, rotation, translation, conf_id=-1)`

就地改写 RDKit 构象坐标；返回同一个 `mol`。

---

## 6. 核心代码

整个模块都围绕**同一个高斯重叠核**构建，它在三个地方反复出现：密度计算、相似性
打分，以及对齐目标函数。下面是 `core/gaussian_shape.py` 中最精简的几段。

### 6.1 形状表征

一个原子变成一个高斯函数，其宽度来自它的范德华半径：

```python
radii = np.array([VDW_RADII.get(e, DEFAULT_RADIUS) for e in elements])
sigmas = radius_scale * radii            # sigma_i = radius_scale * R_vdw
return cls(centers=coords, sigmas=sigmas, elements=elements)
```

分子密度 `rho(r) = sum_i exp(-|r - r_i|^2 / 2 sigma_i^2)` 把每个查询点与每个原子
做广播：

```python
diff = points[:, None, :] - self.centers[None, :, :]   # (M, N, 3)
sq_dist = np.einsum("mnk,mnk->mn", diff, diff)          # (M, N)
gauss = np.exp(-sq_dist / (2.0 * self.sigmas[None, :] ** 2))
return gauss.sum(axis=1)                                # 对原子求和 -> (M,)
```

`[:, None, :]` / `[None, :, :]` 技巧一次性构造出所有配对组合，因此全程保持向量化。

### 6.2 相似性

两个高斯的重叠积分是**解析的**，因此无需格点。总重叠是对所有原子对的双重求和：

```python
diff = self.centers[:, None, :] - other.centers[None, :, :]  # (Na, Nb, 3)
sq_dist = np.einsum("abk,abk->ab", diff, diff)               # (Na, Nb)

sa2 = self.sigmas[:, None] ** 2     # (Na, 1)
sb2 = other.sigmas[None, :] ** 2    # (1, Nb)
sum_s2 = sa2 + sb2                  # (Na, Nb)

prefactor = (2.0 * np.pi * sa2 * sb2 / sum_s2) ** 1.5
pair_overlap = prefactor * np.exp(-sq_dist / (2.0 * sum_s2))
return float(pair_overlap.sum())    # S_AB
```

相似性即归一化的 Tanimoto：

```python
s_ab = self.overlap(other)
s_aa = self.self_overlap()          # = overlap(A, A)
s_bb = other.self_overlap()
return float(s_ab / (s_aa + s_bb - s_ab))   # T 属于 [0, 1]
```

### 6.3 对齐

对齐就是在刚体变换上最大化 `S_AB`。一个函数**同时**返回重叠及其精确梯度，
使 BFGS 快速收敛：

```python
a_prime = mobile_centers @ rot.T + translation   # a' = R a + t
diff = a_prime[:, None, :] - ref_centers[None, :, :]
d2 = np.einsum("abk,abk->ab", diff, diff)
# ... 与 6.2 相同的重叠核 ...
kernel = prefactor * np.exp(-d2 / (2.0 * s2))
overlap = float(kernel.sum())

# 梯度：dS/d(diff_ij) = kernel_ij * (-diff_ij / s2_ij)
coeff = (-kernel / s2)[:, :, None]
grad_per_atom = np.einsum("abk->ak", coeff * diff)   # G_i，每个原子一个 3 维向量

grad_t = grad_per_atom.sum(axis=0)                   # dS/dt = sum_i G_i
grad_angles = np.array([                             # dS/dtheta = sum_i G_i . (dR/dtheta) a_i
    np.sum(grad_per_atom * (mobile_centers @ d_rz.T)),
    np.sum(grad_per_atom * (mobile_centers @ d_ry.T)),
    np.sum(grad_per_atom * (mobile_centers @ d_rx.T)),
])
```

注意这里的 `kernel` 与 6.2 中的重叠公式完全相同——打分与对齐共用它。优化器在
`[rz, ry, rx, tx, ty, tz]` 上最小化 `-overlap` 并做随机重启，然后把变换重新组合回
原始坐标上。

---

## 7. 完整示例

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from core.gaussian_shape import GaussianShape, align_shapes_bfgs, apply_transform_to_mol

# 1) 构建 3D 分子。
mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol)

# 2) 构建其高斯形状。
shape = GaussianShape.from_rdkit_mol(mol, radius_scale=0.8)
print("self-overlap:", shape.self_overlap())

# 3) 比较两个形状（先对齐）。
result = align_shapes_bfgs(mobile_shape, reference_shape, n_starts=8)
print("Tanimoto:", result.tanimoto)

# 4) 将真实分子移动到对齐坐标系并导出。
apply_transform_to_mol(mol, result.rotation, result.translation)
```

### 可运行的示例脚本

- `python examples/example_shape.py`——成对相似性矩阵（已对齐）、一个构象对齐
  演示，以及 2D 密度图。
- `python examples/example_align_sdf.py`——将分子用随机旋转移动约 54 埃，重新对齐，
  并写出用于 PyMOL 的 SDF 文件。

示例演示图：

| 单个形状密度 | 两个形状叠加 |
|---|---|
| ![benzene density](examples/output/benzene_density.png) | ![benzene vs cyclohexane](examples/output/benzene_vs_cyclohexane.png) |

---

## 8. 在 PyMOL 中可视化

`examples/example_align_sdf.py` 会向 `examples/output/` 写出三个文件：

```
load examples/output/ref.sdf            # 位于原点的参考分子
load examples/output/mobile_start.sdf   # 被移开约 54 埃
load examples/output/mobile_aligned.sdf # 重新贴合到参考分子上
```

你应当看到 `mobile_start` 远在空间一侧，而 `mobile_aligned` 恰好叠在 `ref` 之上。

---

## 9. 可以探索的方向

1. **改变 `radius_scale`**（0.5 对比 1.0）：Tanimoto 矩阵如何变化？
2. **仅重原子**形状（`include_hydrogens=False`）：更快，且往往在化学上更有意义。
3. 对齐器中的 **`n_starts`**：最少多少次重启仍能恢复良好的对齐？
4. 同一分子的**不同构象**：此时最优 Tanimoto 会小于 1；相比刚体位移，这是更贴近
   实际的形状匹配测试。
5. **二阶重叠**（对三重高斯乘积做容斥修正）：这里使用的一阶成对求和在密集区域会
   略微高估重叠。试着修正它。

---

## 10. 参考文献

- Grant, J. A.; Pickett, S. D. *A Gaussian Description of Molecular Shape.*
  J. Phys. Chem. 1995, 99, 3503-3510.
- Grant, J. A.; Gallardo, M. A.; Pickett, S. D. *A fast method of molecular shape
  comparison.* J. Comput. Chem. 1996, 17, 1653-1666. (ROCS.)
- Bondi, A. *van der Waals Volumes and Radii.* J. Phys. Chem. 1964, 68, 441-451.
