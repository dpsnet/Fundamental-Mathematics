# Sierpinski 垫上的 Sobolev 空间

## 二维分形的深入分析

---

## 1. Sierpinski 垫的基本性质

### 1.1 构造
顶点为 $v_0, v_1, v_2$ 的等边三角形。

迭代：$S = \bigcap_{n=0}^\infty S_n$，其中 $S_{n+1} = \bigcup_{i=0}^2 f_i(S_n)$

相似变换：$f_i(x) = \frac{1}{2}(x + v_i)$

### 1.2 分形参数
| 参数 | 值 |
|------|-----|
| 相似比 | $r = 1/2$ |
| 复制数 | $N = 3$ |
| Hausdorff 维数 | $d_H = \frac{\log 3}{\log 2} \approx 1.585$ |
| 谱维数 | $d_s = \frac{2\log 3}{\log 5} \approx 1.365$ |

---

## 2. Kigami 构造的狄利克雷形式

### 2.1 图逼近
在第 $n$ 层，$S$ 被 $V_n$（$3 + 3 \cdot 3 + \cdots + 3^n = \frac{3}{2}(3^n - 1)$ 个点）逼近。

定义图拉普拉斯：
$$\mathcal{E}_n(f) = \sum_{x \sim_n y} (f(x) - f(y))^2$$

### 2.2 重归一化
关键：狄利克雷形式需要重归一化：
$$\mathcal{E}(f) = \lim_{n \to \infty} \left(\frac{5}{3}\right)^n \mathcal{E}_n(f)$$

因子 $(5/3)^n$ 使极限有限且非零。

### 2.3 分形拉普拉斯算子
通过 $\mathcal{E}(f,g) = -\int_S f \Delta_\mu g \, d\mu$ 定义。

谱渐近：
$$N(\lambda) \sim \lambda^{d_s/2} = \lambda^{\log 3 / \log 5}$$

---

## 3. Jonsson-Wallin 框架的应用

### 3.1 环境空间
将 $S \subset \mathbb{R}^2$，环境维数 $n=2$。

### 3.2 Sobolev 空间定义
$W^{s,p}(S)$ 使用 $d_H$-维 Hausdorff 测度。

### 3.3 迹定理表述
对于 $s > (2 - d_H)/p$：
$$\gamma: W^{s,p}(\mathbb{R}^2) \to W^{s-(2-d_H)/p,p}(S)$$

具体数值：
- $(2 - d_H) = 2 - \log 3/\log 2 \approx 0.415$
- 对于 $p=2, s=1$：正则性损失约 $0.21$

### 3.4 延拓算子
构造 $E: W^{s-(2-d_H)/p,p}(S) \to W^{s,p}(\mathbb{R}^2)$

**Whitney 分解**: 对 $S$ 的余集（可数多三角形）进行分解。

---

## 4. 与 Kigami 理论的联系

### 4.1 两种框架
| 框架 | 主要对象 | 优势 |
|------|----------|------|
| Kigami | 狄利克雷形式 $\mathcal{E}$ | 直接定义拉普拉斯算子 |
| Jonsson-Wallin | Sobolev 空间 $W^{s,p}$ | 与经典分析兼容 |

### 4.2 对应关系
**猜想**: 对于 $p=2$：
$$\text{Dom}(\mathcal{E}) \cong W^{1,2}(S)$$

这需要验证范数等价性。

---

## 5. 计算框架

### 5.1 有限元逼近
在第 $n$ 层，用分段线性函数逼近：
- 节点：$V_n$
- 基函数：$\phi_i^{(n)}$

### 5.2 离散 Sobolev 范数
$$\|f\|_{W^{s,2}_n}^2 = \sum_{k=0}^n 4^{ks} \mathcal{E}_k(f)$$

### 5.3 延拓的数值实现
对于 $f$ 在 $V_n$ 上的值：
1. 延拓到第 $n$ 层三角形内部（线性插值）
2. 定义外部 Whitney 区域上的值
3. 单位分解平滑

---

## 6. 与 B 方向的联系

### 6.1 维度参数化
Sierpinski 垫族（变化压缩比）：
- 固定 $N=3$
- 变化 $r \in (0, 1/2]$
- $d(r) = \frac{\log 3}{\log(1/r)}$

### 6.2 流方程
设 $r = e^{-t}$，则：
$$\frac{\partial u}{\partial t} = \mathcal{L}_t u$$

其中 $\mathcal{L}_t$ 是随 $t$ 变化的算子。

---

## 7. 研究问题

### 7.1 理论问题
1. Kigami 的 $\mathcal{E}$ 与 JW 的 $W^{1,2}$ 是否等价？
2. 对于 $p \neq 2$ 的 Sobolev 空间如何定义？
3. 谱维数 $d_s$ 与 Hausdorff 维数 $d_H$ 在函数空间中的角色

### 7.2 计算问题
1. 实现延拓算子的数值验证
2. 计算具体函数的范数
3. 验证迹定理的常数

---

## 8. 下一步工作

- [ ] 证明 $\mathcal{E}$ 与 $W^{1,2}$ 范数等价
- [ ] 实现数值延拓算子
- [ ] 与 Kigami 的谱理论对接

---

**状态**: Phase 2 - 深入分析中
