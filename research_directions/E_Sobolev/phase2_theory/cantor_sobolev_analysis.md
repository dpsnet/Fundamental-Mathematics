# Cantor 集上的 Sobolev 空间分析

## 目标
对 middle-third Cantor 集 $C$ 显式分析 $W^{k,p}(C)$。

---

## 1. Cantor 集的性质

### 1.1 基本参数
- **相似比**: $r = 1/3$
- **复制数**: $N = 2$
- **Hausdorff 维数**: $d = \log 2 / \log 3$
- **测度**: $H^d(C) = 1$ (归一化)

### 1.2 结构
$C$ 是自相似集：
$$C = f_1(C) \cup f_2(C)$$
其中 $f_1(x) = x/3$, $f_2(x) = x/3 + 2/3$。

---

## 2. $W^{k,p}(C)$ 的特征

### 2.1 等价范数
根据 Jonsson-Wallin，对于 $f \in W^{k,p}(C)$：
$$\|f\|_{W^{k,p}(C)}^p \sim \|f\|_{L^p(C)}^p + \sum_{n=0}^\infty 3^{nkp} \sum_{I \in \mathcal{C}_n} \int_I |f - f_I|^p dH^d$$

其中：
- $\mathcal{C}_n$: 第 $n$ 层 Cantor 区间
- $f_I = \frac{1}{H^d(I)} \int_I f dH^d$

### 2.2 简化形式
由于 $C$ 的全不连通性：
$$W^{k,p}(C) \cong \{f \in L^p(C) : \sum_{n,j} 3^{nkp} |f_{n+1,j} - f_{n,i}|^p r^{nd} < \infty\}$$

其中 $f_{n,i}$ 是 $f$ 在第 $n$ 层第 $i$ 个区间的平均值。

---

## 3. 与 M-0.2 的联系

### 3.1 M-0.2 的直觉
M-0.2 声称可以"正交化"分形上的函数。

### 3.2 严格解释
**正交化 = Sobolev 空间中的 Gram-Schmidt 过程**

在 $W^{k,2}(C)$ 中，可以构造正交基：
$$\phi_n(x) = \text{分形小波基}$$

这与 Jonsson-Wallin 的框架一致。

---

## 4. 计算目标

### 4.1 数值实验
计算简单函数（如特征函数、多项式限制）的 $W^{k,p}(C)$ 范数。

### 4.2 验证延拓定理
构造具体的延拓算子 $E: W^{k,p}(C) \to W^{k,p}(\mathbb{R})$。

---

**状态**: Phase 2 进行中
