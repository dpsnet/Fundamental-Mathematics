# 重整化群流理论基础

## 为分形函数方程做准备

---

## 1. Wilson 重整化群回顾

### 1.1 基本思想
在统计物理中，重整化群 (RG) 描述系统在不同尺度下的行为：
$$R_b: \{\text{耦合常数}\} \to \{\text{耦合常数}\}$$

固定点对应于普适类：
$$R_b(K^*) = K^*$$

### 1.2 线性化与临界指数
在固定点附近：
$$R_b(K^* + \delta K) \approx K^* + L_b \cdot \delta K$$

特征值 $\lambda_i = b^{y_i}$ 给出临界指数。

---

## 2. 与 M-0 的联系

### 2.1 M-0 的声明
M-0 声称："维度可以重整化"

### 2.2 严格解释
这不是新的物理，而是函数空间的插值理论：
- 从整数维 $d$ 的函数空间出发
- 通过解析延拓定义非整数维空间
- 尺度变换由重标度算子实现

**关键**: 这不改变物理维度，只是数学上的函数空间插值。

---

## 3. 函数方程视角

### 3.1 维度重标度
设 $T_\epsilon: L^2(\mathbb{R}^d) \to L^2(\mathbb{R}^{d+\epsilon})$ 是某种插值算子。

函数方程：
$$\phi_{d+\epsilon} = T_\epsilon(\phi_d) + \epsilon \cdot N(\phi_d)$$

其中 $N$ 是"相互作用项"。

### 3.2 与迹定理的联系
E 方向的延拓算子 $E$ 提供了自然的候选：
$$T_\epsilon = E_{d,\epsilon} \circ \gamma_d$$

其中：
- $\gamma_d$: 从 $d+\epsilon$ 维到 $d$ 维分形的迹
- $E_{d,\epsilon}$: 从 $d$ 维分形到 $d+\epsilon$ 维的延拓

### 3.3 不动点方程
寻找 $\phi^*$ 使得：
$$\phi^* = T_\epsilon(\phi^*) + \epsilon \cdot N(\phi^*)$$

当 $\epsilon \to 0$ 时，这成为微分方程：
$$\frac{d\phi}{d\epsilon} = \mathcal{L}\phi + N(\phi)$$

---

## 4. 数学严格性分析

### 4.1 困难点
1. **算子定义**: $T_\epsilon$ 的精确数学意义
2. **函数空间**: 需要统一的函数空间框架
3. **解析性**: $\epsilon \to 0$ 的极限是否良好

### 4.2 可能的路径
**方向 1**: 使用谱理论
- Kigami 的分形拉普拉斯算子
- 谱维数 $d_s$ 作为有效维数
- 热核的尺度性质

**方向 2**: 使用插值空间
- Lions-Peetre 实插值方法
- 复插值方法
- 建立维度间的同构

---

## 5. 与 E 方向的依赖关系

### 5.1 E 提供的工具
Jonsson-Wallin 框架给出：
- $W^{s,p}(\mathbb{R}^n) \to W^{s',p}(F)$ 的迹算子
- 延拓算子的范数估计
- 分形 Sobolev 嵌入定理

### 5.2 B 需要的额外工作
1. **参数化**: 将分形维数 $d$ 视为连续参数
2. **插值**: 在固定 $n$ 下，随 $d$ 变化建立空间族
3. **流方程**: 关于 $d$ 的微分方程

---

## 6. 具体研究问题

### 6.1 问题 1: 谱维数流
设 $\Delta_d$ 是 Sierpinski 垫上的分形拉普拉斯算子。

定义谱维数：
$$d_s = \lim_{t \to \infty} \frac{2 \log p(t)}{\log t}$$

其中 $p(t)$ 是热核迹。

**问题**: 能否定义"谱维数空间"中的流？

### 6.2 问题 2: 函数空间插值
对于 $0 < d_1 < d_2 < n$，建立：
$$[W^{s,p}(F_{d_1}), W^{s,p}(F_{d_2})]_\theta = W^{s,p}(F_{d_\theta})$$

其中 $d_\theta = (1-\theta)d_1 + \theta d_2$。

---

## 7. 文献与工具

### 7.1 核心文献
- Cardy (1996): Scaling and Renormalization in Statistical Physics
- Gawedzki & Kupiainen (1986): Asymptotic Freedom
- Kigami (2001): Analysis on Fractals

### 7.2 数学工具
- 插值空间理论 (Lions-Peetre, Calderón)
- 分形上的分析 (Kigami, Strichartz)
- 泛函分析中的算子半群理论

---

## 8. 下一步计划

等待 E 方向完成：
- [ ] 迹定理的完整证明
- [ ] 延拓算子的范数估计
- [ ] Sierpinski 垫的具体分析

然后启动 B 的 Phase 2：
- [ ] 定义维度参数化的函数空间
- [ ] 建立流方程的数学框架
- [ ] 研究固定点结构

---

**状态**: Phase 1 - 等待 E 方向输入
