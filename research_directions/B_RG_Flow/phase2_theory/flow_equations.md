# 维度流方程的严格数学框架

## 从函数空间到动力学系统

---

## 1. 引言

### 1.1 目标
建立随维度 $d$ 演化的函数空间动力学：
$$\frac{\partial u}{\partial d} = \mathcal{L}_d u$$

其中 $u(d, \cdot) \in \mathcal{H}_d$ 是随维度变化的函数。

### 1.2 与 RG 流的类比
| Wilson RG | 维度流 |
|-----------|--------|
| 尺度 $x \to \lambda x$ | 维数 $d \to d + \delta d$ |
| 耦合常数 $g$ | 函数 $u$ |
| $\beta$ 函数 | 算子 $\mathcal{L}_d$ |
| 固定点 $g^*$ | 稳定维数 $d^*$ |

---

## 2. 函数空间族

### 2.1 定义
设 $\{\mathcal{H}_d\}_{d \in [0,1]}$ 是 Hilbert 空间族：
- $\mathcal{H}_d = W^{s,2}(C_d)$，其中 $C_d$ 是维数 $d$ 的 Cantor 集
- 内积 $(\cdot, \cdot)_d$ 依赖于 $d$

### 2.2 同构结构
**假设**: 存在同构 $\Phi_d: \mathcal{H}_0 \to \mathcal{H}_d$。

通过 $\Phi_d$，将问题转化到固定空间 $\mathcal{H}_0$。

### 2.3 拉回度量
在 $\mathcal{H}_0$ 上定义：
$$(u, v)_{\mathcal{H}_d} = (\Phi_d u, \Phi_d v)_{\mathcal{H}_d}$$

这给出了参数化的内积族。

---

## 3. 流方程的推导

### 3.1 从延拓算子出发
回顾 E 方向的延拓算子：
$$E_{d_1, d_2}: \mathcal{H}_{d_1} \to \mathcal{H}_{d_2}$$

对于小量 $\epsilon$：
$$E_{d, d+\epsilon} = I + \epsilon \cdot L_d + O(\epsilon^2)$$

### 3.2 无穷小生成元
定义算子 $L_d: \mathcal{H}_d \to \mathcal{H}_d$：
$$L_d = \lim_{\epsilon \to 0} \frac{E_{d, d+\epsilon} - I}{\epsilon}$$

### 3.3 流方程
函数 $u(d)$ 的演化：
$$u(d+\epsilon) = E_{d, d+\epsilon} u(d) \approx (I + \epsilon L_d) u(d)$$

取极限：
$$\frac{du}{dd} = L_d u$$

---

## 4. 算子 $L_d$ 的显式形式

### 4.1 热核表示
对于热方程，维数流与热核相关。

在 $d$ 维空间中，热核：
$$K_d(t, x) = (4\pi t)^{-d/2} \exp\left(-\frac{|x|^2}{4t}\right)$$

### 4.2 对数导数
$$\frac{\partial K_d}{\partial d} = -\frac{1}{2} \log(4\pi t) \cdot K_d$$

### 4.3 流算子
这提示：
$$L_d = -\frac{1}{2} \log(4\pi \Delta^{-1})$$

其中 $\Delta$ 是拉普拉斯算子。

### 4.4 谱表示
在特征基 $\{\phi_k\}$ 上：
$$(L_d u)(x) = -\frac{1}{2} \sum_k \log(4\pi / \lambda_k) \cdot \hat{u}_k \phi_k(x)$$

其中 $\hat{u}_k = (u, \phi_k)$。

---

## 5. 固定点分析

### 5.1 固定点的定义
$u^*$ 是固定点如果：
$$L_d u^* = 0$$

### 5.2 平凡固定点
**常数函数**: $u(x) = c$

- $\Delta u = 0$
- $L_d u = 0$（在适当的正则化下）

### 5.3 非平凡固定点
寻找满足 $L_d u = 0$ 的非零 $u$。

**问题**: 对于热核流，是否存在非平凡固定点？

**答案**: 可能需要修改流方程的定义。

### 5.4 修正的流方程
引入势函数 $V(d)$：
$$\frac{du}{dd} = L_d u - \frac{dV}{dd} u$$

**固定点条件**:
$$L_d u^* = \frac{dV}{dd}(d^*) u^*$$

即 $u^*$ 是 $L_d$ 的特征函数。

---

## 6. 线性稳定性分析

### 6.1 在固定点附近
设 $u = u^* + \delta u$，线性化：
$$\frac{d(\delta u)}{dd} = L_d \delta u - \lambda^* \delta u$$

其中 $\lambda^* = \frac{dV}{dd}(d^*)$。

### 6.2 特征值问题
设 $\delta u = e^{\mu d} v$：
$$L_d v = (\lambda^* + \mu) v$$

稳定性由 $\mu$ 的符号决定：
- $\mu < 0$: 稳定
- $\mu > 0$: 不稳定

### 6.3 临界维数
如果 $d^*$ 是稳定固定点，则它是"临界维数"。

---

## 7. 具体例子：Cantor 集族

### 7.1 构造
设 $C_r$ 是相似比 $r$ 的 Cantor 集。

维数：
$$d(r) = \frac{\log 2}{\log(1/r)}$$

参数 $t = \log(1/r)$，则 $d = (\log 2) / t$。

### 7.2 关于 $t$ 的流
$$\frac{du}{dt} = -\frac{\log 2}{t^2} \frac{du}{dd}$$

### 7.3 数值实验
对于 $u_t \in L^2(C_{r(t)})$，追踪：
$$E(t) = \|u_t\|_{L^2(C_{r(t)})}^2$$

**问题**: $E(t)$ 如何演化？

**猜想**: 对于特定初值，$E(t)$ 有幂律行为。

---

## 8. 数学严格性

### 8.1 存在性
**定理**: 对于光滑初值 $u_0 \in \mathcal{H}_{d_0}$，流方程存在唯一解。

**证明要点**:
1. 将流方程视为 Banach 空间中的 ODE
2. 应用 Picard-Lindelöf 定理
3. 验证 $L_d$ 的 Lipschitz 连续性

### 8.2 正则性
解 $u(d)$ 关于 $d$ 的光滑性：
- 如果 $L_d$ 光滑依赖于 $d$，则 $u(d)$ 光滑

### 8.3 长时间行为
当 $d \to d_{\max}$ 或 $d \to d_{\min}$：
- 解是否收敛到固定点？
- 还是发散？

---

## 9. 与物理的联系

### 9.1 维度正规化
QFT 中的维度正规化：
- 在 $d = 4 - \epsilon$ 维计算
- 取 $\epsilon \to 0$

我们的框架提供了数学基础：
- 函数空间随 $d$ 的插值
- 算子在维度间的延拓

### 9.2 有效场论
低能有效理论：
- 高能自由度被积分掉
- 有效维数可能降低

维度流描述了这种"积分掉"的过程。

---

## 10. 下一步工作

- [ ] 证明 $L_d$ 的显式公式
- [ ] 数值求解流方程
- [ ] 寻找非平凡固定点
- [ ] 与 Kigami 的分形拉普拉斯联系

---

**状态**: Phase 2-3 - 流方程框架构建中

**依赖**: E 方向的迹算子结果
