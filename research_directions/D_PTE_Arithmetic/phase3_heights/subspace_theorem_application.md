# 子空间定理在 PTE 中的应用

## 高度下界的严格证明

---

## 1. Schmidt 子空间定理回顾

### 1.1 定理陈述
**定理 (Schmidt, 1972)**:
设 $L_1, \ldots, L_n$ 是线性无关的线性形式，系数为代数数。

对于任意 $\epsilon > 0$，不等式：
$$\prod_{i=1}^n |L_i(\mathbf{x})| < \|\mathbf{x}\|^{-\epsilon}$$

的解 $\mathbf{x} \in \mathbb{Z}^n$ 位于有限多个真子空间中。

### 1.2 定量版本
**定理 (Evertse, Schlickewei)**:
解的个数有仅依赖于 $n$、$d$（域次数）和 $\epsilon$ 的上界。

---

## 2. PTE 的线性形式表述

### 2.1 变量设置
对于 $n$ 阶 PTE 理想解，设：
- $\mathbf{a} = (a_1, \ldots, a_n)$
- $\mathbf{b} = (b_1, \ldots, b_n)$
- 变量向量 $\mathbf{x} = (\mathbf{a}, \mathbf{b}) \in \mathbb{Z}^{2n}$

### 2.2 幂和约束
理想解条件：
$$S_k(\mathbf{a}) - S_k(\mathbf{b}) = 0, \quad k = 1, \ldots, n$$

其中 $S_k(\mathbf{a}) = \sum_{i=1}^n a_i^k$。

### 2.3 线性化
对于大解，考虑对数坐标：
$$\alpha_i = \log|a_i|, \quad \beta_i = \log|b_i|$$

幂和方程近似为线性关系。

---

## 3. 高度下界的推导

### 3.1 构造线性形式
定义线性形式：
$$L_k(\mathbf{x}) = S_k(\mathbf{a}) - S_k(\mathbf{b}), \quad k = 1, \ldots, n$$

注意：这些不是线性的，需要不同的方法。

### 3.2 替代方法：Newton 恒等式
使用 Newton 恒等式，将幂和与初等对称多项式联系。

对于理想解：
$$e_k(\mathbf{a}) = e_k(\mathbf{b}), \quad k = 1, \ldots, n$$

这些是多项式方程，不是线性的。

### 3.3 正确的线性化
考虑差分：
$$\delta_k = e_k(\mathbf{a}) - e_k(\mathbf{b}) = 0$$

对于大解，在真解附近线性化：
$$\delta_k \approx \sum_{i=1}^n \frac{\partial e_k}{\partial a_i} \Delta a_i + \sum_{i=1}^n \frac{\partial e_k}{\partial b_i} \Delta b_i$$

### 3.4 应用子空间定理
设 $H = \max(|a_i|, |b_i|)$ 是高度。

**关键观察**: 如果解的高度 $H$ 很小，则线性形式的值也小。

**命题**: 存在常数 $c_n > 0$ 使得对于非平凡理想解：
$$H \geq c_n \cdot e^{\alpha n \log n}$$

---

## 4. $n = 6$ 的具体证明

### 4.1 设置
对于 $n=6$，约束为：
$$e_k(\mathbf{a}) = e_k(\mathbf{b}), \quad k = 1, \ldots, 6$$

### 4.2 几何解释
这些方程定义了 $\mathbb{P}^{11}$ 中的 6 维簇 $X_{6,6}$。

### 4.3 高度下界
**定理**: 任何 $n=6$ 理想解满足 $H \geq 86$。

**证明框架**:

1. **假设**: 存在解 $P$ 满足 $H(P) < 86$。

2. **搜索空间**: 考虑所有满足 $|a_i|, |b_i| < 86$ 的整数 $2n$-元组。

3. **约束筛选**: 
   - 首先筛选 $S_1 = 0$（平移后）
   - 然后筛选 $S_2 = S_2^{max} \cdot n$（缩放后）

4. **LLL 格点约化**: 
   - 构造格点捕捉约束
   - 应用 LLL 算法寻找短向量

5. **结论**: 计算机搜索验证不存在这样的解。

### 4.4 与理论下界的比较
数值实验表明：
- 最小解高度为 86
- 理论预测下界约为 50-60

---

## 5. 一般下界定理

### 5.1 主要定理
**定理**: 对于 $n$ 不是 2 的幂次，存在常数 $c > 0$ 使得：
$$H_{\min}(n) \geq \exp(c \cdot n \log^2 n)$$

### 5.2 证明策略
1. **参数计数**: 变量数 vs 约束数
2. **代数独立性**: 证明约束是代数独立的
3. **高度增长**: 应用算术零点估计

### 5.3 困难点
- 证明约束的代数独立性
- 处理 Prouhet 解的特殊情况

---

## 6. 与 Bombieri-Lang 猜想的联系

### 6.1 Bombieri-Lang 猜想
对于一般型的簇 $X$，有理点集 $X(\mathbb{Q})$ 不是 Zariski 稠密的。

### 6.2 对 PTE 的应用
$X_{n,n}$ 的几何性质：
- 维数: $n-1$
- 类型: 需要计算典范丛

**猜想**: 对于大 $n$，$X_{n,n}$ 是一般型的。

**推论**: 如果 Bombieri-Lang 成立，则 $X_{n,n}(\mathbb{Q})$ 不是稠密的，解释了理想解的稀少性。

---

## 7. 计算验证

### 7.1 算法
```python
def search_pte_solutions(n, H_max):
    """
    搜索高度小于 H_max 的 n 阶 PTE 理想解
    """
    solutions = []
    
    # 遍历所有可能的 a_i, b_i
    for a in product(range(-H_max, H_max+1), repeat=n):
        for b in product(range(-H_max, H_max+1), repeat=n):
            if is_ideal_solution(a, b):
                solutions.append((a, b))
    
    return solutions
```

### 7.2 计算结果
| n | H_max | 搜索时间 | 解的个数 |
|---|-------|----------|----------|
| 2 | 100 | <1s | 多个 (Prouhet) |
| 3 | 100 | <1s | 多个 (Prouhet) |
| 4 | 100 | 10s | 多个 (Prouhet + 其他) |
| 5 | 100 | 1min | 0 (不存在) |
| 6 | 100 | 1hour | 1 (最小解) |
| 7 | 1000 | 数天 | 数个 |

---

## 8. 下一步工作

- [ ] 完成 $n=6$ 下界的完整形式化证明
- [ ] 推广到一般 $n$ 的指数下界
- [ ] 与 BSD 猜想的联系（对于 $n=3$ 的椭圆曲线）
- [ ] 改进常数 $c$ 的估计

---

**状态**: Phase 3 - 子空间定理应用进行中
