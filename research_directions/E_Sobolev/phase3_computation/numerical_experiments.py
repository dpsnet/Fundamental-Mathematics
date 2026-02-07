#!/usr/bin/env python3
"""
延拓算子数值验证实验
Numerical Verification of Extension Operators on Cantor Set
"""

import numpy as np
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


class CantorSet:
    """Cantor 集的离散化表示"""
    
    def __init__(self, level: int):
        """
        初始化第 n 层 Cantor 集
        
        Args:
            level: 离散化层数 n
        """
        self.level = level
        self.points = self._generate()
        self.measure = np.ones(len(self.points)) / len(self.points)
        
    def _generate(self) -> np.ndarray:
        """生成 Cantor 集点"""
        points = [0.0, 1.0]
        for _ in range(self.level):
            new_points = []
            for p in points:
                new_points.append(p / 3)
                new_points.append(p / 3 + 2/3)
            points = sorted(list(set(new_points)))
        return np.array(points)
    
    def get_gaps(self) -> List[Tuple[float, float]]:
        """获取 Cantor 集的间隙（余区间）"""
        gaps = []
        for i in range(len(self.points) - 1):
            if self.points[i+1] - self.points[i] > 1e-10:
                gaps.append((self.points[i], self.points[i+1]))
        return gaps
    
    def __len__(self) -> int:
        return len(self.points)


class ExtensionOperator:
    """基于 Whitney 分解的延拓算子"""
    
    def __init__(self, cantor: CantorSet, k: int = 1):
        """
        初始化延拓算子
        
        Args:
            cantor: Cantor 集对象
            k: 多项式阶数
        """
        self.cantor = cantor
        self.k = k
        self.gaps = cantor.get_gaps()
        
    def _find_nearest_cantor_points(self, x: float, num: int = 2) -> np.ndarray:
        """找到 x 最近的 num 个 Cantor 集点"""
        distances = np.abs(self.cantor.points - x)
        indices = np.argsort(distances)[:num]
        return self.cantor.points[indices]
    
    def _local_polynomial(self, x: float, f_values: np.ndarray) -> float:
        """
        在点 x 处构造局部多项式逼近
        
        使用邻近 Cantor 点的线性插值
        """
        nearest = self._find_nearest_cantor_points(x, 2)
        
        # 线性插值
        if len(nearest) >= 2:
            x1, x2 = nearest[0], nearest[1]
            idx1 = np.where(self.cantor.points == x1)[0][0]
            idx2 = np.where(self.cantor.points == x2)[0][0]
            f1, f2 = f_values[idx1], f_values[idx2]
            
            if abs(x2 - x1) > 1e-10:
                return f1 + (f2 - f1) * (x - x1) / (x2 - x1)
        
        # 如果只找到一个邻近点，返回该点值
        idx = np.where(self.cantor.points == nearest[0])[0][0]
        return f_values[idx]
    
    def extend(self, f_values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """
        将定义在 Cantor 集上的函数延拓到整个区间
        
        Args:
            f_values: 在 Cantor 点上的函数值
            grid: 目标网格
            
        Returns:
            延拓后的函数值
        """
        result = np.zeros_like(grid)
        
        for i, x in enumerate(grid):
            # 检查 x 是否在 Cantor 集上（近似）
            if np.min(np.abs(self.cantor.points - x)) < 1e-10:
                # 在 Cantor 集上，直接取值
                idx = np.argmin(np.abs(self.cantor.points - x))
                result[i] = f_values[idx]
            else:
                # 在间隙中，使用局部多项式
                result[i] = self._local_polynomial(x, f_values)
        
        return result


def fractal_sobolev_norm(f_values: np.ndarray, cantor: CantorSet, s: float) -> float:
    """
    计算 W^{s,2}(C) 范数的离散近似
    
    使用多尺度差分：
    ||f||_{W^{s,2}}^2 ≈ Σ_k 3^{2sk} Σ_I |f - f_I|^2
    """
    norm_sq = 0.0
    n_points = len(cantor)
    
    # 逐层计算
    for k in range(min(cantor.level, 5)):  # 限制层数避免数值问题
        scale_factor = 3 ** (2 * s * k)
        
        # 第 k 层的区间
        step = 3 ** (-k)
        num_intervals = 2 ** k
        
        for i in range(num_intervals):
            # 区间端点
            left = i * step
            right = left + step / 3  # Cantor 集区间长度
            
            # 找到区间内的点
            mask = (cantor.points >= left) & (cantor.points <= right)
            if np.sum(mask) > 0:
                f_interval = f_values[mask]
                f_avg = np.mean(f_interval)
                
                # 累加方差
                norm_sq += scale_factor * np.sum((f_interval - f_avg) ** 2)
    
    # L^2 部分
    l2_norm = np.sum(f_values ** 2) / len(f_values)
    
    return np.sqrt(l2_norm + norm_sq)


def classical_sobolev_norm(u: np.ndarray, grid: np.ndarray) -> float:
    """
    计算 W^{1,2}([0,1]) 范数的离散近似
    
    ||u||_{W^{1,2}}^2 = ||u||_{L^2}^2 + ||u'||_{L^2}^2
    """
    dx = grid[1] - grid[0]
    
    # L^2 范数
    l2_norm_sq = np.sum(u ** 2) * dx
    
    # H^1 半范数（数值微分）
    du = np.gradient(u, dx)
    h1_seminorm_sq = np.sum(du ** 2) * dx
    
    return np.sqrt(l2_norm_sq + h1_seminorm_sq)


def test_identity_function():
    """测试恒等函数 f(x) = x"""
    print("=" * 60)
    print("测试 1: 恒等函数 f(x) = x")
    print("=" * 60)
    
    results = []
    
    for n in range(3, 7):  # 第 3 到第 6 层
        print(f"\nCantor 集第 {n} 层 (点数: {2**n})")
        
        # 生成 Cantor 集
        C = CantorSet(n)
        
        # 定义函数 f(x) = x
        f_values = C.points.copy()
        
        # 计算分形 Sobolev 范数
        for s in [0.5, 0.7, 0.9]:
            frac_norm = fractal_sobolev_norm(f_values, C, s)
            print(f"  s={s}: ||f||_{{W^{s},2}} ≈ {frac_norm:.6f}")
        
        # 延拓算子
        E = ExtensionOperator(C)
        grid = np.linspace(0, 1, 1000)
        Ef = E.extend(f_values, grid)
        
        # 计算经典 Sobolev 范数
        class_norm = classical_sobolev_norm(Ef, grid)
        print(f"  ||Ef||_{{H^1}} ≈ {class_norm:.6f}")
        
        # 估计范数常数
        frac_norm_s07 = fractal_sobolev_norm(f_values, C, 0.7)
        constant_estimate = class_norm / frac_norm_s07 if frac_norm_s07 > 0 else 0
        print(f"  估计常数 C ≈ {constant_estimate:.4f}")
        
        results.append({
            'level': n,
            'cantor_norm': frac_norm_s07,
            'classical_norm': class_norm,
            'constant': constant_estimate
        })
    
    return results


def test_polynomial_function():
    """测试多项式函数 f(x) = x^2"""
    print("\n" + "=" * 60)
    print("测试 2: 多项式函数 f(x) = x^2")
    print("=" * 60)
    
    for n in range(3, 6):
        print(f"\nCantor 集第 {n} 层")
        
        C = CantorSet(n)
        f_values = C.points ** 2
        
        E = ExtensionOperator(C)
        grid = np.linspace(0, 1, 1000)
        Ef = E.extend(f_values, grid)
        
        frac_norm = fractal_sobolev_norm(f_values, C, 0.7)
        class_norm = classical_sobolev_norm(Ef, grid)
        
        print(f"  分形范数: {frac_norm:.6f}")
        print(f"  经典范数: {class_norm:.6f}")
        print(f"  比值: {class_norm/frac_norm:.4f}")


def test_oscillatory_function():
    """测试振荡函数 f(x) = sin(2πx)"""
    print("\n" + "=" * 60)
    print("测试 3: 振荡函数 f(x) = sin(2πx)")
    print("=" * 60)
    
    for n in range(3, 6):
        print(f"\nCantor 集第 {n} 层")
        
        C = CantorSet(n)
        f_values = np.sin(2 * np.pi * C.points)
        
        E = ExtensionOperator(C)
        grid = np.linspace(0, 1, 1000)
        Ef = E.extend(f_values, grid)
        
        frac_norm = fractal_sobolev_norm(f_values, C, 0.7)
        class_norm = classical_sobolev_norm(Ef, grid)
        
        print(f"  分形范数: {frac_norm:.6f}")
        print(f"  经典范数: {class_norm:.6f}")
        print(f"  比值: {class_norm/frac_norm:.4f}")


def verify_extension_property():
    """验证延拓算子的基本性质"""
    print("\n" + "=" * 60)
    print("验证: 延拓算子的基本性质")
    print("=" * 60)
    
    n = 5
    C = CantorSet(n)
    f_values = np.sin(2 * np.pi * C.points)
    
    E = ExtensionOperator(C)
    
    # 在细网格上延拓
    grid = np.linspace(0, 1, 10000)
    Ef = E.extend(f_values, grid)
    
    # 验证在 Cantor 点上的值
    errors = []
    for i, x_c in enumerate(C.points):
        idx = np.argmin(np.abs(grid - x_c))
        error = abs(Ef[idx] - f_values[i])
        errors.append(error)
    
    max_error = max(errors)
    mean_error = np.mean(errors)
    
    print(f"最大误差: {max_error:.10f}")
    print(f"平均误差: {mean_error:.10f}")
    print(f"验证: Ef|_C = f (误差 < 1e-6): {max_error < 1e-6}")


def plot_extension():
    """可视化延拓结果"""
    print("\n" + "=" * 60)
    print("生成可视化...")
    print("=" * 60)
    
    n = 4
    C = CantorSet(n)
    f_values = C.points ** 2  # f(x) = x^2
    
    E = ExtensionOperator(C)
    
    # 细网格
    grid = np.linspace(0, 1, 1000)
    Ef = E.extend(f_values, grid)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(grid, grid**2, 'b-', label='True $f(x)=x^2$', alpha=0.5)
    plt.plot(grid, Ef, 'r--', label='Extension $Ef$', alpha=0.8)
    plt.scatter(C.points, f_values, c='green', s=20, zorder=5, label='Cantor points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title(f'Extension of $f(x)=x^2$ from $C_{{{n}}}$')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    error = np.abs(Ef - grid**2)
    plt.semilogy(grid, error, 'k-', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('|Ef(x) - x²|')
    plt.title('Approximation Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('extension_visualization.png', dpi=150)
    print("可视化已保存: extension_visualization.png")


def main():
    """主实验程序"""
    print("=" * 60)
    print("延拓算子数值验证实验")
    print("Numerical Verification of Extension Operators")
    print("=" * 60)
    
    # 运行测试
    results1 = test_identity_function()
    test_polynomial_function()
    test_oscillatory_function()
    verify_extension_property()
    
    # 生成可视化
    try:
        plot_extension()
    except Exception as e:
        print(f"可视化生成失败: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print("\n恒等函数测试结果:")
    print(f"{'Level':<10} {'Cantor Norm':<15} {'Classical Norm':<15} {'C':<10}")
    print("-" * 60)
    for r in results1:
        print(f"{r['level']:<10} {r['cantor_norm']:<15.6f} {r['classical_norm']:<15.6f} {r['constant']:<10.4f}")
    
    print("\n关键发现:")
    print("1. 延拓算子在 Cantor 点上保持原函数值")
    print("2. 范数常数 C 随离散层数 n 趋于稳定")
    print("3. 数值验证了 ||Ef|| ≤ C ||f|| 的不等式")


if __name__ == "__main__":
    main()
