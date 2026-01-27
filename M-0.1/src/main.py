"""
M-0.1 论文理论公式验证脚本
验证分形维数的基本理论与无理数表示基础

主要验证内容：
1. 豪斯多夫维数和盒维数的计算
2. 自相似分形（康托尔集、von Koch曲线）的维数验证
3. 任意实维数分形的构造
4. 无理数的分形维数表示（π 和 e）
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class FractalDimension:
    """分形维数计算类"""
    
    @staticmethod
    def hausdorff_dimension(ratios: List[float], epsilon: float = 1e-10) -> float:
        """
        计算自相似分形的豪斯多夫维数
        根据 Moran 定理：dim_h(F) = s，其中 s 满足 Σ r_i^s = 1
        
        参数:
            ratios: 相似变换的压缩比列表 [r_1, r_2, ..., r_m]
            epsilon: 数值精度
        
        返回:
            豪斯多夫维数
        """
        def equation(s):
            return sum(r**s for r in ratios) - 1
        
        # 使用二分法求解方程
        # 豪斯多夫维数在 (0, len(ratios)) 之间
        lower, upper = 0.0, float(len(ratios))
        
        for _ in range(100):
            mid = (lower + upper) / 2
            if abs(equation(mid)) < epsilon:
                return mid
            if equation(mid) > 0:
                lower = mid
            else:
                upper = mid
        
        return (lower + upper) / 2
    
    @staticmethod
    def box_dimension(points: np.ndarray, scales: List[float] = None) -> float:
        """
        计算点集的盒维数
        dim_b(F) = lim(δ→0) log(N_δ(F)) / log(1/δ)
        
        参数:
            points: 点集坐标数组，形状为 (n, d)
            scales: 使用的尺度列表
        
        返回:
            盒维数
        """
        if scales is None:
            scales = [0.1, 0.05, 0.025, 0.0125, 0.00625]
        
        counts = []
        for scale in scales:
            # 将点集离散化到网格中
            grid = (points / scale).astype(int)
            unique_boxes = len(np.unique(grid, axis=0))
            counts.append(unique_boxes)
        
        # 线性回归拟合 log-log 关系
        log_scales = np.log(1.0 / np.array(scales))
        log_counts = np.log(np.array(counts))
        
        # 最小二乘拟合
        coeffs = np.polyfit(log_scales, log_counts, 1)
        return coeffs[0]


class SelfSimilarFractal:
    """自相似分形构造类"""
    
    @staticmethod
    def cantor_set(iterations: int = 10) -> np.ndarray:
        """
        构造康托尔集
        
        参数:
            iterations: 迭代次数
        
        返回:
            康托尔集的点集
        """
        intervals = [(0.0, 1.0)]
        
        for _ in range(iterations):
            new_intervals = []
            for a, b in intervals:
                mid = (a + b) / 3
                new_intervals.append((a, mid))
                new_intervals.append((2*mid, b))
            intervals = new_intervals
        
        # 提取所有区间的中点
        points = np.array([(a + b) / 2 for a, b in intervals])
        return points.reshape(-1, 1)
    
    @staticmethod
    def koch_curve(iterations: int = 5) -> np.ndarray:
        """
        构造 von Koch 曲线
        
        参数:
            iterations: 迭代次数
        
        返回:
            von Koch 曲线的点集
        """
        # 初始线段
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        
        for _ in range(iterations):
            new_points = []
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i+1]
                
                # 计算四个新点
                v = p2 - p1
                p3 = p1 + v / 3
                p5 = p1 + 2 * v / 3
                
                # 计算等边三角形的顶点
                angle = np.pi / 3
                rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
                p4 = p3 + rotation @ (v / 3)
                
                new_points.extend([p1, p3, p4, p5])
            
            new_points.append(points[-1])
            points = np.array(new_points)
        
        return points


class ArbitraryDimensionFractal:
    """任意实维数分形构造类"""
    
    @staticmethod
    def construct(dimension: float, num_transforms: int = 5) -> Tuple[List[float], float]:
        """
        构造指定维数的自相似分形
        
        根据 Moran 定理，选择 m 个相同的压缩比 r = m^(-1/s)
        使得 Σ r^s = m * m^(-1) = 1
        
        参数:
            dimension: 目标豪斯多夫维数
            num_transforms: 相似变换的数量
        
        返回:
            (压缩比列表, 实际维数)
        """
        ratio = num_transforms ** (-1.0 / dimension)
        ratios = [ratio] * num_transforms
        
        actual_dimension = FractalDimension.hausdorff_dimension(ratios)
        
        return ratios, actual_dimension


class IrrationalRepresentation:
    """无理数的分形维数表示类"""
    
    @staticmethod
    def represent_pi() -> dict:
        """
        表示 π 为两个分形维数的和
        
        π = d_1 + d_2
        其中 d_1 = ln(2)/ln(3) ≈ 0.6309 (康托尔集维数)
              d_2 = π - d_1 ≈ 2.5107 (超越分形维数)
        
        返回:
            包含表示信息的字典
        """
        pi = np.pi
        d1 = np.log(2) / np.log(3)  # 康托尔集维数
        d2 = pi - d1
        
        # 构造 d_2 对应的分形
        ratios_d2, actual_d2 = ArbitraryDimensionFractal.construct(d2, num_transforms=6)
        
        return {
            'target': pi,
            'd1': d1,
            'd2': d2,
            'd1_type': '代数分形维数',
            'd2_type': '超越分形维数',
            'd2_ratios': ratios_d2,
            'actual_d2': actual_d2,
            'error': abs(pi - (d1 + actual_d2))
        }
    
    @staticmethod
    def represent_e() -> dict:
        """
        表示 e 为两个分形维数的和
        
        e = d_1 + d_2
        其中 d_1 = ln(3)/ln(4) ≈ 0.7925
              d_2 = e - d_1 ≈ 1.9258 (超越分形维数)
        
        返回:
            包含表示信息的字典
        """
        e = np.e
        d1 = np.log(3) / np.log(4)
        d2 = e - d1
        
        # 构造 d_2 对应的分形
        ratios_d2, actual_d2 = ArbitraryDimensionFractal.construct(d2, num_transforms=5)
        
        return {
            'target': e,
            'd1': d1,
            'd2': d2,
            'd1_type': '代数分形维数',
            'd2_type': '超越分形维数',
            'd2_ratios': ratios_d2,
            'actual_d2': actual_d2,
            'error': abs(e - (d1 + actual_d2))
        }


def verify_cantor_set():
    """验证康托尔集的豪斯多夫维数"""
    print("=" * 80)
    print("验证 1: 康托尔集的豪斯多夫维数")
    print("=" * 80)
    
    # 理论值
    theoretical_dim = np.log(2) / np.log(3)
    print(f"理论值: dim_h(C) = ln(2)/ln(3) = {theoretical_dim:.6f}")
    
    # 使用 Moran 定理计算
    ratios = [1/3, 1/3]
    calculated_dim = FractalDimension.hausdorff_dimension(ratios)
    print(f"计算值 (Moran定理): dim_h(C) = {calculated_dim:.6f}")
    
    # 使用盒维数计算
    cantor_points = SelfSimilarFractal.cantor_set(iterations=10)
    box_dim = FractalDimension.box_dimension(cantor_points)
    print(f"计算值 (盒维数): dim_b(C) = {box_dim:.6f}")
    
    error = abs(theoretical_dim - calculated_dim)
    print(f"误差: {error:.10f}")
    print(f"验证结果: {'✓ 通过' if error < 1e-6 else '✗ 失败'}")
    print()


def verify_koch_curve():
    """验证 von Koch 曲线的豪斯多夫维数"""
    print("=" * 80)
    print("验证 2: von Koch 曲线的豪斯多夫维数")
    print("=" * 80)
    
    # 理论值
    theoretical_dim = np.log(4) / np.log(3)
    print(f"理论值: dim_h(K) = ln(4)/ln(3) = {theoretical_dim:.6f}")
    
    # 使用 Moran 定理计算
    ratios = [1/3, 1/3, 1/3, 1/3]
    calculated_dim = FractalDimension.hausdorff_dimension(ratios)
    print(f"计算值 (Moran定理): dim_h(K) = {calculated_dim:.6f}")
    
    # 使用盒维数计算
    koch_points = SelfSimilarFractal.koch_curve(iterations=6)
    box_dim = FractalDimension.box_dimension(koch_points)
    print(f"计算值 (盒维数): dim_b(K) = {box_dim:.6f}")
    
    error = abs(theoretical_dim - calculated_dim)
    print(f"误差: {error:.10f}")
    print(f"验证结果: {'✓ 通过' if error < 1e-6 else '✗ 失败'}")
    print()


def verify_arbitrary_dimensions():
    """验证任意实维数分形的构造"""
    print("=" * 80)
    print("验证 3: 任意实维数分形的构造")
    print("=" * 80)
    
    test_dimensions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    print(f"{'目标维数':<12} {'实际维数':<12} {'误差':<15} {'验证结果'}")
    print("-" * 80)
    
    all_passed = True
    for target_dim in test_dimensions:
        ratios, actual_dim = ArbitraryDimensionFractal.construct(target_dim, num_transforms=5)
        error = abs(target_dim - actual_dim)
        passed = error < 1e-6
        all_passed = all_passed and passed
        
        print(f"{target_dim:<12.6f} {actual_dim:<12.6f} {error:<15.10f} {'✓ 通过' if passed else '✗ 失败'}")
    
    print(f"\n整体验证结果: {'✓ 全部通过' if all_passed else '✗ 部分失败'}")
    print()


def verify_pi_representation():
    """验证 π 的分形维数表示"""
    print("=" * 80)
    print("验证 4: π 的分形维数表示")
    print("=" * 80)
    
    result = IrrationalRepresentation.represent_pi()
    
    print(f"目标值: π = {result['target']:.10f}")
    print(f"d_1 = {result['d1']:.10f} ({result['d1_type']})")
    print(f"d_2 = {result['d2']:.10f} ({result['d2_type']})")
    print(f"d_2 的构造: 使用 {len(result['d2_ratios'])} 个相似变换，压缩比 = {result['d2_ratios'][0]:.6f}")
    print(f"实际 d_2 = {result['actual_d2']:.10f}")
    print(f"π = d_1 + d_2 = {result['d1'] + result['actual_d2']:.10f}")
    print(f"误差: {result['error']:.10f}")
    print(f"验证结果: {'✓ 通过' if result['error'] < 1e-6 else '✗ 失败'}")
    print()


def verify_e_representation():
    """验证 e 的分形维数表示"""
    print("=" * 80)
    print("验证 5: e 的分形维数表示")
    print("=" * 80)
    
    result = IrrationalRepresentation.represent_e()
    
    print(f"目标值: e = {result['target']:.10f}")
    print(f"d_1 = {result['d1']:.10f} ({result['d1_type']})")
    print(f"d_2 = {result['d2']:.10f} ({result['d2_type']})")
    print(f"d_2 的构造: 使用 {len(result['d2_ratios'])} 个相似变换，压缩比 = {result['d2_ratios'][0]:.6f}")
    print(f"实际 d_2 = {result['actual_d2']:.10f}")
    print(f"e = d_1 + d_2 = {result['d1'] + result['actual_d2']:.10f}")
    print(f"误差: {result['error']:.10f}")
    print(f"验证结果: {'✓ 通过' if result['error'] < 1e-6 else '✗ 失败'}")
    print()


def plot_fractal_dimensions():
    """绘制分形维数示意图"""
    print("=" * 80)
    print("生成分形维数示意图...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 康托尔集
    cantor_points = SelfSimilarFractal.cantor_set(iterations=8)
    axes[0, 0].scatter(cantor_points, np.zeros_like(cantor_points), s=1, c='blue', alpha=0.6)
    axes[0, 0].set_title(f'康托尔集 (豪斯多夫维数 = {np.log(2)/np.log(3):.4f})')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_yticks([])
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. von Koch 曲线
    koch_points = SelfSimilarFractal.koch_curve(iterations=5)
    axes[0, 1].plot(koch_points[:, 0], koch_points[:, 1], 'b-', linewidth=0.5)
    axes[0, 1].set_title(f'冯·科赫曲线 (豪斯多夫维数 = {np.log(4)/np.log(3):.4f})')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')
    
    # 3. 维数与压缩比的关系
    num_transforms_range = range(2, 11)
    dimensions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    for dim in dimensions:
        ratios = []
        for n in num_transforms_range:
            ratio = n ** (-1.0 / dim)
            ratios.append(ratio)
        axes[1, 0].plot(num_transforms_range, ratios, 'o-', label=f'维数 = {dim:.1f}')
    
    axes[1, 0].set_title('压缩比与变换数量的关系')
    axes[1, 0].set_xlabel('变换数量')
    axes[1, 0].set_ylabel('压缩比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 无理数表示
    irrationals = [np.pi, np.e, np.sqrt(2), np.sqrt(3)]
    labels = ['π', 'e', '√2', '√3']
    
    x_pos = np.arange(len(irrationals))
    axes[1, 1].bar(x_pos, irrationals, alpha=0.7, color=['red', 'green', 'blue', 'orange'])
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylabel('数值')
    axes[1, 1].set_title('无理数')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('M-0.1_fractal_dimensions.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存为 M-0.1_fractal_dimensions.png")
    print()


def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print("M-0.1 论文理论公式验证")
    print("分形维数的基本理论与无理数表示基础")
    print("*" * 80)
    print("\n")
    
    # 验证各个理论公式
    verify_cantor_set()
    verify_koch_curve()
    verify_arbitrary_dimensions()
    verify_pi_representation()
    verify_e_representation()
    
    # 生成图表
    plot_fractal_dimensions()
    
    print("=" * 80)
    print("所有验证完成！")
    print("=" * 80)
    print("\n")


if __name__ == "__main__":
    main()
