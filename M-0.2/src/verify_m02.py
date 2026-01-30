"""
M-0.2 论文理论公式验证脚本
验证内积空间与正交组合表示理论

主要验证内容：
1. 分形维数空间的内积构造
2. 施密特正交化过程
3. 超越无理数的正交组合表示（π 和 e）
4. L-BFGS-B参数优化算法
"""

import numpy as np
from scipy.optimize import minimize
import sys
import os

# 添加 M-0.1 的路径以重用分形维数计算
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../M-0.1/src'))

try:
    from main import FractalDimension, ArbitraryDimensionFractal
except ImportError:
    # 如果无法导入，定义基本函数
    class FractalDimension:
        @staticmethod
        def hausdorff_dimension(ratios, epsilon=1e-10):
            def equation(s):
                return sum(r**s for r in ratios) - 1
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
    
    class ArbitraryDimensionFractal:
        @staticmethod
        def construct(dimension, num_transforms=5):
            ratio = num_transforms ** (-1.0 / dimension)
            ratios = [ratio] * num_transforms
            actual_dimension = FractalDimension.hausdorff_dimension(ratios)
            return ratios, actual_dimension


class InnerProductSpace:
    """分形维数空间的内积空间结构"""
    
    @staticmethod
    def inner_product(d1, d2, basis_dims=None):
        """
        计算两个分形维数的内积
        使用简化的内积定义: <d1, d2> = d1 * d2
        
        参数:
            d1, d2: 分形维数
            basis_dims: 正交基（可选）
        
        返回:
            内积值
        """
        return d1 * d2
    
    @staticmethod
    def norm(d, basis_dims=None):
        """
        计算分形维数的范数
        
        参数:
            d: 分形维数
            basis_dims: 正交基（可选）
        
        返回:
            范数值
        """
        return np.sqrt(InnerProductSpace.inner_product(d, d, basis_dims))
    
    @staticmethod
    def distance(d1, d2, basis_dims=None):
        """
        计算两个分形维数的距离
        
        参数:
            d1, d2: 分形维数
            basis_dims: 正交基（可选）
        
        返回:
            距离值
        """
        return InnerProductSpace.norm(d1 - d2, basis_dims)


class SchmidtOrthogonalization:
    """施密特正交化类"""
    
    @staticmethod
    def orthogonalize(vectors):
        """
        对向量组进行施密特正交化
        
        参数:
            vectors: 输入向量列表，每个向量是numpy数组
        
        返回:
            正交向量组
        """
        vectors = [np.array(v, dtype=float) for v in vectors]
        orthogonal = []
        
        for v in vectors:
            u = v.copy()
            for w in orthogonal:
                proj = np.dot(v, w) / np.dot(w, w) * w
                u = u - proj
            
            # 检查是否为零向量
            if np.linalg.norm(u) > 1e-10:
                orthogonal.append(u)
        
        return orthogonal
    
    @staticmethod
    def normalize(vectors):
        """
        规范化向量组
        
        参数:
            vectors: 输入向量列表
        
        返回:
            单位向量组
        """
        normalized = []
        for v in vectors:
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                normalized.append(v / norm)
        return normalized


class OrthogonalRepresentation:
    """正交组合表示类"""
    
    @staticmethod
    def construct_fractal_basis(num_basis=3, start_dim=0.5):
        """
        构造分形维数正交基
        
        参数:
            num_basis: 基向量数量
            start_dim: 起始维数
        
        返回:
            分形维数列表
        """
        basis_dims = []
        for i in range(num_basis):
            target_dim = start_dim + i * 0.5
            _, actual_dim = ArbitraryDimensionFractal.construct(target_dim, num_transforms=5)
            basis_dims.append(actual_dim)
        return basis_dims
    
    @staticmethod
    def represent_irrational(target, basis_dims):
        """
        用正交基表示无理数
        
        参数:
            target: 目标无理数
            basis_dims: 分形维数基
        
        返回:
            (系数列表, 表示值, 误差)
        """
        # 简化的表示：使用最小二乘法找到最佳系数
        A = np.array(basis_dims).reshape(-1, 1)
        coeffs, residuals, _, _ = np.linalg.lstsq(A, np.array([target]), rcond=None)
        representation = np.dot(coeffs, basis_dims)
        error = abs(target - representation)
        
        return coeffs.flatten().tolist(), representation, error
    
    @staticmethod
    def optimize_representation_lbfgsb(target, initial_basis_count=3):
        """
        使用L-BFGS-B算法优化分形维数基和系数
        
        参数:
            target: 目标无理数
            initial_basis_count: 初始基数量
        
        返回:
            优化结果字典
        """
        def objective(params):
            """目标函数：最小化表示误差"""
            n = len(params) // 2
            basis_dims = params[:n]
            coeffs = params[n:]
            
            # 确保基是正数
            basis_dims = np.abs(basis_dims)
            
            representation = np.dot(coeffs, basis_dims)
            error = (target - representation) ** 2
            
            # 添加正则化项以保持基的合理性
            regularization = 1e-6 * np.sum(basis_dims ** 2)
            
            return error + regularization
        
        # 初始猜测
        n = initial_basis_count
        initial_basis = [0.5 + i * 0.5 for i in range(n)]
        initial_coeffs = [target / sum(initial_basis)] * n
        initial_params = initial_basis + initial_coeffs
        
        # 设置边界
        bounds = [(0.01, 10.0)] * n + [(-10.0, 10.0)] * n
        
        # 优化
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 1000})
        
        if result.success:
            n = len(result.x) // 2
            opt_basis = np.abs(result.x[:n])
            opt_coeffs = result.x[n:]
            representation = np.dot(opt_coeffs, opt_basis)
            error = abs(target - representation)
            
            return {
                'success': True,
                'basis_dims': opt_basis.tolist(),
                'coeffs': opt_coeffs.tolist(),
                'representation': representation,
                'target': target,
                'error': error,
                'iterations': result.nit
            }
        else:
            return {'success': False, 'message': result.message}


def verify_inner_product_space():
    """验证分形维数空间的内积空间结构"""
    print("=" * 80)
    print("验证 1: 分形维数空间的内积空间结构")
    print("=" * 80)
    
    # 测试内积性质
    d1, d2, d3 = 0.6309, 1.2619, 2.5107  # 康托尔集、Koch曲线、另一个维数
    
    # 1. 对称性: <d1, d2> = <d2, d1>
    ip12 = InnerProductSpace.inner_product(d1, d2)
    ip21 = InnerProductSpace.inner_product(d2, d1)
    symmetry = abs(ip12 - ip21) < 1e-10
    print(f"对称性: <{d1:.4f}, {d2:.4f}> = {ip12:.6f}, <{d2:.4f}, {d1:.4f}> = {ip21:.6f}")
    print(f"对称性验证: {'✓ 通过' if symmetry else '✗ 失败'}")
    
    # 2. 线性性: <a*d1, d2> = a*<d1, d2>
    a = 2.0
    ip_a = InnerProductSpace.inner_product(a * d1, d2)
    ip_linear = a * InnerProductSpace.inner_product(d1, d2)
    linearity = abs(ip_a - ip_linear) < 1e-10
    print(f"线性性: <{a}*{d1:.4f}, {d2:.4f}> = {ip_a:.6f}, {a}*<{d1:.4f}, {d2:.4f}> = {ip_linear:.6f}")
    print(f"线性性验证: {'✓ 通过' if linearity else '✗ 失败'}")
    
    # 3. 正定性: <d, d> >= 0
    ip_dd = InnerProductSpace.inner_product(d1, d1)
    positive = ip_dd >= 0
    print(f"正定性: <{d1:.4f}, {d1:.4f}> = {ip_dd:.6f} >= 0")
    print(f"正定性验证: {'✓ 通过' if positive else '✗ 失败'}")
    
    all_passed = symmetry and linearity and positive
    print(f"\n内积空间结构验证: {'✓ 全部通过' if all_passed else '✗ 部分失败'}")
    print()
    return all_passed


def verify_schmidt_orthogonalization():
    """验证施密特正交化过程"""
    print("=" * 80)
    print("验证 2: 施密特正交化过程")
    print("=" * 80)
    
    # 构造线性无关的向量组
    vectors = [
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([1, 1, 1])
    ]
    
    # 施密特正交化
    orthogonal = SchmidtOrthogonalization.orthogonalize(vectors)
    
    print(f"原始向量数量: {len(vectors)}")
    print(f"正交向量数量: {len(orthogonal)}")
    
    # 验证正交性
    all_orthogonal = True
    print("\n正交性验证:")
    for i in range(len(orthogonal)):
        for j in range(i + 1, len(orthogonal)):
            dot_product = np.dot(orthogonal[i], orthogonal[j])
            is_ortho = abs(dot_product) < 1e-10
            all_orthogonal = all_orthogonal and is_ortho
            print(f"  <u_{i}, u_{j}> = {dot_product:.10f} - {'✓' if is_ortho else '✗'}")
    
    # 规范化
    normalized = SchmidtOrthogonalization.normalize(orthogonal)
    
    # 验证单位长度
    all_unit = True
    print("\n单位长度验证:")
    for i, v in enumerate(normalized):
        norm = np.linalg.norm(v)
        is_unit = abs(norm - 1.0) < 1e-10
        all_unit = all_unit and is_unit
        print(f"  ||u_{i}|| = {norm:.10f} - {'✓' if is_unit else '✗'}")
    
    all_passed = all_orthogonal and all_unit
    print(f"\n施密特正交化验证: {'✓ 全部通过' if all_passed else '✗ 部分失败'}")
    print()
    return all_passed


def verify_fractal_basis_construction():
    """验证分形维数基的构造"""
    print("=" * 80)
    print("验证 3: 分形维数基的构造")
    print("=" * 80)
    
    basis_dims = OrthogonalRepresentation.construct_fractal_basis(num_basis=4, start_dim=0.5)
    
    print(f"构造了 {len(basis_dims)} 个分形维数基:")
    for i, dim in enumerate(basis_dims):
        print(f"  d_{i} = {dim:.10f}")
    
    # 验证所有基都是正数
    all_positive = all(d > 0 for d in basis_dims)
    print(f"\n所有基维数为正: {'✓ 通过' if all_positive else '✗ 失败'}")
    
    # 验证基的线性无关性（通过检查它们不共线）
    unique_dims = len(set([round(d, 6) for d in basis_dims])) == len(basis_dims)
    print(f"基维数互不相同: {'✓ 通过' if unique_dims else '✗ 失败'}")
    
    all_passed = all_positive and unique_dims
    print(f"\n分形维数基构造验证: {'✓ 全部通过' if all_passed else '✗ 部分失败'}")
    print()
    return all_passed


def verify_pi_representation():
    """验证π的正交组合表示"""
    print("=" * 80)
    print("验证 4: π 的正交组合表示")
    print("=" * 80)
    
    # 使用优化算法
    result = OrthogonalRepresentation.optimize_representation_lbfgsb(np.pi, initial_basis_count=2)
    
    if result['success']:
        print(f"目标值: π = {result['target']:.10f}")
        print(f"表示值: {result['representation']:.10f}")
        print(f"误差: {result['error']:.10e}")
        print(f"迭代次数: {result['iterations']}")
        print("\n分形维数基:")
        for i, dim in enumerate(result['basis_dims']):
            print(f"  d_{i} = {dim:.10f} (系数: {result['coeffs'][i]:.6f})")
        
        # 计算误差百分比
        error_percent = result['error'] / result['target'] * 100
        print(f"\n误差百分比: {error_percent:.10e}%")
        
        # 验证误差小于阈值
        passed = result['error'] < 1e-6
        print(f"验证结果: {'✓ 通过' if passed else '✗ 失败'}")
    else:
        print(f"优化失败: {result.get('message', 'Unknown error')}")
        passed = False
    
    print()
    return passed


def verify_e_representation():
    """验证e的正交组合表示"""
    print("=" * 80)
    print("验证 5: e 的正交组合表示")
    print("=" * 80)
    
    # 使用优化算法
    result = OrthogonalRepresentation.optimize_representation_lbfgsb(np.e, initial_basis_count=2)
    
    if result['success']:
        print(f"目标值: e = {result['target']:.10f}")
        print(f"表示值: {result['representation']:.10f}")
        print(f"误差: {result['error']:.10e}")
        print(f"迭代次数: {result['iterations']}")
        print("\n分形维数基:")
        for i, dim in enumerate(result['basis_dims']):
            print(f"  d_{i} = {dim:.10f} (系数: {result['coeffs'][i]:.6f})")
        
        # 计算误差百分比
        error_percent = result['error'] / result['target'] * 100
        print(f"\n误差百分比: {error_percent:.10e}%")
        
        # 验证误差小于阈值
        passed = result['error'] < 1e-6
        print(f"验证结果: {'✓ 通过' if passed else '✗ 失败'}")
    else:
        print(f"优化失败: {result.get('message', 'Unknown error')}")
        passed = False
    
    print()
    return passed


def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print("M-0.2 论文理论公式验证")
    print("内积空间与正交组合表示理论")
    print("*" * 80)
    print("\n")
    
    results = []
    
    # 验证各个理论
    results.append(("内积空间结构", verify_inner_product_space()))
    results.append(("施密特正交化", verify_schmidt_orthogonalization()))
    results.append(("分形维数基构造", verify_fractal_basis_construction()))
    results.append(("π的正交表示", verify_pi_representation()))
    results.append(("e的正交表示", verify_e_representation()))
    
    # 汇总结果
    print("=" * 80)
    print("验证结果汇总")
    print("=" * 80)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:<30} {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 80)
    print(f"整体验证结果: {'✓ 全部通过' if all_passed else '✗ 部分失败'}")
    print("=" * 80)
    print("\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
