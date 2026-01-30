"""
严格标准验证脚本
使用标准化的阈值进行理论验证
"""

import numpy as np
import sys

# ==================== 标准验证阈值配置 ====================
VERIFICATION_THRESHOLDS = {
    # A级 - 严格验证 (核心数学恒等式)
    'strict_math': {
        'epsilon': 1e-12,
        'description': '核心数学恒等式'
    },
    
    # B级 - 标准验证 (数值计算)
    'standard_numerical': {
        'epsilon': 1e-9,
        'description': '数值计算标准'
    },
    
    # C级 - 工程验证 (近似方法)
    'engineering': {
        'epsilon': 1e-6,
        'description': '工程应用精度'
    },
    
    # D级 - 趋势验证 (渐近性质)
    'trend': {
        'relative_error': 0.05,
        'ratio_range': (0.95, 1.05),
        'description': '渐近趋势验证'
    },
    
    # 特殊验证 - 素数定理
    'prime_theorem': {
        'ratio_range': (0.95, 1.05),
        'strict_ratio_range': (0.98, 1.02),  # 用于n > 1000
        'description': '素数定理验证'
    },
    
    # 特殊验证 - 谱维流动
    'spectral_dimension': {
        'absolute_error': 0.01,
        'description': '谱维计算精度'
    }
}


class VerificationReporter:
    """验证报告生成器"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, module, test_name, passed, actual_value, threshold, threshold_type):
        """添加验证结果"""
        self.results.append({
            'module': module,
            'test_name': test_name,
            'passed': passed,
            'actual_value': actual_value,
            'threshold': threshold,
            'threshold_type': threshold_type
        })
        
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name:<40} {status}")
        print(f"    实际值: {actual_value}")
        print(f"    阈值标准: {threshold} ({threshold_type})")
    
    def get_summary(self):
        """获取验证摘要"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        return passed, total


# ==================== M-0.4 素数生成函数验证 ====================

def sieve_of_eratosthenes(n):
    """埃拉托斯特尼筛法生成素数"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]


def verify_m04_strict():
    """
    M-0.4: 素数生成函数验证 (严格标准)
    【改进】使用更严格的素数定理比值阈值
    """
    print("=" * 80)
    print("M-0.4: 素数生成函数与模形式验证 (严格标准)")
    print("=" * 80)
    
    reporter = VerificationReporter()
    
    # 1. 素数定理验证 - 使用严格阈值
    print("\n1. 素数定理验证 (严格阈值: 0.95-1.05)")
    print("-" * 60)
    
    test_values = [100, 1000, 10000]
    thresholds = VERIFICATION_THRESHOLDS['prime_theorem']
    
    for n in test_values:
        primes = sieve_of_eratosthenes(n)
        pi_n = len(primes)
        li_n = n / np.log(n)
        ratio = pi_n / li_n
        
        # 对于较大的n使用更严格的阈值
        if n > 1000:
            min_ratio, max_ratio = thresholds['strict_ratio_range']
        else:
            min_ratio, max_ratio = thresholds['ratio_range']
        
        passed = min_ratio < ratio < max_ratio
        
        reporter.add_result(
            'M-0.4', 
            f'素数定理 n={n}', 
            passed, 
            f"π(n)/nln(n) = {ratio:.6f} (阈值: [{min_ratio}, {max_ratio}])",
            f"[{min_ratio}, {max_ratio}]",
            'prime_theorem'
        )
    
    # 2. 素数间隔分布验证
    print("\n2. 素数间隔分布验证")
    print("-" * 60)
    
    n = 1000
    primes = sieve_of_eratosthenes(n)
    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    avg_gap = np.mean(gaps)
    expected_gap = np.log(n)
    gap_ratio = avg_gap / expected_gap
    
    # 使用D级阈值 (0.7-1.3)
    min_gap_ratio, max_gap_ratio = 0.7, 1.3
    gap_passed = min_gap_ratio < gap_ratio < max_gap_ratio
    
    reporter.add_result(
        'M-0.4',
        '素数间隔分布',
        gap_passed,
        f"实际/期望 = {gap_ratio:.4f}",
        f"[{min_gap_ratio}, {max_gap_ratio}]",
        'trend'
    )
    
    passed, total = reporter.get_summary()
    print(f"\nM-0.4 严格验证结果: {passed}/{total} 通过")
    print()
    return passed == total


# ==================== M-0.5 快速收敛算法验证 ====================

def verify_m05_strict():
    """
    M-0.5: 快速收敛算法验证 (严格标准)
    """
    print("=" * 80)
    print("M-0.5: 快速收敛算法验证 (严格标准)")
    print("=" * 80)
    
    reporter = VerificationReporter()
    
    # 1. 拉马努金公式收敛速度验证
    print("\n1. 收敛速度验证 (阈值: 改进比 > 1e6)")
    print("-" * 60)
    
    def ramanujan_pi_term(k):
        import math
        k_factorial = math.factorial(4*k) * (1103 + 26390*k)
        k_denominator = (math.factorial(k)**4) * (396**(4*k))
        return k_factorial / k_denominator
    
    sum_value = 0
    errors = []
    for k in range(3):
        term = ramanujan_pi_term(k)
        sum_value += term
        pi_approx = 9801 / (np.sqrt(8) * sum_value)
        error = abs(np.pi - pi_approx)
        errors.append(error)
    
    improvement = errors[0] / errors[1] if errors[1] > 0 else 0
    threshold = 1e6
    conv_passed = improvement > threshold
    
    reporter.add_result(
        'M-0.5',
        '收敛速度',
        conv_passed,
        f"误差改进比 = {improvement:.2e}",
        f"> {threshold:.0e}",
        'standard_numerical'
    )
    
    # 2. 谱维流动单调性 - 严格验证
    print("\n2. 谱维流动单调性验证 (阈值: 严格单调)")
    print("-" * 60)
    
    def spectral_flow(k):
        return 4.0 - 0.1 * k
    
    k_values = range(10)
    flow_values = [spectral_flow(k) for k in k_values]
    
    # 严格单调递减验证
    monotonic = all(flow_values[i] > flow_values[i+1] for i in range(len(flow_values)-1))
    
    reporter.add_result(
        'M-0.5',
        '谱维单调性',
        monotonic,
        f"序列: {[f'{v:.2f}' for v in flow_values[:5]]}...",
        "严格单调递减",
        'strict_math'
    )
    
    passed, total = reporter.get_summary()
    print(f"\nM-0.5 严格验证结果: {passed}/{total} 通过")
    print()
    return passed == total


# ==================== M-0.7 谱维流动理论验证 ====================

def verify_m07_strict():
    """
    M-0.7: 谱维流动理论验证 (严格标准)
    【改进】使用更严格的谱维计算阈值 (0.01 而非 0.1)
    """
    print("=" * 80)
    print("M-0.7: 谱维流动理论验证 (严格标准)")
    print("=" * 80)
    
    reporter = VerificationReporter()
    
    # 1. 谱维公式验证 - 严格阈值
    print("\n1. 谱维公式验证 (阈值: 误差 < 0.01)")
    print("-" * 60)
    
    def spectral_dim(d_H, theta):
        return 2 * d_H / (1 + theta)
    
    test_cases = [
        (1.0, 0.0, 2.0),
        (0.63, 0.5, 0.84),
        (1.26, 0.3, 1.9385),
        (2.0, 0.5, 2.6667),
    ]
    
    threshold = VERIFICATION_THRESHOLDS['spectral_dimension']['absolute_error']
    
    for d_H, theta, expected in test_cases:
        d_s = spectral_dim(d_H, theta)
        error = abs(d_s - expected)
        passed = error < threshold
        
        reporter.add_result(
            'M-0.7',
            f'谱维 d_H={d_H}, θ={theta}',
            passed,
            f"计算={d_s:.6f}, 期望={expected}, 误差={error:.6f}",
            f"< {threshold}",
            'spectral_dimension'
        )
    
    # 2. 单调性验证
    print("\n2. 谱维流动单调性验证")
    print("-" * 60)
    
    theta_range = np.linspace(0, 1, 11)
    d_H = 1.0
    d_s_values = [spectral_dim(d_H, theta) for theta in theta_range]
    
    monotonic = all(d_s_values[i] >= d_s_values[i+1] for i in range(len(d_s_values)-1))
    
    reporter.add_result(
        'M-0.7',
        '谱维单调性',
        monotonic,
        f"θ=0: {d_s_values[0]:.4f}, θ=1: {d_s_values[-1]:.4f}",
        "单调递减",
        'strict_math'
    )
    
    passed, total = reporter.get_summary()
    print(f"\nM-0.7 严格验证结果: {passed}/{total} 通过")
    print()
    return passed == total


# ==================== M-0.8 Minkowski时空验证 ====================

def verify_m08_strict():
    """
    M-0.8: Minkowski时空验证 (严格标准)
    """
    print("=" * 80)
    print("M-0.8: Minkowski时空验证 (严格标准)")
    print("=" * 80)
    
    reporter = VerificationReporter()
    
    # 1. 度规行列式验证 - A级严格
    print("\n1. 度规行列式验证 (阈值: 误差 < 1e-12)")
    print("-" * 60)
    
    eta = np.diag([-1, 1, 1, 1])
    det_eta = np.linalg.det(eta)
    expected_det = -1.0
    error = abs(det_eta - expected_det)
    
    threshold = VERIFICATION_THRESHOLDS['strict_math']['epsilon']
    det_passed = error < threshold
    
    reporter.add_result(
        'M-0.8',
        '度规行列式',
        det_passed,
        f"det(η) = {det_eta:.15f}, 误差 = {error:.2e}",
        f"< {threshold:.0e}",
        'strict_math'
    )
    
    # 2. Lorentz变换验证
    print("\n2. Lorentz变换验证 (阈值: 误差 < 1e-10)")
    print("-" * 60)
    
    gamma = 1.5
    beta = np.sqrt(1 - 1/gamma**2)
    
    Lambda = np.array([
        [gamma, -gamma*beta, 0, 0],
        [-gamma*beta, gamma, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    eta_transformed = np.dot(Lambda.T, np.dot(eta, Lambda))
    identity_error = np.max(np.abs(eta_transformed - eta))
    
    threshold = VERIFICATION_THRESHOLDS['standard_numerical']['epsilon']
    lorentz_passed = identity_error < threshold
    
    reporter.add_result(
        'M-0.8',
        'Lorentz变换',
        lorentz_passed,
        f"max|Λ^T η Λ - η| = {identity_error:.2e}",
        f"< {threshold:.0e}",
        'standard_numerical'
    )
    
    passed, total = reporter.get_summary()
    print(f"\nM-0.8 严格验证结果: {passed}/{total} 通过")
    print()
    return passed == total


# ==================== M-0.9 PTE问题验证 ====================

def verify_m09_strict():
    """
    M-0.9: PTE问题验证 (严格标准)
    """
    print("=" * 80)
    print("M-0.9: PTE问题验证 (严格标准)")
    print("=" * 80)
    
    reporter = VerificationReporter()
    
    # 1. PTE解幂和验证 - 严格相等
    print("\n1. PTE解幂和验证 (阈值: 精确相等)")
    print("-" * 60)
    
    set1 = [1, 4, 6, 7]
    set2 = [2, 3, 5, 8]
    
    # k=1: 和相等
    sum1, sum2 = sum(set1), sum(set2)
    sum_passed = sum1 == sum2
    
    reporter.add_result(
        'M-0.9',
        'k=1 幂和相等',
        sum_passed,
        f"{sum1} == {sum2}",
        "精确相等",
        'strict_math'
    )
    
    # k=2: 平方和相等
    sum_sq1 = sum(x**2 for x in set1)
    sum_sq2 = sum(x**2 for x in set2)
    sq_passed = sum_sq1 == sum_sq2
    
    reporter.add_result(
        'M-0.9',
        'k=2 平方和相等',
        sq_passed,
        f"{sum_sq1} == {sum_sq2}",
        "精确相等",
        'strict_math'
    )
    
    # 2. 分形维数计算
    print("\n2. 分形维数计算 (阈值: 误差 < 0.01)")
    print("-" * 60)
    
    combined = sorted(set1 + set2)
    unique_points = len(set(combined))
    span = max(combined) - min(combined)
    fractal_dim = np.log(unique_points) / np.log(span + 1)
    expected_dim = 1.0
    error = abs(fractal_dim - expected_dim)
    
    threshold = VERIFICATION_THRESHOLDS['spectral_dimension']['absolute_error']
    dim_passed = error < threshold
    
    reporter.add_result(
        'M-0.9',
        '分形维数',
        dim_passed,
        f"计算={fractal_dim:.6f}, 期望={expected_dim}, 误差={error:.6f}",
        f"< {threshold}",
        'spectral_dimension'
    )
    
    passed, total = reporter.get_summary()
    print(f"\nM-0.9 严格验证结果: {passed}/{total} 通过")
    print()
    return passed == total


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("\n")
    print("*" * 80)
    print("严格标准验证报告")
    print("使用标准化阈值进行理论验证")
    print("*" * 80)
    print("\n")
    
    print("验证阈值配置:")
    print("-" * 60)
    for key, value in VERIFICATION_THRESHOLDS.items():
        if 'epsilon' in value:
            print(f"  {key:<30} ε = {value['epsilon']:.0e} ({value['description']})")
        elif 'ratio_range' in value:
            r = value['ratio_range']
            print(f"  {key:<30} 比值 ∈ [{r[0]}, {r[1]}] ({value['description']})")
        elif 'absolute_error' in value:
            print(f"  {key:<30} 误差 < {value['absolute_error']} ({value['description']})")
    print("\n")
    
    results = []
    
    # 运行各模块验证
    results.append(("M-0.4 素数与模形式", verify_m04_strict()))
    results.append(("M-0.5 快速收敛算法", verify_m05_strict()))
    results.append(("M-0.7 谱维流动理论", verify_m07_strict()))
    results.append(("M-0.8 Minkowski时空", verify_m08_strict()))
    results.append(("M-0.9 PTE问题", verify_m09_strict()))
    
    # 汇总结果
    print("=" * 80)
    print("严格标准验证结果汇总")
    print("=" * 80)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:<35} {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 80)
    print(f"严格标准整体验证结果: {'✓ 全部通过' if all_passed else '✗ 部分失败'}")
    print("=" * 80)
    print("\n")
    
    if not all_passed:
        print("⚠️  警告: 部分验证未通过严格标准，请检查:")
        print("    1. 理论公式是否正确")
        print("    2. 实现代码是否有bug")
        print("    3. 阈值设置是否过于严格")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
