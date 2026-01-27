import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

class SpectralDimensionAnalysis:
    """
    谱维流动分析模块
    """
    
    def __init__(self):
        self.pi = np.pi
        
        # 定义分形维数基
        self.e1 = np.log(7) / np.log(8)
        self.e2 = 2.0
        self.q0 = 1.0
    
    def compute_ramanujan_spectral_dimension(self, k):
        """
        计算拉马努金公式的谱维流动
        """
        # 计算第k项的收敛速度
        R_k = (factorial(4*k) * (1103 + 26390*k)) / (factorial(k)**4 * 396**(4*k))
        
        # 谱维度与收敛速度的对数相关
        spectral_dim = np.log(abs(R_k)) / np.log(k + 1) if k > 0 else 0
        
        return spectral_dim, R_k
    
    def compute_fractal_spectral_dimension(self, l):
        """
        计算统一分形维数表达式的谱维流动
        """
        # 使用π的表示
        lambda1 = 0.8743
        lambda2 = 0.7294
        lambda3 = 0.8647
        
        # 计算第l次迭代的逼近
        approximation = lambda1 * self.e1 + lambda2 * self.e2 + lambda3 * self.q0
        error = abs(approximation - self.pi)
        
        # 谱维度与误差的对数相关
        spectral_dim = -np.log(error) / np.log(l + 1) if l > 0 else 0
        
        return spectral_dim, error
    
    def analyze_ramanujan_flow(self, max_k=20):
        """
        分析拉马努金谱维流动
        """
        print("=" * 80)
        print("拉马努金谱维流动分析")
        print("=" * 80)
        
        k_values = list(range(1, max_k + 1))
        spectral_dims = []
        convergence_rates = []
        
        for k in k_values:
            spectral_dim, R_k = self.compute_ramanujan_spectral_dimension(k)
            spectral_dims.append(spectral_dim)
            convergence_rates.append(abs(R_k))
            
            print(f"k = {k:2d}: 谱维 = {spectral_dim:10.4f}, 收敛速度 = {abs(R_k):.6e}")
        
        print()
        print("单调性分析:")
        monotonic = all(spectral_dims[i] <= spectral_dims[i+1] for i in range(len(spectral_dims)-1))
        print(f"  单调递增: {monotonic}")
        
        print()
        print("有效范围分析:")
        valid_range = all(-10 < d < 10 for d in spectral_dims)
        print(f"  在[-10, 10]范围内: {valid_range}")
        print(f"  最小值: {min(spectral_dims):.4f}")
        print(f"  最大值: {max(spectral_dims):.4f}")
        
        return k_values, spectral_dims, convergence_rates
    
    def analyze_fractal_flow(self, max_l=20):
        """
        分析分形谱维流动
        """
        print()
        print("=" * 80)
        print("分形谱维流动分析")
        print("=" * 80)
        
        l_values = list(range(1, max_l + 1))
        spectral_dims = []
        errors = []
        
        for l in l_values:
            spectral_dim, error = self.compute_fractal_spectral_dimension(l)
            spectral_dims.append(spectral_dim)
            errors.append(error)
            
            print(f"l = {l:2d}: 谱维 = {spectral_dim:10.4f}, 误差 = {error:.6e}")
        
        print()
        print("单调性分析:")
        monotonic = all(spectral_dims[i] <= spectral_dims[i+1] for i in range(len(spectral_dims)-1))
        print(f"  单调递增: {monotonic}")
        
        print()
        print("有效范围分析:")
        valid_range = all(-10 < d < 10 for d in spectral_dims)
        print(f"  在[-10, 10]范围内: {valid_range}")
        print(f"  最小值: {min(spectral_dims):.4f}")
        print(f"  最大值: {max(spectral_dims):.4f}")
        
        return l_values, spectral_dims, errors
    
    def analyze_correspondence(self, max_n=10):
        """
        分析两种方法的谱维流动对应
        """
        print()
        print("=" * 80)
        print("谱维流动对应分析")
        print("=" * 80)
        
        n_values = list(range(1, max_n + 1))
        ramanujan_dims = []
        fractal_dims = []
        differences = []
        
        for n in n_values:
            ramanujan_dim, _ = self.compute_ramanujan_spectral_dimension(n)
            fractal_dim, _ = self.compute_fractal_spectral_dimension(n)
            
            ramanujan_dims.append(ramanujan_dim)
            fractal_dims.append(fractal_dim)
            differences.append(abs(ramanujan_dim - fractal_dim))
            
            print(f"n = {n:2d}: 拉马努金谱维 = {ramanujan_dim:10.4f}, 分形谱维 = {fractal_dim:10.4f}, 差异 = {abs(ramanujan_dim - fractal_dim):10.4f}")
        
        print()
        print("相似性分析:")
        mean_diff = np.mean(differences)
        similarity = mean_diff < 1.0
        print(f"  平均差异: {mean_diff:.4f}")
        print(f"  阈值(1.0)内: {similarity}")
        print(f"  最小差异: {min(differences):.4f}")
        print(f"  最大差异: {max(differences):.4f}")
        
        return n_values, ramanujan_dims, fractal_dims, differences
    
    def plot_comparison(self):
        """
        绘制谱维流动对比图
        """
        k_values, ramanujan_dims, _ = self.analyze_ramanujan_flow(20)
        l_values, fractal_dims, _ = self.analyze_fractal_flow(20)
        n_values, ramanujan_compare, fractal_compare, differences = self.analyze_correspondence(10)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 拉马努金谱维流动
        axes[0, 0].plot(k_values, ramanujan_dims, 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('迭代次数 k', fontsize=12)
        axes[0, 0].set_ylabel('谱维度', fontsize=12)
        axes[0, 0].set_title('拉马努金谱维流动', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 分形谱维流动
        axes[0, 1].plot(l_values, fractal_dims, 'r-s', linewidth=2, markersize=4)
        axes[0, 1].set_xlabel('迭代次数 l', fontsize=12)
        axes[0, 1].set_ylabel('谱维度', fontsize=12)
        axes[0, 1].set_title('分形谱维流动', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 两种方法对比
        axes[1, 0].plot(n_values, ramanujan_compare, 'b-o', label='拉马努金', linewidth=2, markersize=4)
        axes[1, 0].plot(n_values, fractal_compare, 'r-s', label='分形', linewidth=2, markersize=4)
        axes[1, 0].set_xlabel('迭代次数 n', fontsize=12)
        axes[1, 0].set_ylabel('谱维度', fontsize=12)
        axes[1, 0].set_title('两种方法谱维流动对比', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 差异分析
        axes[1, 1].plot(n_values, differences, 'g-^', linewidth=2, markersize=4)
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='阈值(1.0)')
        axes[1, 1].set_xlabel('迭代次数 n', fontsize=12)
        axes[1, 1].set_ylabel('差异', fontsize=12)
        axes[1, 1].set_title('谱维流动差异分析', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('spectral_dimension_analysis.png', dpi=300, bbox_inches='tight')
        print()
        print("=" * 80)
        print("图表已保存为: spectral_dimension_analysis.png")
        print("=" * 80)
        
        return fig

def main():
    analyzer = SpectralDimensionAnalysis()
    
    # 执行分析
    analyzer.analyze_ramanujan_flow(20)
    analyzer.analyze_fractal_flow(20)
    analyzer.analyze_correspondence(10)
    
    # 绘制对比图
    fig = analyzer.plot_comparison()
    
    # 总结
    print()
    print("=" * 80)
    print("总结")
    print("=" * 80)
    print("验证4失败的原因分析：")
    print()
    print("1. 谱维定义验证失败:")
    print("   - 拉马努金谱维流动的值超出了[-10, 10]的合理范围")
    print("   - 这是因为拉马努金公式的收敛速度极快，导致对数计算产生极大的负值")
    print()
    print("2. 拉马努金谱维流动单调性验证失败:")
    print("   - 谱维流动不是单调递增的")
    print("   - 这可能是因为谱维的计算方法不正确")
    print()
    print("3. 分形谱维流动单调性验证失败:")
    print("   - 分形谱维流动不是单调递增的")
    print("   - 这可能是因为分形表达式的误差计算方法不正确")
    print()
    print("4. 谱维流动对应验证失败:")
    print("   - 两种方法的谱维流动差异很大（平均差异54.5441）")
    print("   - 这表明两种方法的谱维计算方法可能存在根本性的差异")
    print()
    print("根本原因:")
    print("  - 谱维的计算方法可能不正确")
    print("  - 需要重新审视谱维的定义和计算公式")
    print("  - 可能需要使用不同的谱维计算方法")
    print("=" * 80)

if __name__ == "__main__":
    main()
