import os
import shutil

def organize_directories():
    base_path = r"e:\FiberGravity-DynamicCoupling\extends\Clifford-Gauge_RheoSpacetime_Unification\统一场理论\系列论文\projects\M-0.3_统一分形维数表达式与拉马努金公式的映射关系"
    
    # 创建v1和v2目录
    v1_dir = os.path.join(base_path, "v1")
    v2_dir = os.path.join(base_path, "v2")
    
    # 创建src和html子目录
    v1_src_dir = os.path.join(v1_dir, "src")
    v1_html_dir = os.path.join(v1_dir, "html")
    v2_src_dir = os.path.join(v2_dir, "src")
    v2_html_dir = os.path.join(v2_dir, "html")
    
    # 创建所有目录
    for dir_path in [v1_dir, v2_dir, v1_src_dir, v1_html_dir, v2_src_dir, v2_html_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("目录结构创建完成：")
    print(f"  {base_path}")
    print(f"    ├── v1/")
    print(f"    │   ├── src/")
    print(f"    │   └── html/")
    print(f"    └── v2/")
    print(f"        ├── src/")
    print(f"        └── html/")
    print()
    
    # 移动v1文件
    v1_src_files = [
        "computational_comparison.py",
        "convergence_analysis.py",
        "modular_space_mapping.py",
        "ramanujan_formula.py",
        "spectral_dimension_flow.py",
        "unified_fractal_expression.py",
        "main.py"
    ]
    
    v1_html_files = [
        "verification_report.html",
        "verification_results.json"
    ]
    
    print("移动v1文件...")
    for file in v1_src_files:
        src_file = os.path.join(base_path, "src", file)
        if os.path.exists(src_file):
            dest_file = os.path.join(v1_src_dir, file)
            shutil.move(src_file, dest_file)
            print(f"  {file} → v1/src/{file}")
    
    for file in v1_html_files:
        src_file = os.path.join(base_path, "html", file)
        if os.path.exists(src_file):
            dest_file = os.path.join(v1_html_dir, file)
            shutil.move(src_file, dest_file)
            print(f"  {file} → v1/html/{file}")
    
    print()
    
    # 移动v2文件
    v2_src_files = [
        "spectral_dimension_flow_v2.py",
        "main_v2.py",
        "generate_html_report_v2.py",
        "redesigned_spectral_dimension_flow.py"
    ]
    
    v2_html_files = [
        "verification_results_v2.json",
        "verification_report_v2.html"
    ]
    
    print("移动v2文件...")
    for file in v2_src_files:
        src_file = os.path.join(base_path, "src", file)
        if os.path.exists(src_file):
            dest_file = os.path.join(v2_src_dir, file)
            shutil.move(src_file, dest_file)
            print(f"  {file} → v2/src/{file}")
    
    for file in v2_html_files:
        src_file = os.path.join(base_path, "html", file)
        if os.path.exists(src_file):
            dest_file = os.path.join(v2_html_dir, file)
            shutil.move(src_file, dest_file)
            print(f"  {file} → v2/html/{file}")
    
    print()
    
    # 移动文档文件
    doc_files = [
        "SPECTRAL_DIMENSION_FLOW_THEORY.md",
        "VERIFICATION4_ANALYSIS_REPORT.md",
        "IMPROVEMENT_PLAN.md",
        "SPECTRAL_DIMENSION_FLOW_SUCCESS_REPORT.md",
        "RESEARCH_SUMMARY.md",
        "V2_SUCCESS_SUMMARY.md"
    ]
    
    print("移动文档文件...")
    for file in doc_files:
        src_file = os.path.join(base_path, "src", file)
        if os.path.exists(src_file):
            # 复制到v1和v2目录
            shutil.copy2(src_file, os.path.join(v1_src_dir, file))
            shutil.copy2(src_file, os.path.join(v2_src_dir, file))
            print(f"  {file} → v1/src/{file}")
            print(f"  {file} → v2/src/{file}")
    
    print()
    
    print("=" * 80)
    print("文件组织完成！")
    print("=" * 80)

if __name__ == "__main__":
    organize_directories()
