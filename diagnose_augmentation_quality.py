"""
诊断GGEUR数据增强质量
检查生成的样本是否与真实样本差异过大
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("GGEUR数据增强质量诊断")
print("="*80)

print("\n根据训练日志分析:")
print("\n观察到的现象:")
print("  1. 数据增强数量正确 ✓")
print("     - Client 1: 380 → 1,250 samples (25 classes × 50)")
print("     - Client 2: 1157 → 1,800 samples (36 classes × 50)")
print("     - Client 3: 982 → 1,550 samples (31 classes × 50)")
print("     - Client 4: 609 → 1,400 samples (28 classes × 50)")
print("\n  2. 训练准确率提升 ✓")
print("     - Round 1: 12.9% → Round 10: 56%")
print("\n  3. 测试准确率很低 ❌")
print("     - Round 1: 1.23% → Round 10: 3.69%")
print("     - 训练/测试差距: 56% - 3.7% = 52.3%")

print("\n" + "="*80)
print("可能的问题")
print("="*80)

print("\n问题1: 生成的样本质量差")
print("  症状: 模型在合成数据上训练良好，但无法泛化到真实数据")
print("  原因:")
print("    - 生成的样本可能偏离真实数据分布太远")
print("    - 协方差矩阵估计不准确（样本太少）")
print("    - 特征值缩放导致方差过小或过大")
print("  检查方法:")
print("    1. 对比原始样本和生成样本的统计特性")
print("    2. 检查协方差矩阵的特征值分布")
print("    3. 可视化原始vs生成样本的分布")

print("\n问题2: LDS导致类别不平衡过于严重")
print("  症状: 某些类别样本太少（0个），无法学习")
print("  从日志看:")
print("    - Client 1: 25个非零类 (40个零样本类)")
print("    - Client 2: 31个非零类 (34个零样本类)")
print("    - Client 3: 28个非零类 (37个零样本类)")
print("    - Client 4: 36个非零类 (29个零样本类)")
print("  问题: 测试集包含所有65个类，但训练时很多类没见过！")
print("  检查方法:")
print("    1. 统计测试集中各类的样本数")
print("    2. 检查测试集是否包含训练时未见过的类")

print("\n问题3: 测试集太小且不平衡")
print("  从日志看:")
print("    - 测试集总共244个样本")
print("    - 65个类 → 平均每类只有 244/65 ≈ 3.75 个样本")
print("  问题: 测试集太小，评估不稳定")

print("\n" + "="*80)
print("关键发现")
print("="*80)

print("\n从日志中的特征值信息:")
print("  Server: Class 0 geometry:")
print("  - Top 5 eigenvalues: [9.999975e-07, 9.999978e-07, 9.99998e-07, 9.99998e-07, 9.999982e-07]")
print("  - Explained var (top 10): 0.0171")
print("\n⚠️  特征值异常小（都接近1e-6）！")
print("   这可能是因为:")
print("   1. cov_regularization: 1e-06 太大，导致特征值被压制")
print("   2. 样本太少，协方差矩阵不稳定")
print("\n   结果: 生成的样本方差太小，都集中在均值附近")
print("   这导致生成的样本缺乏多样性！")

print("\n" + "="*80)
print("建议的修改")
print("="*80)

print("\n修改1: 降低协方差正则化")
print("  当前: cov_regularization: 1e-06")
print("  建议: cov_regularization: 1e-08  # 降低100倍")
print("\n修改2: 关闭特征值缩放（如果影响太大）")
print("  当前: eigenvalue_scaling: True")
print("  建议: eigenvalue_scaling: False  # 先尝试不缩放")

print("\n修改3: 增加每个类的最小样本数（通过调整LDS）")
print("  当前: alpha: 0.1  # 很小，导致很多零样本类")
print("  建议: alpha: 0.5   # 增大，减少零样本类的数量")

print("\n修改4: 使用更保守的增强策略")
print("  当前: target_size_per_class: 50")
print("  建议: target_size_per_class: 30  # 减少生成样本的占比")

print("\n" + "="*80)
print("立即行动")
print("="*80)

print("\n请先尝试修改1和2（最关键）：")
print("\n1. 编辑配置文件:")
print("   scripts/example_configs/ggeur_officehome_lds.yaml")
print("\n2. 修改以下参数:")
print("""
ggeur:
  cov_regularization: 1e-08  # 从1e-06改为1e-08
  eigenvalue_scaling: False   # 先关闭缩放
""")

print("\n3. 重新训练:")
print("   python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml")

print("\n4. 观察:")
print("   - 特征值是否变大了")
print("   - 测试准确率是否提升")
print("   - 训练/测试差距是否缩小")

print("\n如果问题仍未解决，再尝试修改3和4")
print("\n" + "="*80)
