"""
关键诊断问题 - 请回答以下问题帮助定位问题
"""

print("="*80)
print("请提供以下信息：")
print("="*80)

questions = """
1. 原始源码的结果：
   - 最终准确率是多少？(例如：Art: 60%, Clipart: 55%, ...)
   - 训练了多少轮？
   - 每轮的 local_epochs 是多少？

2. 数据增强：
   - 你运行时看到 "CLIENT #X - DATA DISTRIBUTION REPORT" 的输出了吗？
   - 增强后每个客户端有多少样本？
   - 是否每个客户端都有所有 65 个类？

3. 训练过程：
   - 第1轮的训练准确率（train_acc）是多少？
   - 训练 loss 是否在下降？
   - 看到 "Augmentation done" 的日志了吗？

4. 关键日志：
   - 能否提供完整的训练日志（至少前3轮）？
   - 或者截图关键部分？

5. 原始源码对比：
   - 原始源码的数据是如何组织的？
   - 是从 .npy 文件加载的预生成数据吗？
   - 文件路径是什么样的？

回答这些问题后，我能更准确地定位问题所在。
"""

print(questions)

# 同时创建一个诊断脚本
print("\n" + "="*80)
print("或者运行这个诊断脚本查看详细信息：")
print("="*80)
print("""
python -c "
import torch
import os

# Check if augmentation is happening
print('检查关键文件：')
print(f'  run_ggeur.py exists: {os.path.exists(\"run_ggeur.py\")}')
print(f'  Config exists: {os.path.exists(\"scripts/example_configs/ggeur_officehome_lds.yaml\")}')

# Check CLIP cache
cache_dir = 'exp/ggeur_officehome_lds/clip_cache'
if os.path.exists(cache_dir):
    files = os.listdir(cache_dir)
    print(f'\\nCLIP cache files: {len(files)} files')
    for f in files[:5]:
        print(f'  - {f}')
else:
    print(f'\\nCLIP cache directory not found: {cache_dir}')
"
""")
