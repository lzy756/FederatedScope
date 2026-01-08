"""
Plot accuracy comparison between FedAvg and GGEUR_Clip methods.
Usage: python plot_accuracy_comparison.py
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os

# 日志文件路径 - 根据实际情况修改
LOG_DIR = r"D:\Projects\FederatedScope\exp\GGEUR_CNN_convnext"
FEDAVG_LOG = os.path.join(LOG_DIR, "fedavg_alpha0.1_clientsnum60_exp_print.log")
GGEUR_LOG = os.path.join(LOG_DIR, "ggeur_alpha0.1_clientsnum60_exp_print.log")


def parse_log_file(log_path):
    """解析日志文件中的准确率数据"""
    data = {
        'rounds': [],
        'average': [],
        'Art': [],
        'Clipart': [],
        'Product': [],
        'Real_World': []
    }

    if not os.path.exists(log_path):
        print(f"警告: 日志文件不存在: {log_path}")
        return data

    # 匹配准确率行
    # Server: Round 1 MLP Test Accuracy - Art: 0.0096, Clipart: 0.0214, Product: 0.0293, Real_World: 0.0107, average: 0.0177
    pattern = r'Server: Round (\d+) MLP Test Accuracy - Art: ([\d.]+), Clipart: ([\d.]+), Product: ([\d.]+), Real_World: ([\d.]+), average: ([\d.]+)'

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data['rounds'].append(int(match.group(1)))
                data['Art'].append(float(match.group(2)))
                data['Clipart'].append(float(match.group(3)))
                data['Product'].append(float(match.group(4)))
                data['Real_World'].append(float(match.group(5)))
                data['average'].append(float(match.group(6)))

    return data


def plot_average_comparison(fedavg_data, ggeur_data, output_path):
    """绘制平均准确率对比图"""
    plt.figure(figsize=(10, 6))

    # FedAvg
    if fedavg_data['rounds']:
        plt.plot(fedavg_data['rounds'], np.array(fedavg_data['average']) * 100,
                'b-o', label=f"FedAvg (Final: {fedavg_data['average'][-1]*100:.2f}%)",
                linewidth=2, markersize=3)

    # GGEUR_Clip
    if ggeur_data['rounds']:
        plt.plot(ggeur_data['rounds'], np.array(ggeur_data['average']) * 100,
                'r-s', label=f"GGEUR_Clip (Final: {ggeur_data['average'][-1]*100:.2f}%)",
                linewidth=2, markersize=3)

    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Average Test Accuracy (%)', fontsize=12)
    plt.title('FedAvg vs GGEUR_Clip Average Accuracy\n(Office-Home, LDS alpha=0.1, 60 clients, ConvNeXt-Base)', fontsize=14)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)

    max_round = max(fedavg_data['rounds'] + ggeur_data['rounds']) if (fedavg_data['rounds'] or ggeur_data['rounds']) else 100
    plt.xlim([0, max_round + 1])
    plt.ylim([0, 70])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"保存图表: {output_path}")
    plt.show()


def plot_all_domains(fedavg_data, ggeur_data, output_path):
    """绘制所有域的准确率对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('FedAvg vs GGEUR_Clip Accuracy Comparison\n(Office-Home, LDS alpha=0.1, 60 clients)', fontsize=14)

    domains = ['average', 'Art', 'Clipart', 'Product', 'Real_World']
    titles = ['Average', 'Art', 'Clipart', 'Product', 'Real_World']

    for idx, (domain, title) in enumerate(zip(domains, titles)):
        ax = axes[idx // 3, idx % 3]

        if fedavg_data['rounds'] and fedavg_data[domain]:
            ax.plot(fedavg_data['rounds'], np.array(fedavg_data[domain]) * 100,
                   'b-', label='FedAvg', linewidth=2)

        if ggeur_data['rounds'] and ggeur_data[domain]:
            ax.plot(ggeur_data['rounds'], np.array(ggeur_data[domain]) * 100,
                   'r-', label='GGEUR_Clip', linewidth=2)

        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 80])

    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"保存图表: {output_path}")
    plt.show()


def main():
    print("=" * 60)
    print("准确率对比绘图工具")
    print("=" * 60)

    print(f"\n读取日志文件:")
    print(f"  FedAvg: {FEDAVG_LOG}")
    print(f"  GGEUR_Clip:  {GGEUR_LOG}")

    # 解析日志
    fedavg_data = parse_log_file(FEDAVG_LOG)
    ggeur_data = parse_log_file(GGEUR_LOG)

    print(f"\n数据统计:")
    print(f"  FedAvg: {len(fedavg_data['rounds'])} 轮数据")
    print(f"  GGEUR_Clip:  {len(ggeur_data['rounds'])} 轮数据")

    # 打印汇总
    if fedavg_data['rounds']:
        print(f"\nFedAvg 结果:")
        print(f"  最终平均准确率: {fedavg_data['average'][-1]*100:.2f}%")
        print(f"  最佳平均准确率: {max(fedavg_data['average'])*100:.2f}%")

    if ggeur_data['rounds']:
        print(f"\nGGEUR_Clip 结果:")
        print(f"  最终平均准确率: {ggeur_data['average'][-1]*100:.2f}%")
        print(f"  最佳平均准确率: {max(ggeur_data['average'])*100:.2f}%")
    else:
        print(f"\nGGEUR_Clip: 暂无准确率数据（实验可能仍在运行）")

    # 绘图
    if fedavg_data['rounds'] or ggeur_data['rounds']:
        print("\n生成图表...")

        # 平均准确率对比图
        avg_plot = os.path.join(LOG_DIR, "average_accuracy_comparison.png")
        plot_average_comparison(fedavg_data, ggeur_data, avg_plot)

        # 所有域对比图
        full_plot = os.path.join(LOG_DIR, "full_accuracy_comparison.png")
        plot_all_domains(fedavg_data, ggeur_data, full_plot)

        print("\n完成!")
    else:
        print("\n错误: 两个日志文件都没有准确率数据")


if __name__ == '__main__':
    main()
