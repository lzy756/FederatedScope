import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re

def parse_log_file(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            line = re.sub(r'(\d)\s+(\.)', r'\1\2', line)
            try:
                entry = eval(line, {"__builtins__": None}, {"nan": np.nan})
                data.append(entry)
            except Exception as e:
                print(f"解析失败的行：{line}")
                print(f"错误详情：{str(e)}")
                continue

    rounds = []
    val_avg_loss = []
    for entry in data:
        try:
            round_num = int(entry["Round"])
        except (ValueError, KeyError):
            continue

        weighted_avg_key = next((k for k in entry.keys() if "weighted_avg" in k.lower()), None)

        if weighted_avg_key:
            val_avg_loss.append(entry[weighted_avg_key]["test_acc"])
            rounds.append(round_num)
        else:
            val_avg_loss.append(float('nan'))
            rounds.append(round_num)

    # 去重：每轮保留最后一个值
    rounds_np = np.array(rounds)
    val_avg_loss_np = np.array(val_avg_loss)
    unique_rounds = np.unique(rounds_np)
    unique_val_avg_loss = [
        val_avg_loss_np[np.where(rounds_np == r)[0][-1]] for r in unique_rounds
    ]
    return unique_rounds.tolist(), unique_val_avg_loss

# === 路径 1：FedProx ===
path1 = "/root/autodl-tmp/FederatedScope/exp/fedprox_resnet18_on_FashionMNIST@torchvision_lr0.01_lstep1/sub_exp_20250609114257/eval_results.log"
rounds1, val_loss1 = parse_log_file(path1)

# === 路径 2：FedAvg ===
path2 = "/root/autodl-tmp/FederatedScope/exp/FedAvg_resnet18_on_FashionMNIST@torchvision_lr0.01_lstep1/sub_exp_20250609104739/eval_results.log"
rounds2, val_loss2 = parse_log_file(path2)

# === 路径 3：FedSAK ===
path3 = "/root/autodl-tmp/FederatedScope/exp/fedsak_resnet18_on_FashionMNIST@torchvision_lr0.01_lstep1/sub_exp_20250609114717/eval_results.log"
rounds3, val_loss3 = parse_log_file(path3)

path4 = "/root/autodl-tmp/FederatedScope/exp/FedAvg_resnet18_on_FashionMNIST@torchvision_lr0.01_lstep1/sub_exp_20250611153704/eval_results.log"
rounds4, val_loss4 = parse_log_file(path4)

# === 绘图 ===
plt.figure(figsize=(30, 20))
plt.plot(rounds1, val_loss1, marker='o', linestyle='-', label='FedProx', color='b')
plt.plot(rounds2, val_loss2, marker='s', linestyle='--', label='FedAvg', color='r')
plt.plot(rounds3, val_loss3, marker='^', linestyle='-.', label='FedSAK', color='g')  # 新增曲线
plt.plot(rounds4, val_loss4, marker='x', linestyle=':', label='FedAvg (iid)', color='m')  # 新增曲线
plt.xticks(rounds1[::5])
plt.xlabel("Round", fontsize=20)
plt.ylabel("test_acc", fontsize=20)
plt.title("Comparison of test_acc: FedAvg vs FedProx vs FedSAK", fontsize=24)
plt.legend(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)

# 设置纵坐标更精细
min_loss = min(min(val_loss1), min(val_loss2), min(val_loss3))
max_loss = max(max(val_loss1), max(val_loss2), max(val_loss3))
plt.ylim(min_loss - 0.1, max_loss + 0.1)
plt.yticks(np.linspace(min_loss - 0.1, max_loss + 0.1, num=20))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.savefig("FedAvgiid_vs_FedProx_vs_FedSAK_fashionemnist_test_acc.png", dpi=300, bbox_inches='tight')
