import ast
import matplotlib.pyplot as plt

# 存储 round、test_loss 和 test_acc
rounds = []
test_losses = []
test_accs = []

# 假设日志存储在 log.txt 中
with open('exp/FedAvg_rnn_on_ut_har_lr0.0005_lstep5/sub_exp_20250406170537/eval_results.log', 'r', encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            # 将每一行解析为字典
            record = ast.literal_eval(line)
            # 只处理 Role 为 "Server #" 且 Round 为整数的记录
            if record.get('Role') == 'Server #' and isinstance(record.get('Round'), int):
                round_num = record['Round']
                results = record.get('Results_weighted_avg', {})
                # 如果结果中包含 test_loss 和 test_acc 则提取
                if 'test_loss' in results and 'test_acc' in results:
                    rounds.append(round_num)
                    test_losses.append(results['test_loss'])
                    test_accs.append(results['test_acc'])
        except Exception as e:
            # 如果解析出错，直接跳过该行
            continue

# 绘制 test_loss 和 test_acc 随 round 变化的曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rounds, test_losses, marker='o')
plt.xlabel("Round")
plt.ylabel("Test Loss")
plt.title("Test Loss vs Round")

plt.subplot(1, 2, 2)
plt.plot(rounds, test_accs, marker='o', color='orange')
plt.xlabel("Round")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Round")

plt.tight_layout()
plt.show()
