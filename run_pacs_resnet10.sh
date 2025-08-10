#!/bin/zsh

# 显示使用说明
show_usage() {
    echo "联邦学习实验并行执行脚本"
    echo ""
    echo "用法: ./run_exp.sh [窗口数量]"
    echo ""
    echo "参数:"
    echo "  窗口数量    可选，指定要创建的tmux窗口数量 (默认: 3)"
    echo ""
    echo "示例:"
    echo "  ./run_exp.sh          # 使用默认的3个窗口"
    echo "  ./run_exp.sh 2        # 使用2个窗口 (某些窗口将顺序执行多个任务)"
    echo "  ./run_exp.sh 7        # 使用7个窗口 (每个窗口执行一个任务)"
    echo "  ./run_exp.sh 10       # 使用7个窗口 (任务数少于窗口数)"
    echo ""
    echo "说明:"
    echo "  - 当窗口数 < 任务数时，任务将均匀分配到各窗口中顺序执行"
    echo "  - 当窗口数 >= 任务数时，每个任务独占一个窗口"
    echo "  - 所有窗口都会自动激活 'fs' conda 环境"
    echo ""
}

# 检查是否需要显示帮助
if [[ $1 == "-h" ]] || [[ $1 == "--help" ]]; then
    show_usage
    exit 0
fi

# 定义要执行的命令数组
commands[1]="python federatedscope/main.py --cfg ./scripts/test/pacs_resnet10/ditto_pacs.yaml"
commands[2]="python federatedscope/main.py --cfg ./scripts/test/pacs_resnet10/fedavg_pacs.yaml"
commands[3]="python federatedscope/main.py --cfg ./scripts/test/pacs_resnet10/fedprox_pacs.yaml"
commands[4]="python federatedscope/main.py --cfg ./scripts/test/pacs_resnet10/FedBN_pacs.yaml"
commands[5]="python federatedscope/main.py --cfg ./scripts/test/pacs_resnet10/local_pacs.yaml"
commands[6]="python federatedscope/main.py --cfg ./scripts/test/pacs_resnet10/pFedMe_pacs.yaml"
commands[7]="python federatedscope/main.py --cfg ./scripts/test/pacs_resnet10/ours_pacs.yaml"

# 定义任务名称数组
task_names[1]="ditto"
task_names[2]="fedavg"
task_names[3]="fedprox"
task_names[4]="FedBN"
task_names[5]="local"
task_names[6]="PFedMe"
task_names[7]="ours"

# 配置参数
total_commands=7
default_max_windows=3

# 从命令行参数获取窗口数量，如果没有提供则使用默认值
if [[ $# -gt 0 ]]; then
    max_windows=$1
else
    max_windows=$default_max_windows
fi

# 验证窗口数量
if [[ ! $max_windows =~ ^[0-9]+$ ]] || [[ $max_windows -lt 1 ]]; then
    echo "错误: 窗口数量必须是大于0的正整数"
    echo "用法: $0 [窗口数量]"
    echo "示例: $0 3  # 使用3个窗口"
    exit 1
fi

# tmux 会话名称
session_name="federated_exp"

# 检查 tmux 是否已安装
if ! command -v tmux &> /dev/null; then
    echo "错误: tmux 未安装，请先安装 tmux"
    exit 1
fi

# 检查会话是否已存在，如果存在则杀死
tmux has-session -t "$session_name" 2>/dev/null && tmux kill-session -t "$session_name"

# 计算任务分配
actual_windows=$((max_windows < total_commands ? max_windows : total_commands))

echo "创建 tmux 会话: $session_name"
echo "总任务数: $total_commands"
echo "指定窗口数: $max_windows"
echo "实际使用窗口数: $actual_windows"
echo ""

# 分配任务到各个窗口
typeset -A window_commands
current_cmd_index=1

for window_index in {1..$actual_windows}; do
    window_commands[$window_index]=""
done

# 轮询分配任务到窗口
for cmd_index in {1..$total_commands}; do
    window_index=$(((cmd_index - 1) % actual_windows + 1))
    
    if [[ -n "${window_commands[$window_index]}" ]]; then
        window_commands[$window_index]="${window_commands[$window_index]}|$cmd_index"
    else
        window_commands[$window_index]="$cmd_index"
    fi
done

# 显示任务分配
for window_index in {1..$actual_windows}; do
    echo "窗口 $window_index 将执行任务: ${window_commands[$window_index]}"
done
echo ""

# 创建新的 tmux 会话，并设置第一个窗口
tmux new-session -d -s "$session_name" -n "window_1" -c "$(pwd)"

# 为每个窗口设置任务
for window_index in {1..$actual_windows}; do
    if [[ $window_index -gt 1 ]]; then
        # 创建新窗口（除了第一个窗口）
        tmux new-window -t "$session_name" -n "window_$window_index" -c "$(pwd)"
    fi
    
    window_name="window_$window_index"
    task_indices="${window_commands[$window_index]}"
    
    echo "设置窗口 $window_index (任务: $task_indices)..."
    
    # 激活 conda 环境
    tmux send-keys -t "$session_name:$window_name" "conda activate fs" Enter
    
    # 分割任务索引并逐个执行
    IFS='|' read -A task_array <<< "$task_indices"
    
    for task_index in "${task_array[@]}"; do
        task_name="${task_names[$task_index]}"
        command="${commands[$task_index]}"
        
        # 添加任务开始标记
        tmux send-keys -t "$session_name:$window_name" "echo '==== 开始执行任务 $task_index: $task_name ===='" Enter
        
        # 执行命令
        tmux send-keys -t "$session_name:$window_name" "$command" Enter
        
        # 添加任务完成标记和分隔符（除了最后一个任务）
        if [[ $task_index != "${task_array[-1]}" ]]; then
            tmux send-keys -t "$session_name:$window_name" "echo '==== 任务 $task_index 完成 ===='" Enter
            tmux send-keys -t "$session_name:$window_name" "echo ''" Enter
        fi
    done
    
    # 添加窗口完成标记
    tmux send-keys -t "$session_name:$window_name" "echo '==== 窗口 $window_index 的所有任务已完成 ===='" Enter
done

echo ""
echo "所有实验已在 tmux 会话 '$session_name' 中启动"
echo ""
echo "任务分配情况:"
for window_index in {1..$actual_windows}; do
    task_indices="${window_commands[$window_index]}"
    IFS='|' read -A task_array <<< "$task_indices"
    task_list=""
    for task_index in "${task_array[@]}"; do
        if [[ -n $task_list ]]; then
            task_list="$task_list, ${task_names[$task_index]}"
        else
            task_list="${task_names[$task_index]}"
        fi
    done
    echo "  窗口 $window_index: $task_list"
done
echo ""
echo "使用以下命令查看和管理实验:"
echo "  tmux attach-session -t $session_name  # 连接到会话"
echo "  tmux list-windows -t $session_name    # 查看所有窗口"
echo "  tmux list-sessions                    # 查看所有会话"
echo "  tmux kill-session -t $session_name    # 终止整个会话"
echo ""
echo "在 tmux 会话中的快捷键:"
echo "  Ctrl+b, c    # 创建新窗口"
echo "  Ctrl+b, n    # 切换到下一个窗口"
echo "  Ctrl+b, p    # 切换到上一个窗口"
echo "  Ctrl+b, 0-9  # 切换到指定编号的窗口"
echo "  Ctrl+b, w    # 列出所有窗口"
echo "  Ctrl+b, d    # 从会话中分离 (detach)"
echo ""

# # 自动连接到会话 (可选)
# read -q "REPLY?是否要立即连接到 tmux 会话? (y/n): "
# echo
# if [[ $REPLY =~ ^[Yy]$ ]]; then
#     tmux attach-session -t "$session_name"
# fi
