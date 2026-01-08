#!/bin/bash
# GGEUR_Clip 模型上传和配置脚本

echo "============================================"
echo "GGEUR 模型上传和配置"
echo "============================================"

# 1. 上传模型文件
echo ""
echo "步骤 1: 上传模型到服务器"
echo "----------------------------------------"
echo "在本地 Windows 电脑上运行："
echo ""
echo "scp \"C:\\Users\\Dbook\\Downloads\\open_clip_vitb16.bin\" root@10.112.81.135:/root/FederatedScope/"
echo ""
echo "或者使用 PowerShell："
echo 'scp "C:\Users\Dbook\Downloads\open_clip_vitb16.bin" root@10.112.81.135:/root/FederatedScope/'
echo ""

# 2. 登录服务器验证
echo "步骤 2: 登录服务器验证"
echo "----------------------------------------"
echo "ssh root@10.112.81.135"
echo ""
echo "登录后执行："
echo "ls -lh /root/FederatedScope/open_clip_vitb16.bin"
echo ""

# 3. 配置文件已更新
echo "步骤 3: 配置文件已更新"
echo "----------------------------------------"
echo "已修改配置文件："
echo "  scripts/example_configs/ggeur_officehome_lds.yaml"
echo ""
echo "配置内容："
echo "  clip_model: ViT-B/16"
echo "  clip_custom_weights: /root/FederatedScope/open_clip_vitb16.bin"
echo ""

# 4. 同步代码到服务器
echo "步骤 4: 同步代码到服务器"
echo "----------------------------------------"
echo "方法 A: 使用 git (推荐)"
echo "  在服务器上："
echo "  cd /root/FederatedScope"
echo "  git pull"
echo ""
echo "方法 B: 使用 rsync"
echo "  在本地运行："
echo "  rsync -avz --exclude '.git' --exclude '__pycache__' \\"
echo "    D:/Projects/FederatedScope/ root@10.112.81.135:/root/FederatedScope/"
echo ""

# 5. 运行实验
echo "步骤 5: 在服务器上运行实验"
echo "----------------------------------------"
echo "ssh root@10.112.81.135"
echo "cd /root/FederatedScope"
echo ""
echo "# 安装依赖（如果还没安装）"
echo "pip install open-clip-torch"
echo ""
echo "# 清除旧缓存"
echo "rm -rf exp/ggeur_officehome_lds/clip_cache"
echo ""
echo "# 运行实验"
echo "python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml"
echo ""

echo "============================================"
echo "配置完成！"
echo "============================================"
