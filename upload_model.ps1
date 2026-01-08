# PowerShell 脚本 - Windows 本地执行

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "GGEUR 模型上传到服务器" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 配置
$LOCAL_MODEL = "C:\Users\Dbook\Downloads\open_clip_vitb16.bin"
$SERVER = "root@10.112.81.135"
$REMOTE_PATH = "/root/FederatedScope/"

# 检查本地文件是否存在
Write-Host "检查本地模型文件..." -ForegroundColor Yellow
if (Test-Path $LOCAL_MODEL) {
    $size = (Get-Item $LOCAL_MODEL).Length / 1MB
    Write-Host "  ✓ 找到模型文件: $LOCAL_MODEL" -ForegroundColor Green
    Write-Host "    大小: $([math]::Round($size, 2)) MB" -ForegroundColor Green
} else {
    Write-Host "  ✗ 模型文件不存在: $LOCAL_MODEL" -ForegroundColor Red
    Write-Host "    请检查路径是否正确" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "上传模型到服务器..." -ForegroundColor Yellow
Write-Host "  命令: scp `"$LOCAL_MODEL`" ${SERVER}:${REMOTE_PATH}" -ForegroundColor Gray
Write-Host ""

# 执行上传
try {
    scp "$LOCAL_MODEL" "${SERVER}:${REMOTE_PATH}"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ 上传成功！" -ForegroundColor Green
    } else {
        Write-Host "  ✗ 上传失败" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ✗ 上传出错: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "验证上传..." -ForegroundColor Yellow
ssh $SERVER "ls -lh ${REMOTE_PATH}open_clip_vitb16.bin"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "上传完成！" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：" -ForegroundColor Yellow
Write-Host "1. 同步代码到服务器（如果使用git，在服务器上执行 git pull）" -ForegroundColor White
Write-Host "2. 登录服务器: ssh $SERVER" -ForegroundColor White
Write-Host "3. 运行实验: python run_ggeur.py --cfg scripts/example_configs/ggeur_officehome_lds.yaml" -ForegroundColor White
Write-Host ""
