<#
.SYNOPSIS
    将本项目发布到 GitHub：初始化仓库、提交、绑定远程并推送 main 分支。

.DESCRIPTION
    只做本地 git 操作与推送，不会删除或改写工作区文件。
    若仓库已存在（.git 目录在手），会跳过 init 并直接提交新增改动。

.PARAMETER Remote
    GitHub 仓库地址，例如 https://github.com/<user>/campus-face-recognition.git
    或 git@github.com:<user>/campus-face-recognition.git

.PARAMETER Message
    提交信息，默认 "chore: publish campus face recognition system".

.EXAMPLE
    pwsh -File scripts/publish.ps1 -Remote https://github.com/yourname/campus-face-recognition.git
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Remote,

    [string]$Message = "chore: publish campus face recognition system",

    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Args -join ' ') 执行失败（退出码 $LASTEXITCODE）"
    }
}

Write-Host "==> 仓库目录: $root" -ForegroundColor Cyan

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "未找到 git，请先安装 Git for Windows。"
}

if (-not (Test-Path -LiteralPath (Join-Path $root ".git"))) {
    Write-Host "==> 初始化仓库" -ForegroundColor Cyan
    Invoke-Git init -b $Branch
} else {
    Write-Host "==> 已存在 .git，跳过初始化" -ForegroundColor Yellow
}

Write-Host "==> 检查将被提交的大文件（> 50 MB 会被 GitHub 警告）" -ForegroundColor Cyan
$big = Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -gt 50MB -and $_.FullName -notmatch "\\\.git\\|\\data\\|\\\.venv\\|\\__pycache__\\" }
if ($big) {
    $big | ForEach-Object { Write-Host ("    {0,8:N1} MB  {1}" -f ($_.Length / 1MB), $_.FullName.Replace("$root\", "")) -ForegroundColor Yellow }
} else {
    Write-Host "    未发现超过 50 MB 的待提交文件" -ForegroundColor Green
}

Write-Host "==> 暂存改动" -ForegroundColor Cyan
Invoke-Git add -A

& git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "==> 没有新的改动需要提交" -ForegroundColor Yellow
} else {
    Write-Host "==> 创建提交" -ForegroundColor Cyan
    Invoke-Git commit -m $Message
}

$existing = (& git remote) 2>$null
if ($existing -contains "origin") {
    Write-Host "==> 更新 origin 地址" -ForegroundColor Cyan
    Invoke-Git remote set-url origin $Remote
} else {
    Write-Host "==> 添加 origin" -ForegroundColor Cyan
    Invoke-Git remote add origin $Remote
}

Write-Host "==> 推送 $Branch 到 origin" -ForegroundColor Cyan
Invoke-Git push -u origin $Branch

Write-Host ""
Write-Host "完成。请到 GitHub 仓库页面确认 Actions 是否通过，并设置 About/Topics。" -ForegroundColor Green
Write-Host "发布后检查清单见 docs/PUBLISH.md" -ForegroundColor Green
