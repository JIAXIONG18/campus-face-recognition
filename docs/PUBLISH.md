# 发布到 GitHub

本文档说明如何把这个项目变成一个「点开就能看懂、克隆就能跑」的 GitHub 仓库，包括体积策略、提交步骤、认证方式与发布后的检查清单。

---

## 1. 发布内容与体积策略

| 内容 | 是否发布 | 说明 |
|---|---|---|
| `code/`（后端 / 算法 / 前端 / 脚本） | ✅ | 约 0.4 MB |
| `experiments/figures/`（19 张图） | ✅ | 约 2 MB，README 与文档直接引用 |
| `experiments/*.json`（指标 / 消融 / 基准） | ✅ | 让所有数字可核对 |
| `experiments/runs/finetune_full/best.pt` | ✅ | 43.8 MB，使克隆后无需训练即可运行 |
| `experiments/runs/liveness/best.pt` | ✅ | 0.4 MB |
| 其它 `experiments/runs/*/best.pt` | ❌ | 消融产物，可复现，不必入库 |
| `data/`（约 2.0 GB） | ❌ | 数据集与对齐结果，按脚本重建 |
| `experiments/embeddings.npz`、`experiments/logs/` | ❌ | 中间产物 |
| 论文 PDF、`docs/source/`、`defense/` | ❌ | 含姓名、学号等个人信息 |
| `*.db`、`__pycache__`、`.venv`、`.env` | ❌ | 运行时产物与密钥 |

以上规则全部写在 [`.gitignore`](../.gitignore) 中。仓库最终约 **45 MB**，其中 44 MB 是两份权重——低于 GitHub 单文件 100 MB 硬限制，也低于 50 MB 的警告线，**不需要 Git LFS**。

> 如果希望仓库更轻（例如只做代码展示），可以删除权重并改为"首次运行前下载"：
>
> ```bash
> git rm --cached experiments/runs/finetune_full/best.pt
> ```
>
> 但要注意：没有权重时 `/api/health` 会加载失败，README 的"开箱即用"承诺不再成立，需同步修改快速开始章节。

---

## 2. 发布前检查

- [ ] 没有真实校园人脸数据、身份证号、手机号进入仓库（当前只用了公开数据集与虚构学号）；
- [ ] 没有把 `.env`、token、私钥提交上去（`.gitignore` 已覆盖）；
- [ ] 论文 PDF / docx / 答辩材料未提交（`Based on...pdf`、`*.docx`、`defense/` 已忽略）；
- [ ] `experiments/*.json` 中的本地绝对路径不影响他人阅读（仅是路径字符串，不含敏感信息）；
- [ ] `git status` 中不出现 `data/` 与 `*.pt`（预期的两份权重除外）。

检查忽略规则是否生效：

```bash
git status --short
git check-ignore -v data/campus_face.db experiments/runs/abl_full/best.pt
```

---

## 3. 在 GitHub 上创建仓库

1. 打开 <https://github.com/new>；
2. **Repository name**：`campus-face-recognition`（或 `face-recognition-attendance`）；
3. **Description**：`ResNet-18 + ArcFace 人脸识别考勤与门禁系统，CPU 端到端 44 ms，含完整训练/评测/Web 服务`;
4. 选择 **Public**；
5. **不要**勾选 "Add a README file"、"Add .gitignore"、"Choose a license"——本地已有，勾选会造成首次推送冲突；
6. 创建后复制仓库地址。

---

## 4. 初始化并推送

### 方式 A：使用脚本（推荐）

```powershell
pwsh -File scripts/publish.ps1 -Remote https://github.com/<your-name>/campus-face-recognition.git
```

脚本会依次：初始化仓库（若无 `.git`）→ 检查 >50 MB 的文件 → `git add -A` → 提交 → 绑定 `origin` → 推送 `main`。重复执行是安全的（已有提交则只推送新改动）。

### 方式 B：手动命令

```bash
git init -b main
git add -A
git commit -m "feat: campus face recognition system (ResNet-18 + ArcFace + Flask)"
git remote add origin https://github.com/<your-name>/campus-face-recognition.git
git push -u origin main
```

---

## 5. 认证方式

GitHub 已不支持账号密码推送，二选一：

### 5.1 Personal Access Token（HTTPS）

1. 打开 <https://github.com/settings/tokens> → **Generate new token (classic)**；
2. 勾选 `repo` 权限（Fine-grained token 需授权目标仓库的 Contents: Read and write）；
3. 生成后复制（只显示一次）；
4. 推送时用户名填 GitHub 用户名，密码位置粘贴 token：

```bash
git push -u origin main
# Username: your-name
# Password: ghp_xxxxxxxxxxxxxxxx
```

为避免每次输入，可让 Git 记住（Windows 凭据管理器）：

```bash
git config --global credential.helper manager
```

### 5.2 SSH Key

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
# 回车使用默认路径，可设置 passphrase
cat ~/.ssh/id_ed25519.pub        # 复制输出
```

把公钥粘贴到 <https://github.com/settings/keys>，然后把远程地址改为 SSH：

```bash
git remote set-url origin git@github.com:<your-name>/campus-face-recognition.git
git push -u origin main
```

---

## 6. 网络与代理

如果推送报 `Failed to connect to github.com port 443` 或 TLS 错误：

```bash
# 查看当前代理配置
git config --global --get http.proxy
git config --global --get https.proxy

# 设置（按本机代理端口修改）
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 取消
git config --global --unset http.proxy
git config --global --unset https.proxy
```

> 当前这台机器上 `git` 全局配置指向 `socks5://127.0.0.1:10808`，但该端口没有程序监听；同时 `github.com` 被解析到 `57.155.223.240`（非 GitHub 真实地址），HTTPS 握手被重置。也就是说：**必须先启动代理客户端（或在能直连 GitHub 的网络环境下）再推送**，否则任何认证方式都不会生效。

---

## 7. 发布后检查清单

- [ ] 仓库首页 README 正常渲染：徽章、头图、SVG 架构图、表格、公式（ArcFace 的 `$$` 块）都显示正确；
- [ ] `docs/assets/architecture.svg` 能打开（GitHub 会以图片方式渲染 SVG）；
- [ ] **Actions** 页面两个 job（Syntax check / Unit tests）均为绿色；
- [ ] 点开 `experiments/figures/fig_ui_recognize.png` 等图片能正常显示（相对路径没写错）；
- [ ] 仓库 **About** 填写 Description 与 Topics；
- [ ] 在另一台机器（或临时目录）执行 `git clone` + 快速开始步骤，确认能跑起来；
- [ ] 若启用 GitHub Pages，可直接把 `docs/` 作为站点（可选）。

### 推荐的 Topics

```
face-recognition  arcface  pytorch  resnet  flask  opencv  yunet
liveness-detection  biometrics  attendance-system  deep-learning  web-app
```

### 推荐的 About 描述

> 校园人脸识别考勤与门禁系统 · ResNet-18 + ArcFace · Rank-1 84.1% / EER 6.93% · CPU 端到端 44 ms · Flask + 原生 JS

### 社交预览图

Settings → General → Social preview 上传 `experiments/figures/fig_ui_recognize.png`（1418×902，比例合适），分享链接时更直观。

---

## 8. 后续更新

```bash
git add -A
git commit -m "docs: refine evaluation section"
git push
```

建议的提交信息风格（Conventional Commits）：

| 前缀 | 用途 |
|---|---|
| `feat:` | 新功能（接口、页面、模型组件） |
| `fix:` | 缺陷修复 |
| `docs:` | 文档与图表 |
| `refactor:` | 不改行为的重构 |
| `test:` | 测试 |
| `chore:` | 依赖、CI、构建 |

---

## 9. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `remote: Support for password authentication was removed` | 用了账号密码 | 改用 PAT 或 SSH |
| `remote: Permission to ... denied` | token 无 `repo` 权限或用户名错误 | 重新生成 token，确认用户名 |
| `Failed to connect to github.com port 443` | 网络/代理不通 | 启动代理并按第 6 节配置，或换网络 |
| `error: src refspec main does not match any` | 还没有提交 | 先 `git add -A && git commit` |
| `fatal: remote origin already exists` | 重复添加 | `git remote set-url origin <url>` |
| `GH001: Large files detected` / 推送被拒 | 单文件 >100 MB | 移出 `.gitignore` 或用 Git LFS |
| 图片在 README 中不显示 | 路径大小写或相对层级错误 | 用仓库根目录为基准的相对路径 |
| Actions 红了 | 依赖或测试环境差异 | 本地先跑 `pytest -q`，按日志修 |
| Windows 克隆后脚本换行异常 | CRLF | 仓库已含 `.gitattributes` 统一为 LF |

---

## 10. 许可与合规提醒

- 代码以 MIT 发布，保留 [`LICENSE`](../LICENSE)；
- 数据集（LFW / CelebA / ChokePoint / CelebA-Spoof）遵循各自**仅限研究**的许可，不能随仓库分发原始图像，也不可用于商业产品；
- 项目中的学号、姓名、人脸均为公开数据集样本或虚构数据，不对应真实个人；
- 若将来接入真实校园数据，需履行告知同意、最小化采集、加密存储与删除权，并遵守《个人信息保护法》。
