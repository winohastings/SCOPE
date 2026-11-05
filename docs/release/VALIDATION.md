# 发布验证记录

日期：2026-11-05。本次仅整理与发布审查，没有重新运行 GPU 导航实验，没有修改 AdaPercVLN 原实验源码、冻结协议、历史日志、权重或第三方工作区。

## 本地已完成

- 从锁定 SimpleNav 官方提交重建独立副本，Stage1 补丁可应用，14 个依赖文件最终 SHA256 与实测工作区一致。
- 从历史 legacy 源逐文件校验后重建 runtime，238 个运行时文件全部与 Task1 provenance 中的实测版本 SHA256 一致；官方完整代码不纳入 Git。
- SCOPE 既有回归 14 项通过，协议回归 4 项通过；原始输出见本目录两份回归日志。
- 发布新增归档回归 3 项通过，包括完整证据核验、内容篡改拒绝、私钥内容检测。
- 新增与归档的 24 个 Python 文件通过语法解析；不把该检查冒充模型推理验证。
- 四组各 64 个相同 episode，执行失败 0，逐步记录数 G0=829、G1=954、G2=1002、G3=983，总计 3768；当前 token 预算逐行校验通过。
- 所有选中文件与压缩解压内容均与原始源 SHA256 一致，完整清单可由 `tools/verify_release.py` 重新验证。

依赖重建结果由 `.gitignore` 排除。现有 ScopePolicy 集成回归仍引用历史绝对路径的本机外部依赖，不宣称跨机器路径迁移已经完成。

## GitHub 前置条件

`id_ed25519.pub` 指纹为 `SHA256:4eqszijS/nGnOufaRZKudcL3wda6ZKSTK5AUWVTw5KI`，显式使用该密钥时 GitHub 返回 `Hi winohastings! You've successfully authenticated`。这证明 SSH 身份认证成功，不证明目标仓库已存在，也不提供创建仓库的 REST API 授权。

检查时 `git ls-remote git@github-wino:winohastings/SCOPE.git` 返回 `Repository not found`，公共 API 返回 HTTP 404。本机未配置 GitHub CLI/API token，因此创建远端仓库需要账号持有者网页操作或后续 API 授权；不把“准备好推送”写成“已上传成功”。

建议先创建私有空仓库 `winohastings/SCOPE`，不要初始化 README、license 或 gitignore。完成后在本发布目录执行：

```bash
git push \
  --dry-run \
  origin \
  main

git push \
  --set-upstream \
  origin \
  main
```

实际远端推送与重新clone核验完成前，本文件不代表 GitHub 发布完成。
