# 发布验证记录

日期：2026-11-05。本次仅整理与发布审查，没有重新运行 GPU 导航实验，没有修改 AdaPercVLN 原实验源码、冻结协议、历史日志、权重或第三方工作区。

## 本地已完成

- 从锁定 SimpleNav 官方提交重建独立副本，Stage1 补丁可应用，14 个依赖文件最终 SHA256 与实测工作区一致。
- 从历史 legacy 源逐文件校验后重建 runtime，238 个运行时文件全部与 Task1 provenance 中的实测版本 SHA256 一致；官方完整代码不纳入 Git。
- SCOPE 既有回归 14 项通过，协议回归 4 项通过；原始输出见本目录两份回归日志。
- 独立重建的 SimpleNav 副本另通过路径/nDTW回归5项，确保发布补丁不仅能应用且能通过对应数据适配检查。
- 发布新增归档回归 4 项，包括完整证据核验、内容篡改拒绝、私钥内容检测与嵌套Git目录补丁隔离。
- 新增与归档的 24 个 Python 文件通过语法解析；不把该检查冒充模型推理验证。
- 四组各 64 个相同 episode，执行失败 0，逐步记录数 G0=829、G1=954、G2=1002、G3=983，总计 3768；当前 token 预算逐行校验通过。
- 所有选中文件与压缩解压内容均与原始源 SHA256 一致，完整清单可由 `tools/verify_release.py` 重新验证。

依赖重建结果由 `.gitignore` 排除。现有 ScopePolicy 集成回归仍引用历史绝对路径的本机外部依赖，不宣称跨机器路径迁移已经完成。

## GitHub 前置条件

`id_ed25519.pub` 指纹为 `SHA256:4eqszijS/nGnOufaRZKudcL3wda6ZKSTK5AUWVTw5KI`，显式使用该密钥时 GitHub 返回 `Hi winohastings! You've successfully authenticated`。这证明 SSH 身份认证成功，不证明目标仓库已存在，也不提供创建仓库的 REST API 授权。

最初检查时 `git ls-remote git@github-wino:winohastings/SCOPE.git` 返回 `Repository not found`，公共 API 返回 HTTP 404。之后账号持有者创建了公开远端，初始提交 `8724cd49ce3372c251f2a98b2451c2869905c3cf` 包含 Apache-2.0 `LICENSE`。本地合并该提交并原样保留许可证，不使用 force push，也不覆盖用户的初始历史。

本机未配置 GitHub CLI/API token，远端创建不冒充由代理API完成。后续使用现有SSH密钥推送即可：

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

## 实际推送

SSH 推送 dry-run 成功；随后实际执行 `git push --set-upstream origin main` 成功，远端从用户初始提交推进到合并发布提交 `13cc9e750812fb39a4a07d681b2b56ba659a1dab`。再次 `git ls-remote origin refs/heads/main` 返回同一提交；发布未使用强制推送，保留了用户创建的完整远端历史。

本归档包含约17.7MB的选中文件（不含Git元数据和被忽略的本地重建依赖），官方完整仓库、权重与场景未进入Git。原始AdaPercVLN工作区未做改写。

直接GitHub下载链路较慢，额外通过公开加速镜像取得独立完整clone，提交ID与GitHub SSH远端一致；全部归档哈希、256集/3768次预测分母与归档测试通过。该检查不冒充直接链路下载性能。克隆后依赖重建发现父Git仓库影响`git apply`的路径选择，工具改为显式`patch -d`并增加隔离回归；实验源码和结果不变。
