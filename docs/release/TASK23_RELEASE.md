# Stage2 / Task23 发布说明

本次归档导航级 PPO/GAE、独立 Critic、Actor/Critic 联合更新、恢复门禁和正式 4096-transition 实验。未在发布过程中重跑 GPU 训练。原始报告、代码与选中数据保持原内容，压缩文件解压后仍可由 SOURCE_MAP 校验。

## 结果与边界

- 276 个 train episodes、4096 个真实训练 transitions、56 轮更新，Actor/Critic 各 570 个 optimizer steps。
- 固定 32 个 val_seen episodes 验证 5 轮，共 160 episodes / 2452 transitions；验证不计入训练分母。
- 最佳 1033 步：SR 0.5625、SPL 0.42185182981204195。初始化 SR 0.34375、SPL 0.28943913148335776；最终 SR 0.34375、SPL 0.2604978682966848。
- 最终确定性策略出现早 STOP / 零 Focus 退化：498/498 验证 transitions 没有 Focus。本次不构成有效论文基线或泛化结论。
- 训练和验证完成、最终 checkpoint 保存后，OpenGL 清理报 `GL::Context::current(): no current context`，进程 exit 134，不能称为干净退出。
- `COMPLETE.json` 中 cursor 旧一轮；最终恢复审计 `FINAL_CHECKPOINT_AUDIT.json` 的 cursor 为 train_episode_index=276、validation_round=5、next_validation_at=5120。恢复审计是原实验记录，不是本次发布重新加载权重所得。

## 上传内容

- Task23 八个实现脚本、配置、任务要求、协议、报告、trace schema 与相关单元测试。
- 正式运行核心 driver/server/审计日志、结果 CSV/SVG/JSON、436 集结果及配置、6548 条压缩 trace 摘要、56 轮更新记录与固定 targets 数据。
- 成功的 RNG 隔离恢复门禁及失败门禁的精简日志，保留故障证据。
- `TASK23_SOURCE_MAP.json` 记录本次原始文件映射；`TASK23_EXCLUDED_FILES.json` 说明被省略文件。被排除产物的历史 hash 来自原清单，并非发布时重新读取全量 20GB 产物计算。

## 关键训练日志入口

以下路径相对于 `Experiment/SCOPE/Stage2/Task23/artifacts/ppo4096/`，全部随本次提交上传：

| 路径 | 内容 |
|---|---|
| `driver.log` | 正式训练、逐集进度及最终 OpenGL 清理错误 |
| `server.log` | 模型服务、训练请求与服务端执行记录 |
| `runtime/updates/update_*.json` | 56 轮 PPO 的 loss、KL、梯度、参数变化、冻结审计与 minibatch 指标 |
| `summary/training.csv` | 逐轮训练指标表 |
| `summary/loss.svg`、`summary/kl.svg`、`summary/entropy.svg` | loss、KL、熵曲线 |
| `runtime/validation_history.json`、`summary/val_seen.svg` | 五轮固定验证指标与曲线 |
| `runtime/scope.jsonl.gz`、`runtime/trace_summary.json.gz` | 压缩运行事件及逐 transition 摘要 |
| `summary/audit.log`、`EXECUTION_AUDIT.json`、`FINAL_CHECKPOINT_AUDIT.json` | 训练结果、异常退出与原始恢复审计证据 |
| `gpu.csv` | 训练期间 GPU 资源采样 |

## Checkpoint 仅占位

按用户要求，Task23 所有 checkpoint 权重保留原机器本地，GitHub 只保存 11 个 `*.pt.placeholder.json`：正式运行 7 个、成功门禁 4 个。占位包含原路径、字节数、SHA256 和 `uploaded: false`，不可直接由 torch 加载。

完整占位列表见 `TASK23_CHECKPOINT_PLACEHOLDERS.json`。需从可信原机器取得对应文件、核验 SHA256 后，才能加载或恢复。以前任务已发布的权重不在本次变更范围内。

`targets_*.pt.gz` 是固定 advantage/return 数据，不是模型权重。全量 PT traces、episode boundaries、旧策略 snapshots 不上传；本缩减包不能独立重做完整训练 trace replay，也不能仅靠占位恢复训练。原始 README 中部分路径因此仅在原机器有效。

## 依赖与历史兼容

同步 Task22 `actor_runtime.py` 的 episode 验证扩展点，以及 Stage2 `real_rollout.py` 的 reward purpose 和当前 token 读取扩展。旧版源码保存在 `task23_before_sources/`；旧任务验证器按原 hash 验证明确映射的历史快照，绝不覆盖历史合同或跳过哈希检查。原项目 `docs/index.md` 单独保存在 `task23_source_index.md`，用于满足 Task23 原始源码清单。

不包含 Habitat、RxR、R2R、SimpleNav 官方仓库副本、场景资产或基础模型。

## 离线校验

归档验证使用 Python 3.11 标准库，不加载 pickle、不需要 GPU：

```bash
python3.11 tools/verify_task23_release.py
python3.11 -m unittest discover \
  -s tests \
  -p 'test_*archive.py'
```

这仅验证发布包完整性与保留证据的一致性，不代表重新进行导航实验。
