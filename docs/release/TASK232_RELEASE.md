# Stage2 / Task232 发布说明

归档定向修复与受控短续训的实现、冻结协议、关键训练日志、结果与审计证据。保留原始文件内容，不修改 Task21/22/23/231 代码和实验记录；这些任务的顶层 Python 依赖与现有发布版本一致。本次发布不重跑 GPU 训练或导航。

## 实验范围与真实结论

- 从 Task23 update14（1033 transitions）完整状态分叉。U 工程门禁新增 64 条；A、B、C0、C 各新增 512 条真实 on-policy transitions，总计 **2112**。恢复重放不重复计数，旧轨迹不送入训练更新。
- A 仅将 Focus 成本降至 0.0025；B 保持原奖励并使用稳定性配置包（rollout minimum 128、Actor LR 3e-5、Critic LR 1e-4）。B 实际更新 batch 为 256/256，不把最小值当实际 batch。
- **新增 C/C0 是覆盖奖励探索，不是原任务文档条件式 Adapter 修复 C。** C 的 beta=0.01，C0 的 beta=0；没有重训 Adapter、没有引入直接 JS/KL 奖励或改成逐 Focus advantage。
- 四个分支各有 historical32 和 independent32 评估，共 256 集；另有 BASE1033 独立集 32 集。独立集不与历史集共享 episode/trajectory，但不保证场景独立，且已是开发集而非永久未触碰测试集。
- 原4096模型阈值诊断新增 24 集（0.45/0.6 各12），0.5 的12集复用已由 Task231 核验的历史结果；表中36条不是36次新增评估。

| 独立集 | SR | SPL |
|---|---:|---:|
| BASE1033 | 37.500% | 30.262% |
| A | 28.125% | 21.936% |
| B | 34.375% | 25.778% |
| C0 | 28.125% | 21.091% |
| C | 37.500% | 26.592% |

A/B 未通过冻结的1024扩展门槛，不训练 A+B。C/C0 的确定性四Focus重复比例仍为1，联合覆盖约0.035；C 未通过空间探索门槛，不自动扩训，不宣称已找到 Task233 稳定配置。小样本区间跨0，不宣称真实效果已被统计证明为负，也不把 C 的单次 SR 当成有效空间学习。

## 关键训练日志与数据

下列路径相对于 `Experiment/SCOPE/Stage2/Task232/`：

| 路径 | 内容 |
|---|---|
| `branch_A512.log`、`branch_B512.log`、`branch_C0512.log`、`branch_C512.log` | 四分支训练/评估主日志与清理异常 |
| `unchanged_gate.log`、`baseline_independent.log`、`thresholds.log` | 一致性门禁、独立基线与阈值诊断 |
| `artifacts/*/server.log`、`LAUNCH.json`、`gpu.csv` | 服务日志、启动配置、共享整卡资源采样 |
| `artifacts/*/runtime/updates/update_*.json` | 逐更新 loss、ratio、KL、梯度、冻结及 minibatch 指标 |
| `artifacts/*/runtime/updates/targets_*.pt.gz` | 固定 GAE/return/advantage 数据，不是模型权重 |
| `artifacts/*/runtime/probes/` | 0/256/512状态探针；U另有64步探针 |
| `artifacts/*/runtime/scope.jsonl.gz`、`trace_summary.json.gz` | 压缩运行事件及逐 transition 摘要 |
| `artifacts/*/analysis/`、`analysis_partial/` | 行为/覆盖、Critic、奖励分解，partial不替代最终结果 |
| `artifacts/*/evaluation_*.json`、逐集 `result.json`/`config.yaml` | 固定评估与训练逐集证据 |
| `repair_decision.json`、`C_DECISION.json`、各报告和CSV | 门槛决策、配对结果与机制归因 |
| `frozen/`、`*GATE.json`、`C_REWARD_AUDIT.json` | 数据/配置、fork与恢复审计、历史奖励/GAE核验 |

**异常退出如实保留：** A/B 主日志均有 OpenGL context 清理错误，不能将 COMPLETE 视作干净退出。C/C0 明确记录退出码134，发生在结果保存之后；512条训练及每支64集无failure评估单独核对。GPU采样是整卡功耗，不能冒充每分支独占能耗。

## 权重与缩减边界

按用户要求，所有权重保留本地，仅发布 **83 个 `*.pt.placeholder.json`**，覆盖 frozen forks、运行 checkpoint、Actor初始权重和策略 snapshots。占位含本机路径、字节数、发布时实测 SHA256、`uploaded:false`；不含可加载权重。所依赖 Task23 checkpoint 沿用以前发布的占位。需从可信原机器取得文件并验 SHA256，才能加载/恢复。

约27GB原运行产物不整包上传，排除完整 PT traces、episode boundaries、重复逐步日志、缓存和空日志。不上传官方第三方仓库、场景资产和基础模型。保留小型固定 targets 是数据归档，并不表示可以用旧buffer训练。

`FILES_MANIFEST.tsv.gz` 原样描述全量本机产物，不表示所有文件已上传。实际发布映射见 `TASK232_SOURCE_MAP.json`，排除项见 `TASK232_EXCLUDED_FILES.json`；排除项历史hash来自原清单，未在发布时重读所有27GB文件。累计 `FILE_MANIFEST.json` 校验实际发布内容。

`DATA_MANIFEST.json`、trace摘要与事件流等以gzip保存；原脚本读取原文件名时需先在独立实验副本解压。原脚本还包含机器路径并依赖真实权重、完整历史traces和双环境。此缩减包不支持独立完整训练重放；源 `test_repair.py` 中完整fork测试需要本地真实权重，不能拿占位运行。原测试不为发布而偷偷跳过或弱化，发布校验使用下面的专用验证器。

## 只读发布验证

```bash
python3.11 tools/verify_task232_release.py
python3.11 -m unittest discover \
  -s tests \
  -p 'test_*archive.py'
```

验证器使用 Python 3.11 标准库，不加载 pickle，不调用 GPU；核验训练/更新/评估分母、结果指标、门槛结论、原始审计和占位哈希。历史完整恢复与数值审计不是此次发布重新执行训练所得。

发布时验证记录在 `docs/release/task232_validation/`：原项目使用本地可信权重完成19项修复/覆盖奖励测试（含完整fork参数及optimizer动量一致性）；发布目录通过78项模型相关回归测试与24项归档测试，共121项。另用原统计函数只读重算8组配对bootstrap/McNemar以及A/B扩训、C探索门槛，结果与原记录完全一致。没有运行会覆盖原交付物的汇总主程序。

全部本次新增/修改文件见 `docs/release/TASK232_CHANGED_FILES.txt`。
