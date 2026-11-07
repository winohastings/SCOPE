# Stage2 / Task231 发布说明

本次发布失败归因与匹配反事实诊断，不启动训练，不修改 Task23 原始记录。七个 Python 文件（六个实现脚本、一个测试脚本）及原始文档保持源内容。依赖的 Task21/22/23 顶层 Python 实现与现有发布版本逐文件比较一致，无需覆盖历史实现。

## 核心数据与结论

- 五个 checkpoint、train/val_seen 各 128 个固定状态，共 1280 次 Actor 探针。
- 历史五轮固定 32 集，共 160 条配对记录；数值审计覆盖 4096 训练 transitions、56 轮更新。
- 新反事实诊断为 1033/2054/4096 × native/forced4 × 32 集，共 192 集。最终数据来自两个完整分片；早期中断目录只保留证据，不追加到分母。完整分片复用早期已完成 episode，原 trace 路径可能跨目录。
- 匹配诊断为 8 episodes、16 states、128 branches，恢复检查全部有效。
- 4096 forced4 的 SR 从 native 的 34.375% 恢复到 53.125%，配对 exact McNemar p=0.109375，不能宣称显著提升或泛化。
- 1033→2054 在 forced4 下仍从 56.25% 降至 43.75%；STOP 次数不足以解释这段退化。局部 ROI 进展差异的置信区间跨 0，不据此证明 Adapter 失效。
- 相同导航结果下，4→0 Focus 可增加 0.04 reward，支持存在省感知动机，但不能确证为全部退化的原因。感知 STOP 与导航终止是不同机制。

## 文件与日志入口

以下路径相对于 `Experiment/SCOPE/Stage2/Task231/`：

| 文件 | 保留内容 |
|---|---|
| `README.md`、`REPORT.md`、`diagnosis.json` | 原始执行说明、结论、假设判定与边界 |
| 顶层 `*.csv` | 策略探针、配对结果、反事实、匹配动作/场景、阈值敏感性、reward 分解 |
| `artifacts/offline/` | 数值逐步记录、梯度、融合审计、来源和环境证据 |
| `artifacts/supplement/`、`artifacts/gradient_components/` | held-out Critic 与 policy/entropy 梯度分解 |
| `counterfactual32_shard0.log`、`counterfactual32_shard1.log` | 两个完整分片的主执行日志 |
| `matched8.log`、`offline.log` | 匹配对照与离线诊断日志 |
| `artifacts/*/server.log` | 各次运行服务端关键日志，包含失败/中断尝试 |
| `artifacts/*/runtime/scope.jsonl.gz`、`trace_summary.json.gz` | 压缩运行事件与逐步摘要 |
| `artifacts/*/counterfactual_results.json`、`matched_results.json` | 原始结果，保留完成状态和复用来源 |

Task231 没有新增训练，不能把诊断日志称为新训练日志。Task23 正式训练日志继续保留在已有归档。

## 权重与排除项

checkpoint 保留原机器本地；Task231 的 23 个 Actor 初始/快照文件仅上传 `*.pt.placeholder.json`，包含本地来源、字节数、实测 SHA256 与 `uploaded: false`。Task231 使用的五个 Task23 训练 checkpoint 沿用原 Task23 占位，未上传权重。加载需取得可信原文件并核验 SHA256，占位不能由 torch 加载。

约 11GB 完整运行产物不整包上传：排除 PT traces、episode boundaries、冗余逐步 JSON、缓存和空日志。保留逐集结果/配置与关键 CSV/JSON，不包含官方第三方仓库、场景数据或基础模型。

原 `FILES_MANIFEST.tsv` 以 `.gz` 原样保留，描述的是原机器全量产物，不代表所有文件都已上传。实际上传映射见 `TASK231_SOURCE_MAP.json`；省略文件见 `TASK231_EXCLUDED_FILES.json`，其中历史 SHA256 来自原清单，未在本次发布重新读取全部 11GB 被排除数据计算。实际发布文件由累计 `FILE_MANIFEST.json` 校验。

源脚本含原机器绝对路径，并需要两套环境、真实权重和完整 Task23 traces。缩减包只支持离线证据校验，不保证独立完整重跑诊断。原完整数值/冻结审计为历史证据，本次未重跑 GPU 导航或训练。`summarize.py` 会更新原交付文件，验证发布包应使用下面的只读验证器。

## 发布校验

```bash
python3.11 tools/verify_task231_release.py
python3.11 -m unittest discover \
  -s tests \
  -p 'test_*archive.py'
```

验证器不加载 pickle，不需要 GPU，检查累计文件哈希、192 集分母与配对、CSV/JSON 指标一致性、恢复门禁、1280 次探针、历史数值审计及 checkpoint 占位。

发布时通过 78 项模型相关回归测试、5 项 Task231 诊断测试和 19 项归档测试。另以原始汇总函数重新计算六组反事实统计和 matched 统计（含配对/bootstrap），与 `diagnosis.json` 完全一致；未运行会覆盖交付物的汇总主程序。日志保存在 `docs/release/task231_validation/`。全部本次新增/修改文件见 `docs/release/TASK231_CHANGED_FILES.txt`。
