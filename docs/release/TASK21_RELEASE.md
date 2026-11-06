# Stage2 Task2.1 增量发布

本次按仓库整理与发布职责执行。项目根AGENTS、索引与任务书已读，角色文件仍缺失；不伪造缺失流程。不重新执行GPU采集、校准或导航评估，不改AdaPercVLN原始文件。

## 代码与关键证据

- 原样新增 `fusion.py`、`runtime.py`、`capture.py`、`calibrate.py`、`config.json`、README及REPORT，新增原14项Task21单元测试。
- 保留Task2.1任务书与Stage2终态不变量文档；后者描述未来Actor/RL目标，不代表Task21已完成这些目标。
- 最终 `artifacts/release/` 完整保留adapter、calibration、validation与scope日志；采集manifest保留8个状态来源及图像SHA256，但不上传RGB NPZ。
- 保留最终release、capture、48项相关测试及已知全套测试失败日志；不上传被最终release替代的首轮calibration副本与Python缓存。
- 最终Adapter仅2204字节，是任务明确要求的校准交付物，作为 `.gitignore` 的精确例外纳入Git。它不是SimpleNav/Qwen/Actor/Critic权重，不扩大其他权重上传范围。

## 结果边界

来自R2R train的8个初始状态：episode1–6用于校准，7–8用于诊断，96/32样本；诊断场景与校准场景并不独立。120步仅优化一个门参数，gate=0.7967281342；BF16诊断覆盖MSE为0.001231112，非覆盖误差0。6次无Focus同状态回放特征/动作差均0，N=0..4当前Qwen token均64；额外ViT编码量仍为64+64N。

只证明短校准与接口合同，不证明SR/SPL改善、跨场景泛化、完整训练或完整闭环评估。旧Task4/Stage1数据、源码及依赖补丁保持原样。

## 发布清单与复现

`TASK21_SOURCE_MAP.json` 列出所有原样新增文件和源SHA256，`TASK21_EXCLUDED_FILES.json` 列出排除项及hash；`FILE_MANIFEST.json` 为累计发布清单，`SOURCE_MAP.json` 保留累计来源。报告中的 `FILES_MANIFEST.tsv` 指原实验完整清单，本发布包以这些JSON清单为准。

`tools/verify_task21_release.py` 使用标准库验证累计文件、Adapter hash、96/32样本、120步、6次无Focus、64-token与冻结门禁；不反序列化checkpoint。模型Python可用 `torch.load(..., weights_only=True, map_location="cpu")` 对可信checkpoint做参数/溯源核验，不需要GPU。

采集图像、官方依赖和导航权重须另行准备；仅有发布包不能直接重跑图像校准。旧代码保留机器路径，不能冒称跨机器开箱即用。复用既有依赖重建工具，不新增官方源码补丁。

## 本次发布验证

- 重新运行48项SCOPE测试（含14项Task21）及4项Stage1协议回归，全部通过；日志见 `TASK21_SCOPE_REGRESSION.log`、`TASK21_PROTOCOL_REGRESSION.log`。
- CPU安全加载最终Adapter，仅1个有限参数，sigmoid为0.7967281342；数据划分、采集manifest、配置与episode ID完全匹配。结果见 `TASK21_CHECKPOINT_VALIDATION.json`。
- 新增3项Task21归档测试验证真实产物、checkpoint hash不匹配拒绝与当前token预算漂移拒绝；同时保留原Stage1/Task4归档回归。
- 无GPU证据核验检查完整hash、96/32样本、120步、6次无Focus回放和N=0..4的64-token合同。历史全套的无关导入失败仍保留于原 `tests_all.log`，不宣称已修复。

```bash
python3.11 tools/verify_task21_release.py

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p test_scope_task21.py
```
