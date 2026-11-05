# SCOPE Glance–Focus 执行入口

实现与实测报告：`../../docs/Logs/Task1/implementation_report.md`。

R2R 专项报告：`../../docs/Logs/Task1/R2R/implementation_report.md`。
R2R 与 RxR 按官方指南共用 RXR step-8000 权重，不需要另一个 R2R checkpoint。

本目录保留已验证的 SimpleNav RXR legacy 运行时副本，来源为本机原有
`/tmp/SimpleNav-RXR-official`。原因是现有 checkpoint 实际加载该版本，而非
SimpleNav 主干中的新模型接口。无需再依赖容易被清理的 `/tmp` 运行时代码。
`legacy_runtime` 的逐文件来源、原始及当前 SHA256 见报告目录的 `runtime_provenance.json`。
权重、Qwen 模型及 Habitat 环境没有复制，仍使用本机既有路径。

## 启动服务

在 AdaPercVLN 根目录执行，物理 GPU 0 与原有物理 GPU 1 的 10093 基线服务相互独立：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/scope_server.py \
  --checkpoint /home/data/czh/Third_Party/baselines/SimpleNav/local/checkpoints/vlnce/navvla_qwen35_cpm_rxr/checkpoints/steps_8000_pytorch_model.pt \
  --telemetry /home/data/czh/AdaPercVLN/docs/Logs/Task1/recheck_server_steps.jsonl \
  --smoke-selector \
  --port 10094
```

`--smoke-selector` **不是已学习策略**。仅用于验证 `learned_2` / `learned_adaptive`
调用、顺序选择和 STOP 的链路。真实策略必须改用 `--selector /absolute/path/selector.py`。
不提供二者时，learned 模式显式报错，绝不静默替换为 fixed/random。

选择器文件导出 `select(*, glance, selected, available, allow_stop, focus_index, rng)`：
返回候选 ID 0–15 或 STOP=16；`selected` 按序包含先前获得的 ROI 及原分辨率裁剪图像，
`available` 排除重复候选。每步最多调用两次；非 adaptive 模式禁止 STOP。
Fixed-grid 选择 front.R1、front.R4（候选 0、3）。Random 的随机源由
seed / episode ID / prediction step 的 SHA256 派生，不依赖 Python 随机哈希。

## 冒烟与聚合

```bash
/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/run_smoke.py \
  --output docs/Logs/Task1_recheck \
  --max-steps 2
```

两任务各六模式，每组相同两集、每集最多两个策略步。它会设置四视图 512×512，
其余几何、seed、8-waypoint 执行、动作统计量和停止规则沿用现有配置。
两步上限是显式的**冒烟截断**，不是完整导航评测或论文性能结果。
原始 episode artifacts 存在输出目录的 `runs/`，聚合表为 `aggregate.csv`。
使用新输出目录可避免框架复用已完成结果。

仅执行 R2R 六模式（其余默认行为不变，省略 `--task` 仍执行两个任务）：

```bash
/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/run_smoke.py \
  --output docs/Logs/Task1/R2R_recheck \
  --task r2r \
  --max-steps 2
```

R2R 本次独立产物位于 `docs/Logs/Task1/R2R/`。`--max-steps` 为显式冒烟上限，
不是官方完整评测步数；如需正式评测，另行按官方配置确定协议，不将冒烟指标当正式分数。

```bash
/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/run_smoke.py \
  --output docs/Logs/Task1 \
  --aggregate-only

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope_*.py'
```

真实固定观测检查需要原有未改动 10093 服务和本目录 10094 服务同时在线：

```bash
/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/verify_fixed.py \
  --output docs/Logs/Task1/recheck_equivalence.json \
  --observations docs/Logs/Task1/fixed_habitat_observation.npz
```

SimpleNav 外部仓库的插桩修改仅限 runner worker、Habitat backend/runtime；
可重放差异见报告目录的 `benchmark_instrumentation.patch`，修改前文件保留在
`before_sources/`。原有仓库其他未提交修改均未回退，也未提交 commit。
