# SimpleNav R2R-CE / RxR-CE 基线单集闭环验证

## 范围与结论

2026-11-05，在 `/home/data/czh/Third_Party/baselines/SimpleNav` 使用现有 RXR step-8000 checkpoint，完成 R2R-CE 和 RxR-CE 的 `val_unseen` 固定四视角闭环评测。每个任务先运行一集，再在最终代码上重复同一集；不是挑选不同 episode，也不是全量性能实验。

两任务均完成模型加载、相机观测、WebSocket 推理、8-waypoint 执行、模型停止、指标计算和 artifacts 保存。两次运行的导航指标一致；两任务最终均为完成 1 集、失败 0 集、未完成 0 集、SR=1。这里只确认本机单集端到端可用，不代表全量评测无异常或性能已复现。

根目录引用的 `agents/exec-agent.md` 不存在；本次按照用户明确授权进行调试和执行，未提交 git commit，保留仓库原有修改。

## 环境和模型

### Task1 R2R 后续记录

官方 `docs/guides/VLNCE_TRAINING_AND_EVALUATION.md` 明确：RxR 训练的同一份
step-8000 checkpoint 同时评测 R2R `val_unseen` 和 RxR `val_unseen`。
本日志中的 RXR 权重跑 R2R 就是该共享权重口径，不需要独立 SimpleNav-R2R 权重。
后续 Task1 的 R2R 多分辨率六模式独立执行与固定观测等价性见
`Task1/R2R/implementation_report.md`，原始 episode、逐步插桩和成本 CSV 同目录。
这项后续工作不替换本日志的既有单集基线结果，也不宣称已完成全量评测。

| 项目 | 实际值 |
|---|---|
| checkpoint | `local/checkpoints/vlnce/navvla_qwen35_cpm_rxr/checkpoints/steps_8000_pytorch_model.pt` |
| backbone | `/home/data/czh/Third_Party/models/Qwen3.5-4B` |
| 模型服务 | `deployment/model_server/legacy_rxr_server.py` |
| 兼容模型代码 | `/tmp/SimpleNav-RXR-official`；它是独立运行依赖，重启服务前必须确认仍存在 |
| 模型 Python | SimpleNav `.venv/bin/python`，Python 3.10 |
| Habitat Python | `/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python` |
| Habitat 版本 | Habitat-Lab 0.3.1，Habitat-Sim 0.3.3 |
| GPU | 模型 GPU 1，Habitat worker GPU 0 |
| 服务地址 | `ws://127.0.0.1:10093` |
| 相机 | 固定 `front/left/right/rear`，256×256，未开启 AdaPerc 随机传感器 |
| 动作与统计量 | `anchor_relative_body_frame_xyz_yaw`，horizon 8，`vln_train_train` |
| 停止 | `mean_adjacent_waypoint_translation <= 0.03` |

不是将原始 Qwen 权重直接当导航模型，也不是验证当前通用加载器对旧权重的兼容性；使用的是已训练 SimpleNav RXR 导航 checkpoint 和现有 legacy 兼容服务。

## 修复

SimpleNav 路径相对于 `/home/data/czh/Third_Party/baselines/SimpleNav`。

| 文件 | 本次变化 |
|---|---|
| `benchmark/vlnce/inputs.py` | R2R 输入同时接受标准 `datasets/`、旧 `R2RCE/`、本机 `R2R-CE/`；兼容 gzip 和普通 JSON；空标准目录不会遮蔽有效本地文件 |
| `benchmark/common/simulators/habitat/vlnce031_datasets.py` | Habitat 的 R2R annotation / GT 路径采用相同候选目录策略 |
| `benchmark/common/runner/worker.py` | 将 RXR 稠密 GT nDTW 逻辑推广到 R2R；两任务均按 GT `locations` 和去除连续重复位置后的预测轨迹计算 |
| `benchmark/vlnce/r2r/config_standard_local.yaml` | 新建本机标准固定四视角 R2R 配置；指定正确数据、场景和 worker Python；默认 `max_samples: null` |
| `tests/test_vlnce_r2r_paths.py` | 新建目录、JSON/gzip、annotation/GT、空目录回退回归测试 |
| `tests/test_vlnce_ndtw.py` | 新建两任务稠密 GT 加载和连续重复位置去除测试 |

最初 R2R 汇总 nDTW=0.2101214777，来自稀疏 `reference_path`；同一条轨迹的 Habitat nDTW=0.8993868291。修复并重新闭环运行后，汇总 nDTW=0.8993868294，与 Habitat 一致。没有仅手工改写历史 summary。

## 最终结果

| 任务 | episode | 场景 | 完成/失败 | policy steps | SR | NE (m) | SPL | nDTW |
|---|---|---|---|---:|---:|---:|---:|---:|
| R2R-CE val_unseen | 1 | zsNo4HB9uLZ | 1 / 0 | 8 | 1 | 0.935289 | 1.000000 | 0.8993868294 |
| RxR-CE val_unseen guide English | 4 | 2azQ1b91cZZ | 1 / 0 | 54 | 1 | 1.407199 | 0.104104 | 1.3074765687e-14 |

两任务终止原因均为 `waypoint_motion_stop`，不是强制缩短最大步数。RXR 虽最终导航成功，但路径长达 67.66 m，相比参考路径绕路明显，SPL 和 nDTW 低；不能据单集成功宣称路径质量好。

R2R 保存 7 份动作 step JSON、57 个轨迹位姿；RXR 保存 53 份动作 step JSON、425 个轨迹位姿。停止预测本身不执行动作，不单独生成动作 step JSON，因此 policy steps 比动作文件数多 1。

全量 R2R `val_unseen` dry-run 成功解析 1839 集、11 个场景；检查真实场景文件及 GT episode 覆盖，无缺失。没有执行这 1839 集的全量模型推理。

## 结果与校验产物

SimpleNav 最终运行目录：

- `local/eval_results/vlnce/simplenav_r2rce_standard_verified_20261105/`
- `local/eval_results/vlnce/simplenav_rxrce_standard_verified_20261105/`

每个目录包含解析后的配置、episode / worker 计划、worker 日志、summary、`eval_info.json`、`trajectory.json`、动作 step JSON。

本项目 `outputs/baseline_smoke_20261105/`：

- `verification.json`：最终两任务结果，以及动作有限值、8×4 维度、固定传感器、正常停止、nDTW 与 Habitat 一致、复跑导航指标一致的断言结果。
- `regression_tests.log`：最终 5 项回归测试全部通过。
- `path_tests.log`、`ndtw_tests.log`：分项回归测试输出。
- `r2r_full_dry_run.json`：1839 集全量规划。
- `model_server.log`：模型加载和服务日志。
- `r2r_eval.log`、`rxr_eval.log`：初次运行输出。
- `r2r_verified_eval.log`、`rxr_verified_eval.log`：最终代码复跑输出。

初次运行保留在 SimpleNav 的 `simplenav_r2rce_standard_smoke_20261105` 和 `simplenav_rxrce_standard_smoke_20261105` 目录，不覆盖已有 50 集实验。

注意 `summary.json.pending_episodes` 表示本次启动规划时待运行的数量，不是运行结束后剩余数量；本次结束时 `unresolved_episodes=0`。

## 复现命令

所有命令先进入 SimpleNav 根目录。两个任务共用一个模型服务；不要重复占用 10093 端口。交付时服务仍运行在 GPU 1，下面启动命令用于服务未运行时。

```bash
cd /home/data/czh/Third_Party/baselines/SimpleNav
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/data/czh/Third_Party/baselines/SimpleNav
.venv/bin/python deployment/model_server/legacy_rxr_server.py \
  --checkpoint local/checkpoints/vlnce/navvla_qwen35_cpm_rxr/checkpoints/steps_8000_pytorch_model.pt \
  --legacy-root /tmp/SimpleNav-RXR-official \
  --qwen-root /home/data/czh/Third_Party/models/Qwen3.5-4B \
  --port 10093
```

另一个终端运行评测；使用新的 run name，否则框架会复用/跳过已完成的 episode。worker 按配置使用物理 GPU 0。

```bash
cd /home/data/czh/Third_Party/baselines/SimpleNav
export PYTHONPATH=/home/data/czh/Third_Party/baselines/SimpleNav
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python benchmark/vlnce/r2r/eval.py \
  --config benchmark/vlnce/r2r/config_standard_local.yaml \
  --override benchmark.max_samples=1 \
  --override output.run_name=simplenav_r2rce_standard_recheck

/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python benchmark/vlnce/rxr/eval.py \
  --config benchmark/vlnce/rxr/config_standard_local_50.yaml \
  --override benchmark.max_samples=1 \
  --override output.run_name=simplenav_rxrce_standard_recheck
```

R2R 全量入口已准备好，以下命令没有单集限制，但本次未执行：

```bash
/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python benchmark/vlnce/r2r/eval.py \
  --config benchmark/vlnce/r2r/config_standard_local.yaml \
  --override output.run_name=simplenav_r2rce_standard_full
```

运行本次新增回归测试：

```bash
/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python -m unittest discover \
  -s tests \
  -p 'test_vlnce_*.py'
```

## 边界

- 单集验证不能证明全量性能、全部场景稳定性或等同发布分数。
- SR、SPL 等沿用现有 SimpleNav evaluator 与模型停止规则；步骤中 Habitat 原生 `success/spl` 在发出 STOP 前可能仍为 0，不能将动作前的这些值直接当最终汇总。
- 保留 Qwen processor、Gym、图形驱动相关警告；没有为本任务改动无关依赖。
- 旧模型兼容实现仍依赖 `/tmp/SimpleNav-RXR-official`，不是自包含的新模型发布包。
