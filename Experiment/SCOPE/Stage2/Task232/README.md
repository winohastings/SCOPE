# Task2.3.2 定向修复与受控短续训

执行者任务；冻结规则见`PROTOCOL.md`，实际结果见`REPORT.md`及`repair_decision.json`。
只写Task232；不改变Task2.1/2.2结构，不重训Adapter，不启动GA-PPO。

用户后续新增C是**新增覆盖奖励探索**，不是原文的条件式Adapter修复C。
新增协议见`C_PROTOCOL.md`，结果见`C_REPORT.md`、`C_DECISION.json`；原`C_REVIEW`只保留旧语义。

## 文件与作用

| 文件 | 作用 |
| --- | --- |
| `prepare.py` | 核验真实1033 checkpoint，准备episode/trajectory独立评估集、固定探针和完整状态fork |
| `runtime.py`、`server.py` | 独立进程内参数化Focus成本、保留原PPO算法、禁止历史buffer更新、保存奖励分量 |
| `run.py` | 新真实rollout、原样fork门禁、256/512探针、A/B训练、两套评估及原模型阈值诊断 |
| `metrics.py` | STOP/camera/position/KL探针，真实及参考链条件ROI几何统计 |
| `audit.py` | 新旧前向状态/动作逐位核验，训练参数/optimizer/RNG/cursor恢复核验 |
| `analyze.py` | 新轨迹Focus/Coverage、held-out Critic、成本和reward汇总 |
| `matched_initial.py` | 独立32集相同reset初态逐位匹配与真实连续Focus/首chunk对照 |
| `decision.py` | 冻结门槛检查、配对bootstrap/McNemar、A+B证据与扩展决策 |
| `finalize.py` | 原4096阈值对照汇总、原始产物完整性复验、全量文件清单 |
| `c_review.py` | 从Task231原始记录复核C触发条件，不冒称已执行Adapter修复 |
| `test_repair.py` | 成本单因素、ROI指标、完整fork、独立清单和扩展门槛测试 |
| `prepare_c.py`、`novelty.py` | 完整状态C/C0分叉、逐Focus新增面积奖励及训练目标接入 |
| `report_c.py`、`test_novelty.py` | C/C0空间与导航独立判读、真实GAE目标重算、覆盖奖励回归测试 |
| `attribute_c.py` | 覆盖奖励量级、固定Critic GAE分解、camera输出bias梯度和几何反事实审计 |

`frozen/`含原样train顺序、原32+独立32评估集合、A/B/U配置、迁移后的完整checkpoint。
独立集不与原32共享episode或trajectory，但不保证场景独立，也不是永久未触碰测试集。
checkpoint保留AdamW动量和step；B仅改参数组LR；所有配置变更进入新provenance。
历史trace只用于固定探针和U前向审计，**不能送入更新器**。

## 环境与预算

- 模型/分析环境：`/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python`。
- Habitat环境：`Assumption/assumption1/envs/correctnav_hab033/bin/python`。
- 从仓库根目录运行；设置`PYTHONDONTWRITEBYTECODE=1`避免其他目录缓存写入。
- 本机原数据、权重与两个环境是运行前提，不能用GitHub占位文件代替checkpoint。
- 输出目录必须不存在；以下reproduce目录用于避免覆盖本次证据。
- A/B初始各512新增transition；1024必须提供已通过冻结门槛的decision和512完整checkpoint。
- 精确256/512预算可在episode边界主动截断，用真实后继Glance bootstrap，不冒称terminal。
- 原样U门禁64新增transition单独列示；同批新数据恢复重放不计为新增transition。
- `gpu.csv`为整卡功耗，包含并发renderer/其他进程，不作为分支独占能耗。

## 复现命令

已有`frozen/`不重新生成；如要完整重新准备，先在独立实验副本内运行`prepare.py`，
该脚本拒绝覆盖本目录现有frozen产物。不要手改冻结配置后冒称同一实验。

### U一致性门禁

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task232/run.py \
  --output Experiment/SCOPE/Stage2/Task232/artifacts/reproduce_gate \
  --branch U \
  --mode gate \
  --port 10240 \
  --model-gpu 0
```

### A：只降低Focus成本

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task232/run.py \
  --output Experiment/SCOPE/Stage2/Task232/artifacts/reproduce_A512 \
  --branch A \
  --mode train \
  --budget 512 \
  --port 10240 \
  --model-gpu 0 \
  --gate Experiment/SCOPE/Stage2/Task232/artifacts/reproduce_gate/TRAINING_GATE.json
```

### B：原奖励与PPO稳定性包

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task232/run.py \
  --output Experiment/SCOPE/Stage2/Task232/artifacts/reproduce_B512 \
  --branch B \
  --mode train \
  --budget 512 \
  --port 10241 \
  --model-gpu 1 \
  --gate Experiment/SCOPE/Stage2/Task232/artifacts/reproduce_gate/TRAINING_GATE.json
```

### 新C覆盖奖励与C0对照

`prepare_c.py`只运行一次生成C/C0完整fork，拒绝覆盖。已有`frozen/config_C*.json`时不要重跑准备。
以下运行C；C0将branch改为C0、output改为独立目录，同时换空闲port/model-gpu。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task232/run.py \
  --output Experiment/SCOPE/Stage2/Task232/artifacts/reproduce_C512 \
  --branch C \
  --mode train \
  --budget 512 \
  --port 10240 \
  --model-gpu 0 \
  --gate Experiment/SCOPE/Stage2/Task232/artifacts/unchanged_gate/TRAINING_GATE.json
```

保存进程退出码到run下`PROCESS_EXIT_CODE.txt`；只有完整512训练与两套32集评估才做正式分析。
两分支各运行`analyze.py --run RUN_DIR`，再运行：

```bash
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task232/attribute_c.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task232/report_c.py \
  --c Experiment/SCOPE/Stage2/Task232/artifacts/branch_C512 \
  --control Experiment/SCOPE/Stage2/Task232/artifacts/branch_C0512
```

最后运行`finalize.py`更新总报告与全量SHA清单。C固定512筛查，不自动扩至1024。

### 1033独立集基线

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task232/run.py \
  --output Experiment/SCOPE/Stage2/Task232/artifacts/reproduce_baseline \
  --branch U \
  --mode baseline \
  --port 10241 \
  --model-gpu 1
```

### 原4096模型阈值纯评估

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task232/run.py \
  --output Experiment/SCOPE/Stage2/Task232/artifacts/reproduce_thresholds \
  --branch U \
  --mode thresholds \
  --port 10240 \
  --model-gpu 0
```

### 分析、决策与测试

对baseline、A、B分别运行分析（以下示例为本次A）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task232/analyze.py \
  --run Experiment/SCOPE/Stage2/Task232/artifacts/branch_A512
```

```bash
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task232/decision.py \
  --baseline Experiment/SCOPE/Stage2/Task232/artifacts/baseline_independent \
  --a Experiment/SCOPE/Stage2/Task232/artifacts/branch_A512 \
  --b Experiment/SCOPE/Stage2/Task232/artifacts/branch_B512
```

```bash
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task232/matched_initial.py \
  --baseline Experiment/SCOPE/Stage2/Task232/artifacts/baseline_independent \
  --a Experiment/SCOPE/Stage2/Task232/artifacts/branch_A512 \
  --b Experiment/SCOPE/Stage2/Task232/artifacts/branch_B512
```

```bash
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task232/matched_initial.py
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task232/c_review.py
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task232/finalize.py
```

```bash
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest \
  Experiment.SCOPE.Stage2.Task232.test_repair
```

全部创建文件路径、大小、SHA见`FILES_MANIFEST.tsv`。训练与评估的原始trace、模型checkpoint、
restore门禁、配置和成本日志均保留，不用汇总表替代原始证据。
