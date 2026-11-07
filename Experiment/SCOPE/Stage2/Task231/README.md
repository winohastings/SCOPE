# Task2.3.1：失败归因与匹配反事实诊断

执行者任务，包含观测层分析，不运行训练更新，不修改 `Task23/`。最终结论见
`REPORT.md` 与 `diagnosis.json`。`agents/exec-agent.md` 在本仓库缺失，依据用户授权、
根 `AGENTS.md`、Task2.3.0 与 Task2.3.1 文档执行。

## 入口与数据契约

| 脚本 | 功能 |
| --- | --- |
| `diagnose.py` | 真实checkpoint/原始产物SHA验证、五轮配对统计、固定状态Actor探针、4096条数值审计 |
| `counterfactual.py` | 三checkpoint的native/forced4真实导航；按场景执行，可分片、复用已完整结束的诊断episode |
| `server.py` | 只在新诊断进程内覆盖动作选择；禁止optimizer更新；记录匹配恢复与teacher特征检查 |
| `matched.py` | 独立reset+固定导航前缀的1-chunk learned/fixed/random/no-Focus对照 |
| `supplement.py` | held-out Critic、训练采样Focus统计与成本算术 |
| `summarize.py` | 汇总6组各32集与匹配对照，生成CSV、diagnosis、报告及全文件清单 |
| `test_diagnostics.py` | 配对统计、bootstrap、reward成本及GAE边界回归 |

依赖本机已有真实权重、Habitat数据、双环境以及Task23原始运行目录。
GitHub占位checkpoint不能替代真实权重。输出目录必须位于本目录下；执行脚本拒绝
覆盖已有运行目录。汇总脚本只更新本目录下的最终交付文件。

- 固定状态：train和val_seen各128条，五模型共享完全相同的detached ActorState；reward/GT不进入Actor。
- STOP count代理：`stationary_p_expected_focus_proxy`假设后续STOP概率保持首状态值；
  `first_decision_deterministic_focus_proxy`不是新的真实闭环Focus count。真实次数见导航结果。
- `paired_episodes.csv`为历史五轮长表，通过`episode_uid`配对；CI以episode为抽样单位。
- SR沿用Task23：真实导航终止且距离小于3米；截断时在3米内不自动算成功。
- 新评估固定原32集，先12集探索，因该子集无法复现1033→2054净成功数下降而扩至32集。
- continuation只复用落盘的完整episode；中断episode的trace保留但不计数。两GPU分片
  按预先固定的scene round-robin顺序取偶/奇索引，不按成功率分配。
- forced4不改变模型参数、camera/position输出或导航种子，只忽略感知STOP选择。
- matched固定ROI：四相机各中央一次；随机ROI：三个固定种子，相机均匀、合法中心均匀。
  三种随机结果先在同state平均，再按episode聚合bootstrap，避免把分支当独立episode。
- matched独立reset并重放相同1033/no-Focus首chunk前缀；比较姿态、全部图像历史、
  指令、pre-Focus tensor以及模型实际BATS context blocks。无效恢复分支不合并。
- teacher为新编码的512px四视角全图特征，面积对齐至4×4当前网格；teacher仅在导航
  动作产生后计算误差，不传给Actor或导航模型作输入。
- 数值审计覆盖全部4096条旧训练transition；head梯度在每update每Focus count首个
  样本上取整个head的未裁剪范数，属于post-update重放，不是优化器历史梯度的伪复原。
- `artifacts/gradient_components/head_gradient_components.csv`进一步分离policy/entropy
  梯度；STOP-only分支也可能因为entropy正则具有camera/position梯度，不能误作动作信用。
- `source_integrity.json`最终重验原run全部15918个归档产物与6个Actor源文件；
  `habitat_environment.json`、`artifacts/offline/provenance.json`记录当前两套环境。
  原run没有完整package lock，因此历史环境逐包一致性不作保证，以native重放实测补充。
- 不使用val_unseen，不重新校准Adapter，不进行正式优化。

## 重跑命令

从仓库根目录执行，替换输出目录名称避免覆盖。模型环境是Python3.10，Habitat环境是
Python3.9；不要混用。建议设置`PYTHONDONTWRITEBYTECODE=1`，避免触碰其他目录缓存。

### 离线诊断

```bash
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task231/diagnose.py \
  --output Experiment/SCOPE/Stage2/Task231/artifacts/reproduce_offline \
  --device cuda:1 \
  --states 128
```

```bash
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task231/supplement.py \
  --output Experiment/SCOPE/Stage2/Task231/artifacts/reproduce_supplement \
  --device cuda:1
```

### 完整3×2×32集诊断

将policy/entropy梯度分开审计（读取已完成离线诊断的分层样本）：

```bash
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task231/supplement.py \
  --output Experiment/SCOPE/Stage2/Task231/artifacts/reproduce_gradient_components \
  --device cuda:1 \
  --offline Experiment/SCOPE/Stage2/Task231/artifacts/reproduce_offline \
  --gradients-only
```

以下为单进程版本，不依赖已有中间结果。显存、数据均已就绪时可使用`--shards 2`
和不同`--shard-id`、`--model-gpu`、`--port`启动两个不重叠分片。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task231/counterfactual.py \
  --output Experiment/SCOPE/Stage2/Task231/artifacts/reproduce_counterfactual32 \
  --episodes 32 \
  --port 10231 \
  --model-gpu 0
```

### 匹配1-chunk对照

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task231/matched.py \
  --output Experiment/SCOPE/Stage2/Task231/artifacts/reproduce_matched8 \
  --episodes 8 \
  --port 10232 \
  --model-gpu 1
```

### 汇总与测试

汇总已有本次完整产物（默认读取两个32集分片、offline、matched8和supplement）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task231/summarize.py
```

汇总独立重跑产物：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task231/summarize.py \
  --offline Experiment/SCOPE/Stage2/Task231/artifacts/reproduce_offline \
  --counterfactual Experiment/SCOPE/Stage2/Task231/artifacts/reproduce_counterfactual32 \
  --matched Experiment/SCOPE/Stage2/Task231/artifacts/reproduce_matched8 \
  --supplement Experiment/SCOPE/Stage2/Task231/artifacts/reproduce_supplement
```

```bash
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest \
  Experiment.SCOPE.Stage2.Task231.test_diagnostics
```

完整文件路径、大小、SHA256见`FILES_MANIFEST.tsv`。原始运行和失败/中断尝试保留在
`artifacts/`；`INCOMPLETE.json`不可当作完整结果，`COMPLETE.json`也不替代配对和冻结核验。
