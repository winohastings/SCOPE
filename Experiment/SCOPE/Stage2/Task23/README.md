# Stage2 Task2.3：导航级 PPO-only Actor–Critic

执行者角色；冻结协议见 `PROTOCOL.md`，固定参数见 `config.json`。
最终执行事实以 `REPORT.md` 和对应运行目录中的 JSON/trace/checkpoint 为准。
本目录不实现 Group Sampling、RLOO 或 GA-PPO，不访问 val_unseen。

已执行4096真实train transitions及5轮固定验证，详见`REPORT.md`。
最佳checkpoint来自1033步；最终4096步模型确定性零Focus退化，不能宣称效果验收通过。
正式run结束清理时有OpenGL exit134，已独立验证全部轨迹和最终checkpoint完整；
该限制保留在原始日志和报告中，不应把COMPLETE理解成进程干净退出。

## 组件

| 文件 | 职责 |
| --- | --- |
| `critic.py` | 独立128 hidden导航价值网络，只使用 Focus 前的 Glance/语言状态 |
| `learning.py` | 导航奖励、terminal/truncation GAE、Task2.2逐决策概率PPO、独立优化器及RNG恢复 |
| `data.py` | 授权train与预先固定的32集scene-balanced val_seen清单和来源哈希 |
| `runtime.py` | 继承Task2.2真实感知Actor，冻结审计、on-policy buffer、checkpoint及选模 |
| `server.py` | session.v1推理/反馈/训练RPC；仅在episode边界允许更新与恢复 |
| `run.py` | 复用Task4反馈与官方Habitat worker；真实门禁、正式训练、周期验证及成本采样 |
| `audit.py` | 恢复前后参数、优化器、RNG、数据游标逐位一致性审计 |
| `summarize.py` | 训练与验证分开汇总、奖励/轨迹核验、CSV及SVG曲线 |

唯一新视觉接口是已冻结Task2.1 Adapter与IncrementalPerception，Actor沿用Task2.2。
SimpleNav、视觉编码器、语言embedding、Adapter均冻结；当前视觉接口始终64 token。
感知四次Focus上限不是导航terminal；导航截断需要真实后继Glance估值。
旧Task4的17类离散Actor和PPO更新器不在训练路径中。

## 环境与复现

从仓库根目录执行。模型环境Python3.10与Habitat环境Python3.9分离。
GPU0运行冻结模型/训练器，GPU1运行Habitat。输出目录必须不存在；不覆盖历史结果。
首次生成数据清单后，后续运行传入同一清单；程序会将其原字节复制到新输出目录。

### 必须先执行真实训练门禁

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task23/run.py \
  --output Experiment/SCOPE/Stage2/Task23/artifacts/reproduce_gate \
  --stage gate \
  --port 10123 \
  --manifest Experiment/SCOPE/Stage2/Task23/artifacts/gate/DATA_MANIFEST.json
```

门禁独立采集64个真实导航transitions，真正执行更新；恢复更新前checkpoint并重做
相同更新，比较参数/优化器/RNG逐位一致。这次重复更新不算新增导航数据。
只有`TRAINING_GATE.json`中`passed=true`才允许启动正式训练。

### 初始4096主轨迹预算

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task23/run.py \
  --output Experiment/SCOPE/Stage2/Task23/artifacts/reproduce_ppo4096 \
  --stage train \
  --port 10123 \
  --manifest Experiment/SCOPE/Stage2/Task23/artifacts/gate/DATA_MANIFEST.json \
  --gate Experiment/SCOPE/Stage2/Task23/artifacts/reproduce_gate/TRAINING_GATE.json
```

先评估初始化，再每约1024已训练transitions验证，最后再次验证。
固定32集val_seen不是完整778集官方评估。SR优先、SPL次之选模，持平保留早期模型。
即使4096预算完成，单seed/小验证子集也不能证明方法有效或收敛。

### 恢复与后续预算

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task23/run.py \
  --output Experiment/SCOPE/Stage2/Task23/artifacts/resumed_ppo4096 \
  --stage train \
  --port 10123 \
  --manifest Experiment/SCOPE/Stage2/Task23/artifacts/ppo4096/DATA_MANIFEST.json \
  --gate Experiment/SCOPE/Stage2/Task23/artifacts/gate_rng_isolated/TRAINING_GATE.json \
  --resume Experiment/SCOPE/Stage2/Task23/artifacts/ppo4096/runtime/checkpoints/latest.pt \
  --budget 4096
```

完整checkpoint包含live/old Actor和Critic、两个optimizer、所有RNG、配置/来源哈希、
主轨迹/验证计数、数据游标、best及未更新buffer的episode文件引用。
必须保留原运行目录，checkpoint中的buffer引用不是独立打包副本。
只能在episode边界恢复；中断episode或部分optimizer更新不得直接续用。
恢复后新目录日志只含新增部分，累计计数来自checkpoint；跨段分析需合并原始运行。
`--budget 10000`或`--budget 20000`保留为后续显式授权选项，本任务不自动扩预算。

### 测试与结果审计

```bash
OMP_NUM_THREADS=2 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope*.py'

OMP_NUM_THREADS=2 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task23/summarize.py \
  --run Experiment/SCOPE/Stage2/Task23/artifacts/ppo4096 \
  --verify-traces

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task23/audit.py \
  --compare Experiment/SCOPE/Stage2/Task23/artifacts/gate_rng_isolated/runtime/checkpoints/gate_after.pt \
  --replay-before Experiment/SCOPE/Stage2/Task23/artifacts/gate_rng_isolated/runtime/checkpoints/gate_before.pt \
  --output Experiment/SCOPE/Stage2/Task23/artifacts/gate_rng_isolated/FRESH_PROCESS_RESTORE.json
```

`COMPLETE.json`要求4096已训练且buffer为空；`INCOMPLETE.json`表示尚未完成。
`runtime/checkpoints/best_val_seen.pt`是预声明规则选出的最佳模型，可能仍是初始化。
`latest.pt`是完整恢复点，`update_*.pt`保留周期验证时的完整状态。
`runtime/traces/*.pt`与episode boundary文件保留Task2.2协议及PPO字段；
`runtime/updates/`包含所有更新指标与固定advantage/return；`summary/`包含曲线和成本。
完整checkpoint使用pickle读取，只能加载自己生成且可信的文件。
