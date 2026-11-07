# Task233 导航价值驱动的 AGM 恢复训练

入口协议：`docs/Tasks/Stage2/stage2_task2.3.3_v2_train_gates.md`。
执行状态以各新run目录内`PROGRESS_GATE.json`、`COMPLETE.json`和`teacher_signal_audit.json`为准；仅存在启动配置或ready文件不代表门禁通过。

## 已实现范围

Gate0只读审计：合法连续latent候选、冻结update14空间策略探针、当前位姿geodesic专家、真实8-waypoint chunk执行、逐分支reset/prefix replay、真实递增Focus历史、独立噪声校准、teacher保留集配对bootstrap、自动失败关闭。

`teacher_candidates.py`生成8候选及3随机对照；`expert_scoring.py`定义专家与动作坐标；`server.py`只提供冻结推理而无训练RPC；`gate0.py`执行校准与环境对照；`analyze.py`产生门禁及报告；`test_gate0.py`做单元测试。

Gate1/2/3必须依次解锁，当前没有宣称实现或执行下游训练入口。Gate0为监督候选数据，不能传给on-policy Group或Critic。no-Focus仅是零计算量参考，其他策略按相同真实历史和追加一次Focus比较；完整序列和整集SR/SPL验收不由此替代。

## 复现

### 全train扩采至256 anchors

`expand_anchors.py`按 `EXPANSION_PROTOCOL.md` 将搜索范围从旧54场景扩大至全train的61场景，保留64个非校准anchor并新增192个。实际完成256个唯一非校准anchor（118个可信标签），独立核验通过；Gate1/2/3未运行。范围、成本以及微步j与split不均衡限制见 `artifacts/fulltrain256_v1/REPORT.md`，逐文件交付清单见同目录 `DELIVERY_FILES.txt`。新增7个原train-development场景不能再标为训练未见盲测。

复采必须指定不存在的新输出目录；以下验证命令只复核已保存的证据，不重新执行全部环境分支：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task233/verify_expansion.py \
  --output Experiment/SCOPE/Stage2/Task233/artifacts/fulltrain256_v1
```

### Gate1空间热启动

已执行有限预算的STOP隔离空间监督热启动，结果为`Gate1=INCONCLUSIVE`：可信heldout加权NLL和候选top1均退化，按协议锁定Gate2/3。报告和独立复核见`artifacts/gate1_v1/REPORT.md`与`INDEPENDENT_VERIFICATION.json`；本次没有执行真实导航Gate1验收、Group或PPO。

### 原Gate0入口

从项目根目录执行；模型Python3.10与Habitat Python3.9分离。需要本机冻结SimpleNav/Qwen、Task21 Adapter、Task23 update14、R2R train及MP3D资产。新输出目录必须不存在；程序拒绝覆盖旧run。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task233/gate0.py \
  --output Experiment/SCOPE/Stage2/Task233/artifacts/reproduce_gate0 \
  --port 10233 \
  --model-gpu 0 \
  --habitat-gpu 1 \
  --wall-hours 3
```

```bash
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest \
  Experiment.SCOPE.Stage2.Task233.test_gate0
```

对完成run重算审计（只改指定新run中的派生分析，不改原始JSONL）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task233/analyze.py \
  --output Experiment/SCOPE/Stage2/Task233/artifacts/reproduce_gate0
```

核心产物：DATA_MANIFEST/RUN_CONFIG、SOURCE_INTEGRITY_BEFORE/AFTER、FROZEN_THRESHOLDS、candidate_results.jsonl、candidate_quality/candidate_pairs/paired_effects/strata.csv、restore_exactness、teacher_signal_audit、PROGRESS_GATE、REPORT、runtime状态与推理telemetry。基础设施失败保留INCOMPLETE及日志，不删除失败尝试。
