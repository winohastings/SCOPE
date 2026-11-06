# Task2.1：固定导航 token 的空间视觉融合

验收报告：`REPORT.md`。最终实测产物为 `artifacts/release/`；`artifacts/calibration/`
保留首轮校准及检查，不作为最终计时结果。配置为 `config.json`，完整文件与哈希见
`FILES_MANIFEST.tsv`。本任务未启动 Actor/PPO 或 val_unseen 评估。

## 接口

从项目根目录、使用 SimpleNav Python 环境调用：

```python
from Experiment.SCOPE.Stage2.Task21.runtime import Focus, FusionPolicy

policy = FusionPolicy(
    output="outputs/task21_inference",
    checkpoint="Experiment/SCOPE/Stage2/Task21/artifacts/release/adapter.pt",
)
session = policy.new_session(episode_id="1", statistics_key="vln_train_train", seed=211)
```

`frames` 沿用现有 SimpleNav RPC frame 结构，包含 `images`（四路 uint8 RGB 512×512）、
`body_pose`、可选 `timestamp_s`。`instruction` 是任务原始指令。

- 无 Focus：`policy.predict_session(session, frames=frames, instruction=instruction)`。
- 固定中心：`policy.predict_center(session, frames=frames, instruction=instruction, camera=0)`。
- 指定相机与位置：`policy.predict_session(session, frames=frames, instruction=instruction, focus=[Focus(2, 0.37, 0.63)])`。
- 顺序累积：传入最多四个 `Focus`；允许重复。`Focus.camera` 为 0/1/2/3，分别为 front/left/right/rear。
- 位置为归一化的原图裁剪中心 `(x,y)`，均属于 `[0,1]`。不是旧 Task4 的 16 个离散候选编号。
- 新 episode 必须建立新 session。当前特征只存在于本次推理，不允许跨 episode 复用。
- `SpatialFusion(glance, updates)` 可直接用于后续 Actor 的逐步视觉状态更新；无更新返回原张量对象。
- 未加载已完成校准的 checkpoint 时，运行时拒绝带 Focus 的导航调用。

新接口保持现有四个 Glance 图片占位、相机标记、TVI、Qwen scatter 和动作解码方式；
不得再向 `scope_blocks` 追加 Focus。旧 `Stage2/runtime.py` 和旧训练入口保留原行为，
后续 Task2.2 应显式复用此目录的新接口，不能把旧追加 token 路径当作空间融合。

## 复现

在项目根目录执行；输出目录必须尚不存在，避免覆盖实验产物。GPU0 用于模型，GPU1
用于 Habitat；不要在对应设备已有其他任务时直接运行。

1. 从已经冻结的 train 划分采集 8 个真实 R2R/Habitat 初始状态：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task21/capture.py \
  --output Experiment/SCOPE/Stage2/Task21/artifacts/recheck_capture
```

2. 校准单标量门，保存并重载 checkpoint，验证冻结、无 Focus 一致性和不同 Focus 数量的成本：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task21/calibrate.py \
  --capture Experiment/SCOPE/Stage2/Task21/artifacts/recheck_capture \
  --output Experiment/SCOPE/Stage2/Task21/artifacts/recheck_calibration \
  --config Experiment/SCOPE/Stage2/Task21/config.json
```

3. 运行相关回归单测（不需要 GPU）：

```bash
OMP_NUM_THREADS=2 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope*.py'
```

## checkpoint 契约

`adapter.pt` 独立于导航 checkpoint，包含 geometry version、单参数 adapter state、冻结
导航 checkpoint SHA256、划分 SHA256、采集 manifest SHA256、配置、校准/诊断 episode ID。
加载时检查版本、导航权重、划分、完成标志和参数有限性，再强制冻结。

不覆盖 Task4 的 Actor/Critic checkpoint，不改变 episode-boundary cursor/RNG 规则。
本模块不提供 Habitat 中途恢复；未来 RL 保存 provenance 时应记录 adapter checkpoint
SHA256。当前校准脚本是确定性短跑入口，不提供优化器中途续训。
