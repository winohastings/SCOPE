# Task2.2 Spatial Query Actor 与主动注视闭环

本目录实现完整 STOP → camera → continuous latent position Actor，默认最多四次 Focus。
只使用 `../Task21/` 的空间融合，不追加 Focus 图片块或 token。最终报告为 `REPORT.md`，
逐文件交付与哈希为 `FILES_MANIFEST.tsv`。

## 主要接口

从项目根目录使用 SimpleNav 的 Python 环境：

```python
from Experiment.SCOPE.Stage2.Task22.actor_runtime import ActorPolicy

policy = ActorPolicy(
    output="outputs/task22_local",
    mode="training-sampling",
    actor_checkpoint="Experiment/SCOPE/Stage2/Task22/artifacts/final_runtime/actor_initial.pt",
)
session = policy.new_session(episode_id="1", statistics_key="vln_train_train", seed=222)
response = policy.predict_session(session, frames=frames, instruction=instruction)
```

`frames` 沿用旧 Stage2 frame，还必须携带 `stage2_scene` 与 `stage2_purpose`。
`response` 的动作仍由冻结 SimpleNav 输出。执行实际导航后，必须提供原 Task4
`ExecutionFeedback` 格式调用 `complete_step`，才允许下次预测。用原 FeedbackModel/
FeedbackBackend 可自动接好反馈、实际位移、碰撞测量及截断 bootstrap。

`mode` 必须显式区分：

- `deterministic`：STOP 阈值 0.5，camera argmax，latent 使用均值。轨迹标为不可用于 PPO。
- `training-sampling`：按联合策略采样，使用独立 per-session CUDA Generator；只表示
  采样方式，**不进行训练**。新 episode 从指定 seed 初始化，所有 latent 与 RNG 状态保存。

Actor 的推理模式使用 `actor_mode`，与旧 `ScopePolicy.mode=glance_only` 分离。

## 增量视觉接口

`Task21/runtime.py:FusionPolicy.predict_session(..., controller=callback)` 在原生视觉
forward 已获得 Glance 后调用控制器，再把最后的融合结果交给原 Qwen scatter。

控制器接收 `IncrementalPerception`：

- `glance`：原始四路 Glance，不随 Focus 改变。
- `current`：当前融合网格，恒为 `(4,16,2560)`。
- `acquire(Focus(...))`：只编码本次 ROI、调用冻结 Adapter、更新融合网格、历史坐标及
  精确像素并集 coverage，然后允许 Actor 重新决策。
- `coverage`：每相机 4×4 单元的已观测面积比例；重复裁剪不重复累加覆盖面积。
- `history`：最多四行 `[camera_onehot(4), normalized_center_x, normalized_center_y, valid]`。
- `finish()`：关闭本次感知，之后拒绝继续读取 ROI。

Glance 沿用原本与历史图像一起编码的 batch，不额外重编码当前图来喂导航。N=0 返回
原 Glance；BATS、历史图片占位、历史池化和缓存输出格式不变。不能保留控制器到下一
导航步使用，也不能从旧 `CurrentFeatureBank` 的离散候选 ID 推断连续 ROI。

## 可执行复现

先确认 GPU0、GPU1 和端口 10122 可用。所有新输出目录必须尚不存在。

终端一启动冻结导航与 Actor 服务：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task22/server.py \
  --output Experiment/SCOPE/Stage2/Task22/artifacts/recheck_runtime \
  --port 10122 \
  --mode training-sampling \
  --actor-checkpoint Experiment/SCOPE/Stage2/Task22/artifacts/final_runtime/actor_initial.pt
```

终端二运行真实 Habitat 的两集受限闭环：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task22/run_habitat.py \
  --output Experiment/SCOPE/Stage2/Task22/artifacts/recheck_sampling \
  --port 10122 \
  --mode training-sampling \
  --count 2 \
  --max-steps 8
```

前一运行完成后，在同一服务上执行确定性模式：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/Task22/run_habitat.py \
  --output Experiment/SCOPE/Stage2/Task22/artifacts/recheck_deterministic \
  --port 10122 \
  --mode deterministic \
  --count 2 \
  --max-steps 8
```

关闭自己的模型服务后，单独重载权重进行概率/冻结/同状态干预验证：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/Task22/validate.py \
  --runtime Experiment/SCOPE/Stage2/Task22/artifacts/recheck_runtime \
  --output Experiment/SCOPE/Stage2/Task22/artifacts/recheck_validation
```

运行回归测试（无 CUDA 时只跳过 CUDA 重放项）：

```bash
OMP_NUM_THREADS=2 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope*.py'
```

## Task2.3/GA-PPO 交接

完整协议在 `TRACE_SCHEMA.md`。直接复用 `SpatialQueryActor`、`ActorState`、
`LatentAction`、`replay_decisions`、`joint_log_probability`、`load_episode_traces`。

`navigation_transition` 把已完成且符合采样条件的 trace 转成原 Task4 `Transition`，
继续使用原 `gae_targets` 与 `FixedTargets`；`next_state` 已由 episode loader 按时序
连接，末端截断必须有真实后继观测的 bootstrap 特征。

不能直接调用旧 `PPOTrainer.update`：它仍假设 17 类离散动作与旧 GRU，也调用旧
`joint_log_probability`。Task2.3 应使用这里的分层概率 evaluator 接入其优化循环，
而不是伪造离散 mask。此任务没有提前实现或运行正式 PPO/Critic 更新。

不支持从 trace 自动恢复 Habitat 中途状态，不声称只靠特征就能生成未观察的 Focus。
同状态环境分支须沿用旧 `real_rollout.diagnose` 的 episode reset + 已执行动作前缀重放，
重新校验图像 hash；本次验证只进行已捕获真实初始观测的感知干预，不把它冒充环境反事实回报。
