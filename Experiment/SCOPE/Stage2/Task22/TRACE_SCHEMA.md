# AGM 感知轨迹协议：agm-task22-v1

## 动作与概率

Actor 输入仅有 `G,L,M,B`，不包括距离 oracle、奖励、teacher 或环境终止标签。
G/M 为 `(4,16,2560)`，L 为 2560 维原 Task4 冻结指令 embedding 均值；B 为
`coverage:(4,16,1)`、`history:(4,7)` 和 `count:0..4`。

STOP 动作为 `stop=True,camera=None,latent=None`，概率仅为 `log(p_stop)`。
继续观察为 `stop=False,camera:int,latent:float32[2]`：

`logpi = log(1-p_stop) + log p(camera) + sum(log Normal(u;mu_camera,std_camera))`。

`u` 是实际采样的潜变量，不是反推自舍入后的像素。像素中心使用 float64 计算
`center=96+320*sigmoid(u)`，然后将 `center/512` 传入 Task2.1 `Focus`。
直到 Task2.1 的 `Focus.roi` 才按 `floor(center-96+0.5)` 舍入，以同一整数 ROI 裁剪、
映射与计算 coverage。数学上中心在 (96,416)；浮点 sigmoid 饱和可得到端点，仍保持
192² 合法裁剪，不裁断或重新估计 latent。

本协议定义动作测度为 latent-u 空间，因此 joint log probability **不添加像素位置的
Jacobian**。PPO 用同一个 u 重算新旧策略密度；不能将此值称为整数 ROI 的概率质量，
也不能将它与 x/y 空间的 density 混合。position entropy 记录二维 Gaussian 的潜空间
微分熵；`joint_latent = H(stop)+(1-p_stop)*(H(camera)+E_camera H(u|camera))`。

第四个 Focus 后直接结束，termination=`budget`；不再调用 Actor、不创建虚构 STOP
decision，不额外增加任何 STOP log probability。0..3 次后结束必须是实际策略 STOP。

## 持久化格式

每个导航 transition 一个 `.pt`，只保存普通 dict/list、标量和 Tensor，可使用
`torch.load(..., weights_only=True)`。所有 observation 和 old probability 已 detach，
CPU 原精度保存，不经过 JSON 浮点往返。文件写临时路径后原子替换。

| 字段 | 含义 |
|---|---|
| `schema,transition_id` | 协议版本；唯一 session UUID + episode ID + navigation index |
| `episode_id,scene_id,provenance` | 数据身份、冻结导航/Adapter/划分 SHA256 |
| `actor_checkpoint_sha256` | 生成旧概率的 Actor 版本 |
| `mode,ppo_eligible` | deterministic 不允许进入 PPO；training-sampling 仍需实际完成反馈 |
| `decisions[]` | 每次决策完整 G/L/M/B、STOP/camera/u、float64 center、整数 ROI、old joint logprob、各熵、旧分布参数、前后 Focus 次数、时间和 transition ID |
| `old_sequence_log_probability` | 所有实际感知 decisions 的 logprob 和；不含 budget 伪动作 |
| `final_state` | 最终 G/L/M/B；可直接检查融合和覆盖 |
| `rng_before,rng_after` | 本导航步独立 Actor Generator 的状态 |
| `navigation_actions,navigation_stop` | 冻结 SimpleNav 输出；感知 STOP 不直接修改导航 STOP |
| `current_visual_tokens,context_grids` | 当前固定 64，历史另计 |
| `feedback` | 原 Task4 ExecutionFeedback + 实际 body displacement / prediction error |
| `teacher_for_reward_only` | 实际导航反馈中的距离/碰撞/终止信号；仅供后续 reward，不进入 ActorState |
| `terminated,truncated` | 导航结束与时间/外部步数截断分离 |
| `bootstrap_state` | 仅在非终止末端截断时保存真实后继 Glance 状态 |

未完成 feedback 的 trace 不能用于训练；基础设施失败不补造 reward。运行时一次最多
一个 pending action；完成反馈前拒绝继续预测。`finish(update=True)` 明确拒绝。

## Replay 与长度边界

- `replay_decisions` 校验形状、有限性、冻结观察、计数、STOP 顺序、latent/ROI 一致性、
  状态身份及序列 logprob 和。跨 CPU/CUDA 的 float64 sigmoid 坐标比较容差为 1e-10
  像素；整数 ROI 必须完全相同。不会修改 latent 或旧概率来强行通过重放。
- `Actor.evaluate` 可在梯度模式重算；无 dropout。空间层显式使用统一 attention 路径，
  避免 eval/inference fused Transformer 与训练路径差异改变策略概率。
- `load_episode_traces` 使用 `episode_boundary_*.pt` 中的有序路径，不按文件名 UUID
  猜测 episode 顺序；检查所有 transition 连续、同 checkpoint、无中间终止。
- loader 增加内存中的 `next_state` 引用：下一导航步第一决策状态，或末端 bootstrap；
  terminal 时为 None。不改写历史 trace 文件。
- bootstrap 为真实后继图像四路直接编码；其无历史 batch 调度与原导航混合历史 batch
  不同，这个差异在字段中明示，不宣称两种调度的 BF16 特征严格一致。
- 每次 perception 最多四个 decision；每 episode 最多 200 个导航 transition，达到
  上限明确报错，由外部 runner 做显式截断并调用 finish，禁止静默丢弃旧状态。
- 磁盘逐步保存，内存只持有当前 pending trace，session 只累计路径；summary 为轻量
  字段。无无限增长的 GPU 历史特征缓存，但长运行仍需外部配置磁盘配额与分片。

## checkpoint 与恢复边界

`actor_initial.pt` 保存初始化/加载后的权重、结构配置、seed、冻结依赖哈希和零 optimizer
update 计数。仅提供 Actor 权重重载；本任务不提供优化器续训接口。

`episode_boundary_*.pt` 保存完成 episode cursor、Actor RNG 状态、provenance、trace
路径和冻结审计。它不是 Habitat 中途 snapshot。后续恢复应在 episode 边界开始新
episode；不得把同一个场景 reset 当成此前任意中间状态。

对同状态 Group 分支，只能重用已保存感知状态的概率计算；若需要新的 ROI 编码或真实
导航后继奖励，必须显式获取原始图像/重放环境动作前缀。旧值、reward、next_value 由
Task2.3 的 Critic/奖励管线提供，`navigation_transition` 不代填任何 teacher。
