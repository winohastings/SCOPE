# Task2.3 PPO数据与恢复契约

继承Task2.2 `TRACE_SCHEMA.md`，不改变latent action、ROI、ActorState或BATS历史契约。
每个trace仍保存全部感知决策状态与旧联合logprob，真实8×4导航动作和执行反馈。

## 导航transition扩展

| 字段 | 定义 |
| --- | --- |
| `purpose` | `train`或`val_seen`；只有train允许进入PPO |
| `policy_version` | 采样时冻结Actor/Critic的update编号 |
| `old_value` | 本步第一次感知决策、Focus前状态的旧Critic估值 |
| `navigation_reward` | 完整导航反馈计算的固定奖励；episode结束后写回 |
| `old_next_value` | train transition后继状态旧Critic值；真terminal为0 |
| `next_state` | 同episode下一步Focus前状态；truncation则是真实后继Glance编码 |
| `terminated` / `truncated` | 真导航终止/时间或预算截断；互斥，不表示感知STOP |

`transition_id`带唯一session前缀，重复采样同数据ID也不跨episode计算GAE。
`teacher_for_reward_only`含geodesic距离，仅供奖励使用，不传入ActorState。
每轮buffer必须来自同一`policy_version`，由完整episode边界清单组成。
PPO计算逐感知决策ratio，STOP/camera/latent position分层联合概率由Task2.2
`Actor.evaluate`重建。潜变量高斯logprob与实际ROI映射沿用Task2.2，不引入像素概率。

## 更新数据

`runtime/updates/targets_NNNN.pt`中的advantages、returns、old_values按该轮
`update_NNNN.json`的`boundary_paths`顺序，再按各episode trace顺序对应。
advantages在导航transition维度标准化；returns是标准化前GAE加old value。
每个navigation advantage共享给该步全部实际感知决策；强制第四次Focus结束无额外STOP。
JSON保留全部实际执行minibatch指标、梯度范数、参数变化、KL早停标记及冻结审计。

## Checkpoint

`trainer`保存live/old Actor/Critic、两套AdamW状态、更新计数与全部RNG。
外层保存配置来源、数据游标、累计主轨迹/验证transition数、最佳验证记录及
未更新buffer的`buffer_boundaries`。恢复时从原文件重建buffer并校验旧价值。
只允许无活动session时保存/恢复。RPC心跳不使用训练Python RNG；server禁用自动ping，
客户端RPC接收超时及ping等待为1200秒，避免Habitat场景加载导致伪断线。

轨迹单文件可用`weights_only=True`读取；包含NumPy/Python状态的完整checkpoint仅可
对本项目可信产物使用`weights_only=False`。恢复目录必须保留所有被引用的原始trace。
