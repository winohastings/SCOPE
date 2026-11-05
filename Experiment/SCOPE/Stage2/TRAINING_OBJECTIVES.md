# 训练接口、奖励和 PPO/GAE

## 三种训练场景

| 场景 | 相同 Actor 核心的配置 | 预算/反馈 |
|---|---|---|
| A | navigation_state=False、action_feedback=False、perception_stop=False | 固定两Focus，无跨导航历史；单步内顺序Focus记忆保留 |
| B | navigation_state=True、action_feedback=True、perception_stop=False | 固定两Focus，使用真实执行反馈 |
| C | navigation_state=True、action_feedback=True、perception_stop=True | 0–2 Focus，自适应感知停止，成本感知奖励 |

所有场景默认use_language/use_focus_memory=True，其余开关可独立构造；没有另外的
一次性 Selector。Qwen3.5、视觉encoder、TVI与DiT-B默认全部冻结，Actor/Critic参数独立。

## 环境奖励

只在真实 R2R train 场景动作后读取 teacher：

`r = 0.2*clip((d_before-d_after)/3,-1,1) - 0.01 - 0.02*contact`

真实导航结束另加 `2*success + 0.5*SPL - 0.5*(not success)`。pilot/环境时间截断不加
该终止项，使用真实后继 Critic bootstrap。距离为 Habitat 原生 geodesic；缺失或非有限
teacher显式拒绝。contact来自真实连续动作中step_filter请求/返回位置的差异，表示阻挡
或滑移，是可测的控制接触代理，不声称物理碰撞力或完整碰撞次数。

C再减 `0.01*(64*k/192) + 0.001*perception_ms/1000`；A/B成本系数为零。
一整个导航chunk只记一次reward，不给两次Focus重复发同一环境奖励。动作/STOP只改变
感知，不瞬移机器人，不使用未来未选择高分辨率特征。真实时延进入奖励会有系统噪声；
单元checkpoint恢复验证使用已固定rollout，真实重采样不能因此声称逐位相同的时延奖励。

## PPO

一个导航transition存储完整顺序感知轨迹，旧联合logprob为各实际决策logprob之和，
包括显式感知STOP；达到两Focus预算后自动完成，不加入额外假的随机STOP。

`ratio = exp(sum(logpi_new) - sum(logpi_old))`

Actor loss为负 clipped surrogate（epsilon0.2）减0.01联合条件熵；Critic loss为0.5 MSE。
Actor和Critic各自AdamW lr3e-4，梯度norm裁剪1。GAE gamma0.99、lambda0.95，真正终止
不bootstrap，截断bootstrap但阻断GAE trace；episode边界永不传播优势。

旧logprob/old value、returns与advantages在rollout完成后固定并detach，同一update所有
epoch重复使用，绝不边更新critic边重算target。保存每次合法动作mask，PPO重放重新生成
mask并逐位核对；不允许重复ROI或禁用STOP时采STOP。

小规模实测每rollout优化2epoch、minibatch<=16；这是工程验证配置，不宣称已选择完整
训练超参。State GRU采用长度1截断反向传播，保留真实执行反馈的可微单步GRU转换，历史
hidden detach；没有长序列burn-in/BPTT，因此大规模稳定性仍需进一步研究。

## 恢复与权限

episode边界checkpoint原子写入，保存Actor/Critic、各自optimizer、update/transition计数、
Python/NumPy/Torch/CUDA RNG、动作generator、下一训练episode cursor、regime/seed、
主干/数据划分hash。provenance或配置不符直接拒绝，dev不允许update，heldout purpose拒绝。
restore在无活动session时执行；支持训练从保存cursor继续，不重复假装恢复的episode。

本轮只允许两训练episode以内、每集<=16策略步的pilot；没有启动199680transition/seed或
全规模PPO，也没有val_unseen评估。成本外推只是参考，不是执行授权。
