# 导航状态轨迹与信息边界

## 提交时序

每次 predict 创建唯一 pending 导航动作；pending 未完成时禁止下一次 predict。
`feedback` 在真实 Habitat 执行后调用，校验预测 action chunk 与起始 body pose，随后
更新 State GRU。导航运动 STOP 不执行动作，但必须先由客户端确认终止，再提交显式
no-action terminal feedback。诊断分支只预测/执行受控 chunk 后销毁 session，不训练。

episode reset 创建全零 Actor/Critic 独立状态；finish 清空跨步、Focus 和记忆。显式
finish 之后才能在 episode 边界保存 checkpoint。中途基础设施异常不伪造终止奖励。

## 每步持久化字段

`runtime/trajectories.jsonl` 每行包含：

| 字段 | 语义 |
|---|---|
| episode_id、step、purpose | 执行归属及 train/dev 权限 |
| bank、encoder_calls、context_grids | 当前像素/位姿 key、编码来源/形状/grid、真实调用 |
| glance_summary | 四 Glance token 的2560维均值 |
| focus_summaries | 按获取顺序分别记录每块64 token的2560维均值 |
| selected_roi_sequence、trace | 实际选中 ROI 序列及显式感知 STOP |
| predicted_waypoints | 原预测八 waypoint，anchor-relative body xyz/yaw |
| before、after、executed_poses | 原预测时真实 body pose、执行后真实 body pose及逐 waypoint 实际位姿 |
| actual_body_displacement | 起始body坐标系中的实际 xyz/yaw 位移 |
| prediction_execution_error | 最后预测 waypoint 与实际末位移差，yaw wrap到[-pi,pi) |
| current_tokens、perception_ms、model_ms | 当前64+64k token、真实含特征采集墙钟、完整预测墙钟 |
| navigation_done、termination、action_executed | 导航结束、原因和是否确实执行，不与感知 STOP 混用 |
| actor_state_norm_after_feedback、critic_state_norm_after_feedback | 真实反馈后隐状态更新的审计值 |
| reward、reward_teacher | 仅训练日志：奖励及其 teacher 来源；不进入 Actor/Critic 观测 |

GRU 输入反馈向量18维：实际位移4、预测末waypoint4、预测执行误差4、token归一化1、
感知秒数1、模型秒数1、导航done1、action_executed1、实际waypoint数/8共1。
输入还含本步 Glance+所选 Focus 联合摘要。相机/ROI顺序由视觉块及 Focus GRU 保留。

## 坐标

使用外部 worker 已验证的 world FRD 和 body pose 协议。标准 observation `pose.yaw`
与模型 `body_pose.yaw` 相差pi/2；实际执行 waypoint poses 在反馈前同样转换。
world位移由起始body yaw的逆旋转得到 forward/right/down，不直接把world坐标差当body。
最终误差不使用GT轨迹、目标坐标、geodesic路径或导航成功标签。

## 无泄漏合同

`PolicyObservation` 仅有 Glance、语言、已获取 Focus/ID、历史隐状态、视觉摘要、实际反馈。
不存在 reward、goal、distance、success、SPL、nDTW 或碰撞 teacher 字段。GT 只在客户端
训练奖励路径读取，server 完成动作后先验校验 teacher，再构造独立 rollout。
dev 永不构造训练 Transition；server拒绝dev PPO更新，数据守卫拒绝跨scene/episode划分。
审计日志保存GT并不等于模型观察GT；两条路径必须始终区分。

## 截断和恢复

真正导航终止 bootstrap=0。人为 pilot 步数限制或 Habitat 原生时间上限是截断，须从真实
后继观测计算 Critic bootstrap；截断不跨episode继续GAE，也不注入伪成功/失败终止奖励。
当前仅支持episode边界精确恢复：checkpoint含独立Actor/Critic、优化器、RNG、划分/主干
hash、训练episode cursor和专用动作采样generator。原 Habitat 环境无中途快照接口，因此
明确拒绝 mid-episode 保存，不宣称能任意时刻精确续跑。
