Task 3 — 完整导航级 PPO/GAE Actor–Critic 训练（PPO-only 正式基线）
目标
完成真实 R2R/Habitat 导航反馈的 on-policy 训练闭环，为后续 GA-PPO 提供通用 rollout、reward、critic、update、checkpoint 及可比较的 PPO-only 模型。不是 toy RL 或降级验证。
Codex 实施范围
1. 新增独立 Critic：仅使用每个导航步 Glance 后、Focus 前 s_k^0 的视觉与指令信息，输出 V_phi(s_k^0)；架构参照文档 128 hidden、一层空间编码、attention pooling、MLP；Actor/Critic 各自优化器。
2. 联通导航级 reward：`r_k=0.2clip((d_k-d_(k+1))/3,-1,1)-0.01-0.01N_focus+r_terminal`；成功 +2，失败真终止 -0.5，时间截断 0。用导航 geodesic 距离，但 仅作为训练回报。校验动作块长度、实际 terminal 与 truncation 边界。
3. 主轨迹从 on-policy Actor 采样，SimpleNav 执行动作块，缓存每步的决策序列、旧策略概率、旧 Critic 估计、导航奖励、后继状态。完整 episode 后做 GAE(0.99,0.95) 与 lambda-return；PPO ratio 以每个真实感知决策的联合 logprob 计算；每条感知序列内部求和，再按导航状态平均，避免 Focus 长度或复制样本数无意加权。
4. 训练与采样使用冻结旧策略快照；每轮积累不少于约 64 个真实导航 transitions，按既有 episode 边界收尾；更新约两轮后丢弃旧 on-policy buffer。保存 actor/critic/optimizer/RNG/config、checkpoint 和可复现轨迹摘要。
5. 跑完整 4096 个主轨迹 transition 起始预算，并保留继续到 10k–20k 的正式训练配置。记录原始训练动态，而不是只挑选最好 checkpoint。
交付物及验收
- mode=PPO-only 可重现的官方训练入口、Critic、replay/rollout buffer、GAE/PPO update、完整 checkpoint、训练曲线及 val_seen 周期评估。
- 在冻结旧策略与新策略相同的开始时刻，重要性采样 ratio≈1；不同 episode 不混算 GAE；terminal 与 truncation 处理正确；loss/entropy/gradient/critic prediction 有限。
- 训练能完成定义的主轨迹预算、产生有效导航反馈、保持 0–4 Focus 可选和 64 当前视觉 token 不变量；报告 SR、SPL、Focus 分布与实际计算成本。
- 成功训练不等于验证有效：如果导航表现没有改善，要先定位 Adapter、奖励归因、早停塌缩与导航动作对 Focus 的敏感性，不能通过增加实验数量掩盖失败。