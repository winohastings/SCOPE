Task2.3.2 — Targeted repair, controlled short continuations
定位：选择经 Task2.3.1 支持的修复方向；避免同时重写 Actor、Adapter、奖励和 PPO。

A. 若 H1 主导：保持初始 Actor/Adapter/导航接口不变，测试 focus_cost=0.0025（相对原 0.01）的小成本分支，并保存 progress/terminal/cost 各分量；同时对原模型作不同 STOP 阈值的纯评估诊断，但不把修改阈值本身冒称策略学习成功。可选探索 cost warm-up（先0或0.0025，随后逐渐增加）需作为独立标注实验，避免组合后无法归因。

B. 若 H2 主导：保持原奖励，试 PPO stability package：rollout>=128（或256）真实 transitions/update、Actor LR=3e-5、Critic LR=1e-4，其余算法结构不动；第一轮先不动 clipping、熵系数和模型架构。把此视为一组工程稳定性配置，不声称已识别是 LR 还是 batch 单独有效。必要时第二轮再做单因素拆分。

C. 若 H3 有直接负面证据：只对 Adapter/视觉融合做定向修复（如提高 train scene 覆盖、相机/ROI 位置均衡、控制 gate 权重），并通过 matched-budget Focus 一步导航效果和 Glance 无退化验证；不要盲目联合训练大型 Adapter，避免改变项目主线。

省算力试验设计：优先从 update14 (1033 transitions) 的真实 Actor/Critic checkpoint fork 新 run，保留数据 manifest、下一个训练 episode 顺序与可比评估清单；配置修改必须使用新的实验名/训练配置/优化器状态规则/哈希，不能伪称“原 Task23 断点续跑”。原训练 1033→2054 的退化是历史参考；先做少量 unchanged-fork 一致性核验，确保后续比较有效。对 A/B 每个分支先采集 512–1024 新 on-policy transitions，不复用旧策略轨迹作更新。每 256 transitions 做固定状态 Actor probe，每 512 transitions 跑 8–12 episode 的相同快速诊断。根据表现选择保留分支，再进入 Task2.3.3。

交付：少量明确配置分支、真实训练日志与 checkpoints、旧/新结果对照、冻结/恢复/ratio 复验、为何采用最终修复的证据。不得只按一次小验证 SR 最大值挑参数。