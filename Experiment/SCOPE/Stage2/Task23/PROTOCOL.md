# Task2.3 PPO-only 冻结执行协议

协议冻结日期：2026-11-06；配置为 config.json。不得在看到验证表现后修改奖励、数据子集
或checkpoint选择规则并仍声称同一训练。任何新配置需单独运行、单独计数。

## 范围及数据

- 唯一新视觉接口为 Task2.1 SpatialFusion/IncrementalPerception；固定当前64 token。
- Actor 直接加载 Task2.2 `final_runtime/actor_initial.pt`，不替换空间结构和分层动作。
- 新增独立128 hidden Critic：视觉/语言独立投影、一层空间Transformer、语言attention
  pooling、128→128→1 value MLP；只使用本导航步 Focus 前的 G 和 L。
- 冻结 SimpleNav、Qwen视觉及指令embedding、动作头、Task2.1 Adapter。
- 训练只用原 `r2r_selector_train_dev.json` 授权的9837条训练指令；seed=233打乱一次，
  用完整显式顺序采样，跨epoch按同一列表循环，episode ordinal继续增长。
- 从官方 val_seen 778条中预先抽取32条scene-balanced子集（32个场景，每场景先1条），
  所有评估复用同一清单、同一episode种子。子集只用于周期验证和选模，不进入PPO。
- 不读取 val_unseen 数据内容，不用它选 checkpoint。训练/验证 ID 必须以 split+ID
  区分，不能仅凭数字ID误判集合重叠。
- 32集是固定验证子集，不等同完整778集 val_seen 结果，更不能代替最终论文全量评测。
  全量评估与多随机种子结论若未执行，须明确标注，不能声称已证明方法有效。

## 奖励与动作

`r=0.2*clip((d_before-d_after)/3,-1,1)-0.01-0.01*N_focus+r_terminal`。
距离来自真实 Habitat geodesic measurement，仅进入reward sidecar，不进入Actor/Critic。
真导航成功终止+2，失败真终止-0.5；运行中、native time limit、外部步数/预算截断的
terminal bonus为0。成功阈值严格 `geodesic_distance<3m`。

导航动作块必须是既有8×4；保留原相对机体系转换、waypoint motion STOP阈值0.03、
四相机512观测、每episode Habitat native最多200步和导航policy最多200步。
native上限通常先触发，不能将其当作失败真终止。Task2.2的感知预算STOP不构成导航终止。

SR 按真实导航终止且geodesic距离<3m计算；SPL使用真实初始geodesic最短距离与实际
累计路径长度。保留benchmark原始结果以便审计，不能把中途进入成功半径的OSR当SR。

## On-policy / GAE / PPO

- 每轮持有独立冻结 old Actor/Critic；采样期间不更新live或old策略。
- 每一导航transition保存完整感知序列、每个真实决策的旧联合logprob、old V、reward、
  实际next state、terminal/truncation。四次Focus后的强制结束不增加STOP概率。
- 完整episode后入buffer，累计≥64个真实导航transition，在episode边界更新。
- gamma=.99、lambda=.95；terminal bootstrap=0，truncation用真实后继Glance的old V，
  但在truncation/不同session episode边界切断GAE递推。重复同一数据episode也不跨run混算。
- advantage和lambda-return计算一次后固定；advantage仅在navigation-state维度标准化。
- 使用 `Task22.Actor.evaluate` 重算逐感知决策的分层联合概率。ratio是每个decision的
  `exp(new_logprob-old_logprob)`，**不是整条序列的概率乘积ratio**。
- 每个decision使用同一navigation advantage；先对序列内clipped损失求和，再对
  navigation state取均值。不能展平成所有decision后直接平均，也不复制长序列样本。
- AdamW：Actor LR=1e-4，Critic LR=2e-4，weight_decay=.01，独立optimizer；两轮epochs，
  16个navigation states/minibatch，clip=.2，global gradient clip=1。
- Critic损失为0.5*MSE(value,固定lambda-return)，不训练ROI/相机/Focus收益预测器。
- 初始独立熵权重 STOP=.001、camera=.001、latent-position=.0001；三者分别监测。
  此为预声明初始设置，不是验证集超参搜索。非负样本KL估计均值超过.05时提前停止
  该轮余下minibatches，明确记录；不偷偷调低学习率。
- 每轮更新后清空buffer，刷新old快照。不同policy_version的轨迹不能混入同一次更新。
- 不加入Group Sampling、RLOO、GA-PPO、专家/teacher辅助loss、碰撞或重叠额外奖励。

## 执行顺序、预算与恢复

1. 单测验证reward、GAE、损失归约、snapshot隔离、optimizer与恢复。
2. 真实gate采集64个独立主轨迹transition；执行真正optimizer.step，验证Actor/Critic及
   STOP/camera/position/value head梯度；从update前完整checkpoint恢复并重复相同更新，
   检查参数、optimizer、RNG、cursor逐位一致。gate不计入4096正式训练预算。
3. 从Task2.2初始权重重新开始正式PPO-only，先评估初始化val_seen子集，然后训练4096
   个主轨迹transition。最后一集可由明确预算截断，必须bootstrap，不补造transition。
4. 每累计约1024个已训练transition，在更新后的episode边界评估相同32集val_seen，
   正式末尾再评估。验证、gate、重复更新核验的成本单独计数。
5. 最佳checkpoint按val_seen SR优先、SPL次之，同分保留更早模型。同时保存所有update
   动态、评估点checkpoint、latest及best，不能只报告最佳曲线片段。
6. 配置保留10000/20000延长预算，但本次只授权4096。独立运行wall-time上限6小时；
   若触及资源限制，在episode边界保存完整buffer和状态，明确INCOMPLETE及真实已执行数。

完整checkpoint包括live/old Actor/Critic、两个optimizer、Python/NumPy/Torch/CUDA
RNG、minibatch RNG、配置、冻结依赖/manifest hash、采集/训练计数、数据cursor、best
记录及未更新buffer的episode边界引用。只在无活动episode时保存/恢复；不承诺Habitat
中途snapshot。每集Actor采样seed由固定base seed+数据ordinal导出，cursor恢复后保持。

正式训练完成也不意味着已优于初始化；必须同时报告初始化、最终和选中checkpoint，
诊断早停塌缩、Focus敏感性、奖励归因与验证噪声。未改善不得通过追加随机实验掩盖。
