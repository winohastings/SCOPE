# Task233 Gate0 预注册协议 v1

日期：2026-11-07；执行授权：用户本轮明确要求按Gate0→Gate3执行。
根规范指定的agents/exec-agent.md仍缺失，本轮以用户指定Task233任务文件作为执行细则，记录此偏差而不再重复以缺失文件为由停止。没有获得改写旧日志或解冻模型的授权。

## 预算与划分

R2R train原manifest中seed233随机排列，按scene轮询选36个不重复trajectory；前4集仅校准（8 anchors），接着16集为未来teacher训练保留，最后16集为teacher held-out（不得参与warm-up梯度）。每集初态与冻结SimpleNav no-Focus一步自主到达状态各1个anchor，总判门目标64；自主状态不是GT同时间索引。初态是专家轨迹起点，当前协议不声称覆盖专家轨迹全部中间状态。任意提前终止导致数量不足则INCONCLUSIVE/BLOCKED，不看结果替换episode。

每episode的j=ordinal%4，对同一anchor先真实顺序执行相机0..j−1中央Focus；所有边际候选从同一融合状态出发。这是受控真实历史，不是静态feature伪造，也不声称这些历史来自Actor。每个候选只追加一次Focus；完整learned四Focus导航策略评估留给Gate1，不用此边际探针冒充。

每anchor固定8候选：四相机中央latent=0 + 各相机一个seed固定的分散连续latent。候选全部真实执行chunk，允许报告oracle-best-of-8上界，但不可用作部署策略。额外执行固定微步预算learned、3个随机相机/合法中心候选，以及相同已融合状态立即感知STOP；fixed参考为candidate_(j%4)。j>0另跑完全Glance-only以生成主导航前缀。校准集另重复STOP与candidate_0。最多1008次预测、最多3小时，不启动任何optimizer，不构建Critic训练buffer。

## 当前位姿专家与动作比较

只在Habitat进程用R2R train goal和当前pose查询可达最短geodesic路径，选最近可达goal；这是一种目标条件导航专家，不是释放模型的逐帧GT动作标注，也不保证遵循指令中的每个中间地标。记录该局限并以真实环境结果校验其代理有效性。distance<3m时专家导航STOP，否则沿路径按0.25m间距采8个anchor-relative body waypoints（总2m，8行与模型chunk执行语义一致）。模型时间尺度没有额外速度GT，固定步长是一项显式代理假设，不能因代理失败认定Focus本身无效。

Habitat坐标转换采用现有backend约定(world_x,world_y,world_z)=(-habitat_z,habitat_x,-habitat_y)，随后用当前anchor yaw转换forward/right/down及相对yaw。每条专家动作经实际backend.project_action_to_world往返核验，最大误差≤1e-5。预测S_GT=−[mean squared xyz Euclidean error + 0.1 mean wrapped yaw error² + 导航STOP不一致(0/1)]。无效路径或坐标审计失败立即停止并保留原因，不能生成硬伪标签。

真实环境评价记录完整8个waypoints的实际执行数、动作stop、terminal/truncation与geodesic progress；模型motion-stop时执行真实Habitat STOP。只报告chunk progress/terminal事件，不冒充整集SR/SPL。GT/距离/路径只在环境侧评分，不进入Actor、Qwen或Critic输入。

## 复原与冻结

每分支独立reset并重放相同no-Focus导航前缀；核验pose、全部图像历史、instruction、metrics、环境步数/terminal/STOP、模型实际context blocks和真实递增ActorState的hash。Glance-only的感知状态与j>0边际状态本就不同，只比环境/context，不能合并状态标签。任何匹配失败停止整次门禁；不得仅恢复pose。

只读Task23 update14旧Actor作为learned探针。模型/Adapter沿用可信原checkpoint并核对provenance；冻结Qwen/SimpleNav/vision/Adapter/Actor，保持四相机、192²→256² ROI与64当前视觉token。保存原始latent，不由整数ROI反解。server没有训练RPC、optimizer或Critic。所有样本data_kind=teacher_candidate、ppo_eligible=false；即便候选来自Actor也不是on-policy Group。

## 校准后冻结的机械判据

仅前8个独立校准anchor重复分支估计执行噪声，随后在判门数据前写FROZEN_THRESHOLDS.json。预定公式：min_progress=max(0.02米,5×最大重复progress误差)；min_GT_margin=max(1e-4,5×最大重复S_GT误差)；可信比例至少25%。可信标签须最佳/次优GT margin过阈值，且GT选中候选相对微步STOP的实际progress增益≥min_progress。所有判门状态都参与收益统计，不能只选可信样本报告效果。

held-out上，GT选中候选分别对fixed、3seed平均random、微步STOP，按状态配对→episode均值→10000次episode-cluster bootstrap(seed233)，报告95%CI并用三重比较Bonferroni的98.333%CI判门：三项下界均>0且均值≥min_progress；可信比例≥25%、64评价anchor/32held-out、所有复原/冻结通过才PASS。oracle-best三项也报告，但不能替代GT选择收益；rank相关性、场景/来源/j分层均报告，GT排序只改变选择不能改变评估样本。GT或收益缺乏区分、区间跨零、可信比例不足判INCONCLUSIVE而非证明无效。没有解锁Gate1不得创建训练checkpoint。

Gate1通过后才另行冻结warm-up/Group/PPO运行配置。严格预算的完整learned/fixed/random/no-Focus SR/SPL及held-out Critic验收仍属后续门禁，未运行就写NOT_RUN，不能拿历史结果补齐。
