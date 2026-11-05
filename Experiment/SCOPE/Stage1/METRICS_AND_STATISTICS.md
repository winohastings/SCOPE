# 指标、聚合与统计比较

状态：待审协议，不是已得到的统计结论。配置与数据列表以同目录主协议为准。

## 1. 分析单位与主指标

分析单位是完整manifest中的episode/instruction，不是policy step或waypoint。
R2R的1839集、RxR English guide的3669集，分别对三个配对seed block运行；
同一物理路径的多条指令及同scene样本可能相关，不把全部行当独立试验。

确认性质量指标是nDTW（原始0–1单位）；SR为安全性指标。另报OSR、NE(m)、SPL、
path_length(m)、policy steps、termination reason、完成数、失败数、所有planned episode数。
SR/OSR/SPL在论文展示为百分比时统一乘100，nDTW亦可展示×100但必须标单位；
检验与置信区间先在0–1单位计算。不能将两步smoke中的SR当正式性能。

冻结当前evaluator及GT路径逻辑：
- nDTW使用GT `locations`，预测轨迹只去除连续重复位置，success_distance=3m。
- NE、导航成功与路径长度沿用现有SimpleNav evaluator，保留euclidean设置；
  不从Habitat动作前未STOP的success字段直接代替最终SR。
- 明确区分导航stop、最大步数、数据done与执行失败；SPL短路径定义不在本任务另写一套。

## 2. 聚合表与分母

主性能表每任务每组一行：每seed对**全部锁定episode**的算术均值，然后对三个seed均值取平均；
三个seed原始均值和样本标准差一并发布。因为各seedepisode数相同，等价于全配对行的平均。
绝不先按scene取不加权均值冒充episode均值；scene宏平均只作附录诊断。

固定配对键 `(task, split, manifest_hash, scene_id, episode_id, run_block)`。
G4的block42/43/44分别固定selector训练seed101/202/303；不能按结果重新配对。
三个seed同时包含导航随机性和G4训练随机性，不能宣称仅测到了某一种随机性。

保留三层数据：逐步JSON/JSONL、逐episode表、配对跨组比较表。最低列：
group、selector hash、checkpoint hash、manifest hash、scene、episode、global ordinal、
run seed、session seed、steps、各导航指标、failure类别、cost有效标志、硬件/依赖指纹。
G2额外记录cyclic phase和pair，G3记录随机digest，G4记录两次选择及weight hash。

### 错误处理

现有Task1 summary对failure排除后平均，不能直接充当Stage1最终论文分母。
基础设施故障（服务断开、场景加载失败）用相同seed和完全reset最多重试2次，原失败记录保留。
仍未解决则该任务/block标为不完整，禁止生成正式配对推断表或只对完成交集宣称性能。

可稳定复现、由方法导致的OOM/非法动作属于方法失败：预先记录分类，主SR/OSR/SPL/nDTW
按0纳入planned分母；NE/path_length保留缺失，单列“成功产出轨迹子集均值＋方法失败率”，
不能把缺失值填0后称全样本NE。另给不插补的原始结果及保守失败敏感性分析。
成本按失败前已消耗记录并标censored，不能因提前崩溃而宣称低耗时。
如果故障分类不明确，先阻塞正式表，不由研究者看结果后决定删谁。

## 3. 配对效应量与置信区间

主比较G4−G2、G4−G3的nDTW。每episode先在三个配对block内作差，再跨block平均。
报告绝对差、95%配对置信区间；SR/SPL报告百分点差，同时给原始0–1单位CSV。

候选CI方案：10000次scene-cluster paired bootstrap，seed=20261105。
每次从11个scene有放回抽11个，复制所抽scene的全部episode；三run block另有放回抽3个，
同一抽样索引用于两个比较组。每次对复制后的全部episode/所抽block计算差的算术均值，
按实际复制episode数加权，不能先不加权平均scene均值。取2.5%/97.5%分位数。
保存bootstrap索引、实现版本与CSV。整个scene成块能保留同scene/路径/多指令依赖，
但仅11scene、3seed的区间不稳定，应明确小cluster限制，不将此解释为跨所有环境分布保证。

逐步延迟的CI也不能用step当独立样本；闭环成本先在episode内聚合，再以同样配对cluster重采样。
固定观测成本benchmark另按episode来源cluster汇总，不与独立闭环轨迹比较混在一起。

## 4. 确认性检验与多重比较

预指定R2R nDTW的两项比较构成一个family，双侧alpha=0.05、Holm校正：
排序p后较小者阈值0.025，较大者在前者拒绝后阈值0.05。
候选检验为paired scene sign-flip：11个scene穷举2^11=2048个共同符号，
对同scene全部episode与seed差施加同一个符号，统计量为episode加权总体均值。
exact enumeration p为 `count(abs(T_perm)>=abs(T_obs))/2048`，无需Monte Carlo加1。
它依赖零假设下scene差的符号可交换/对称性，不是随机分组实验的无条件因果保证；
必须同时报告scene-cluster CI和每scene效应，不能只给一个显著星号。

SR非劣性阈值预设−0.02（负2百分点）。同时检查G4−G2、G4−G3、G4−G0三项，
为此安全性family使用Bonferroni：每项one-sided 98.3333%下置信限，
即同一cluster bootstrap分布的1.6667%分位数，须全部大于−0.02。
这是待审批的实用margin，不声称已有充分power；审批时可在不查看heldout结果条件下调整并留版本。

“学习选择有益”须：两项nDTW均正向且Holm通过，并满足上述SR安全性。
若显著性不足，写“未检出优势/区间仍宽”，不等价于证明无效；
若SR恶化超margin、预算不等或有重复编码，不能声称等预算改进。
G0/G1的其他比较、NE/OSR/SPL与scene分层结果为探索性，完整展示，不追选显著项。

RxR是冻结selector的泛化验证，单独展示同样效应量/CI，默认探索性；
不与R2R混合分母，也不能看到R2R后换RxR语言/episode列表。
如后续需要确认性泛化检验，应在运行前另行注册family及其alpha分配。

## 5. 成本效应与论文表格

性能表：N planned/valid/method failures、SR、OSR、NE、SPL、nDTW、path length、steps。
成本表：当前/context/实际encoder token、视觉GFLOPs、模型step延迟各阶段、渲染延迟、
端到端每episode耗时、p50/p90/p95、峰值显存、selector额外成本，训练成本另列。
确认性效率声明须在预设质量守门条件满足后比较G4与G0的episode总模型耗时和GFLOPs，
不能只用192<256或排除selector/历史/失败成本推导“更高效”。

闭环每步成本受到不同轨迹与历史长度影响，另报告同固定观测/相同history重放microbenchmark。
同固定观测不意味着策略选择相同ROI；输入状态相同、各组按自己规则选择，所有实际成本都计算。
两类表不可跨口径拼成单个speedup。预热、计时边界和FLOPs覆盖率见成本协议。
