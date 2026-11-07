Task2.3.3 v2：导航价值驱动的 AGM 恢复训练
版本：2026-11-07。计划文件，未执行训练。 核心证据来自  Experiment/SCOPE/Stage2/Task232/REPORT.md、`C_REPORT.md`、`C_MECHANISM.md`，以及 Task231/Task23 原始报告。原 Task2.3.3 的“高效重训与早警”保留，但从单纯修 PPO 改为先训练导航相关空间选择，再训练自适应感知预算。正式 GA-PPO 仍归 Task2.4，Task2.3.3 允许限量 Group 机制验证，不冒充已完成完整 GA-PPO。
执行状态（2026-11-07本轮更新）：用户明确重新授权执行Task233，缺失的 `agents/exec-agent.md` 如实记录，本轮按本文件和独立预注册 `Experiment/SCOPE/Stage2/Task233/TEACHER_PROTOCOL.md` 执行，不再以该缺失重复阻断。Gate0已启动真实候选审计；最终状态以Task233新run的 `PROGRESS_GATE.json` 和 `teacher_signal_audit.json` 为准，不能把启动或实现代码视为PASS。Gate1/2/3在上游通过前保持锁定。不改变任何历史日志、模型或Adapter。

1. 起点及归因
- Task232 的 A（低 cost）、B（低 LR/较大 rollout）、C（新增覆盖奖励）在各自 512 个新 on-policy transitions 内，均未改善确定性四次 Focus 的重复 ROI 现象；重复率依旧 100%，四相机总面积覆盖约 3.52%。
- C0 是与 C 同起点、同预算但 beta=0 的实际重采样对照，不是旧轨迹重算。独立集 C/C0 的 SR 分别为37.5%/28.125%，配对差95%CI为[-6.25pp,25pp]；真实首chunk C−C0进展为−0.001930米，95%CI为[-0.025125,0.023430]。两者重复率均100%，不能把单次SR差视为空间导航价值已恢复。
- C 的奖励与 GAE 算法一致性已通过审计；reward 确实进入更新，不能再笼统认定“奖励代码漏算”。
- C 初始 Focus 贡献约 49.49% 的 bonus；即使该相机与位置不帮助导航，也能拿到它。实际 bonus 与 Focus 次数高度相关（r≈0.964）。全部感知决策共享一个导航级 GAE advantage，无法直接监督当前微步哪一个 camera/ROI 好。
- C 的确定性相机主方向概率约 0.70，随 j=0→3 近乎不变；C0 约 0.51。未证明 Actor 的 coverage/history 因果地改善了 camera/position 的条件决策。
- Task231 的 matched one-chunk learned vs no-Focus / fixed / random 导航进展差置信区间大多跨零；Task2.1 teacher feature-MSE 的改善不是导航收益保证。Task232 独立 Critic 的 held-out terminal-only EV 接近 0 或为负。
- 结论：优先回答“在当前冻结 SimpleNav+Adapter 下，是否存在值得选择的 Focus？”；若存在，先给 Actor 清晰的 task-grounded 教师信号，再考虑成本优化。若不存在，则不是扩大 RL 预算能解决的问题。
2. 不变量、实验隔离和对照
冻结 SimpleNav/Qwen、vision encoder、Task2.1 Adapter；192² ROI→256²、四相机、0–4 Focus、固定64个当前视觉 Token、相机与连续latent-u动作协议均不变。GT/专家标签、geodesic、参考轨迹仅用来生成离线教师评分、监督损失或训练环境奖励，不得成为 Actor/Critic 在推理或 on-policy rollout 中的输入。STOP 是感知 STOP，不是导航 waypoint/任务 STOP。

保留 Task23 update_0014 作为历史高分对照，但不强制从该 checkpoint 继续训练：它的 camera prior 已偏向同一方向。构建两个可审计的候选初始化：① Task22 初始 Actor（检查/重设 camera logits 的近似均匀性）；② Task23 update14 trunk 继承，但相机/位置输出层做显式 reset 或 bias calibration。仅用固定状态/小规模专家验证选启动方案，不依赖单次32集 SR。Critic 从既有弱基线重用还是重置应明示，不跨改变奖励目标默用旧 optimizer 动量。

原历史32集、Task232 独立32集均为开发诊断集，不能被重新标记为最终盲测；再预留一个不参与调参的验证锁定集用于后期检验，val_unseen 不参与训练与调参。专家/GT监督、候选教师标签与 learner-induced 再标注只来自 R2R train；其他 split 只验证，不产生训练标签、梯度或训练用状态缓存。train 内监督训练/保留集按 episode/trajectory 分离并报告场景分层，不能将同一 anchor 的不同候选拆到两侧。训练分叉要固定 manifest、场景、随机数、候选集和计算预算；报告 train/teacher/validation/GPU成本分别计数。所有旧 checkpoint 在 GitHub 均可能是 placeholder，应从可信本机恢复并核对 SHA256。
3. Gate 0：先检验“存在可学的 Focus 导航信号”
目的：避免从无导航价值差异的 Focus 候选中硬选“专家标签”，导致模型学到任意 camera bias。
1. 从 R2R train 选跨场景至少64个 anchor；既包含专家轨迹状态，也包含模型自行走到的偏航状态。后者必须在当前位姿重求有效专家导航动作，不能使用 GT 轨迹同一时间索引的动作。
2. 逐状态提供 8 个预定义/采样的合法 Focus 候选（四相机中心+分散空间候选/当前 Actor 候选），以及 Glance-only；候选固定数和抽样来源记清。候选先走冻结 SimpleNav 预测，再在预算可承受范围内用真实 Habitat one-chunk 重放评价。对需要 j>0 的状态，每个候选必须从相同真实递增感知历史出发。
3. 候选评价分层：`S_GT = -L_nav(a_pred, a_expert(current pose))` 是廉价代理，要在导航命令正确坐标/时间尺度比较；`S_env = true geodesic progress` 和真实 stop/终止，是验证用实际结果。必要时可增加短前缀/后续块，但禁止把单个 chunk 的进展宣传为最终 SR。
4. 数据标注 gain_i = S_i - S_no_focus、最佳/次优候选 margin、相机/ROI、同预算固定/随机参考、匹配状态验证、候选真实执行成本。对小 margin/排名不稳的状态不提供强硬 camera/position 伪标签，保留无效/近似并列标识。
5. 重要的上限检查：比较同状态 K 候选的 oracle-best（仅为可达性能上界，不可用作推理策略）与 fixed/random/no-focus。若 best-of-K 对导航行为/进展也缺少可测优势，停止 Actor 模仿预训练，单独诊断 Adapter/视觉接口/导航模型对 Focus 的敏感性，不用监督损失粉饰。
产物 teacher_signal_audit.json、`candidate_pairs.csv`、`candidate_quality.csv`、`restore_exactness.json`。报告配对分布、bootstrap CI、有效标签比例、GT代理与真实进展相关性。将其中与训练采集分离的状态用于监督泛化验证。Gate0 的阈值应按噪声/置信区间在运行前冻结，而不是根据结果事后挑选。

Gate0 执行契约：
- 预注册至少64 anchors的 episode/scene/source/j 分层、8候选生成种子、GT动作损失及权重、环境分支预算、重复执行噪声审计子集、置信区间方法、最小实际收益和最小可信标签比例。先以独立校准子集确定容差并冻结，再评估判门子集；不得用同一结果调阈值再判通过。缺失任一字段记为 BLOCKED，而非默认通过。
- 记录当前位姿专家动作的来源、可达性、导航STOP语义、全局到局部坐标转换及与SimpleNav实际chunk长度一致的比较方法。偏航状态重新查询专家；无效路径、专家动作不可定义或坐标审计失败的样本不得硬标注，必须报告排除数量与原因。
- 每条候选保存原始连续latent-u、相机、合法ROI映射、实际执行ROI、完整真实感知前缀及状态标识。端点不得用无穷latent表示，也不得从裁剪/取整后的ROI逆推监督latent。重复真实前缀用于更新融合记忆、coverage和history，不能只改计数或几何覆盖。
- 对真实chunk评估预先固定候选子集，包含GT代理选中候选、fixed、预设随机种子与不追加Focus基线；不得只执行看起来获益的候选。仅当全部8候选均执行时报告真实环境 best-of-8，否则写明“已执行子集上界”，未执行项不插补环境收益。GT选中候选的独立真实收益才支持教师可用性；oracle-best 有收益而GT排序无预测力时，不得直接进入warm-up。
- 对 j>0，边际基线是在相同已融合状态立即感知STOP，不是抹除已有Focus的Glance-only；j=0才与Glance-only一致。总预算对照与微步边际对照分表报告。
- PASS 须同时有合法可复原样本、超过预注册噪声/实际效应门槛的候选差异、足够可信标签，以及GT教师选择在保留状态上的真实配对收益。置信区间覆盖零或标签比例不足记 INCONCLUSIVE 并停止；不能将未证明差异写成已经证明无效。任何预注册上限耗尽都不得临时扩到4096。
4. Gate 1：Expert-guided Spatial Warm-up（主要创新改造）
只优化 Actor 的 camera + latent position 空间头及必要共享 trunk，Critic 暂不承担感知微步价值；STOP 先固定或以预声明的 N=2 为主要教师生成预算（可少量 N=1/3/4），随后再单独学习 STOP。这样先隔离“看哪里”，避免早期低可信 PPO 把学习推向“不看”。训练限定使用教师候选能真实影响冻结导航行为的状态，不能强制每状态四相机均匀注视。

从 Gate0 的真实候选经验生成软偏好或带置信权重的标签：
\[
\mathcal L_{\rm spatial}=-\mathbb E_{(s,a^*)}\left[w(s)\left(\log\pi_\theta(v^*\mid s)+\log p_\theta(u^*\mid v^*,s)\right)\right]
\]
其中 u* 是候选执行时真实的 latent（非 ROI 整数反解），`w(s)` 由候选导航质量差/标签置信度控制。无显著优势的候选跳过或使用低权重软排序。GT 只用于选/权重，不作为 Actor 输入；不能宣称 top-of-K 是全局最优 ROI。可用同状态胜/负候选 ranking 作为对照，但先实现最小加权 BC 基线。

极重要：逐次 Focus 后必须真实更新 M, coverage, history 并重新执行教师候选选择/监督下一步；不能把四个 Focus 当作独立的同一初始状态标签。显式记录 j=0..3 的条件 camera 分布；用真实有效的 Focus1 历史状态检验选择偏好是否变化。教师可包含“看过这里后，重复相同 ROI 不再改善动作”的负例，但不要以均匀视角为训练目标。

加入少量 learner-induced states 的专家重新标注（DAgger-style），避免只在原始 GT 路径上训练然后在偏航状态失效。保留任务重要的导航停止（离目标近时）案例。避免训练成只会选择固定四次：此阶段只验证空间能力，不宣称学会了动态计算预算。

监督验收比 SR 更早：教师保留集候选排序准确性、camera 命中/位置误差（必须按候选质量解释）、真实匹配环境中 one-chunk gain、ROI有效新增覆盖、重复率、与等四次固定/随机策略的对照。初始目标作为诊断建议：相较 3.52% 覆盖至少有显著改善、重复比例不再100%；但不能仅靠覆盖过门，必须有 task-grounded gain，否则停止扩大预算。
5. Gate 2：小样本 Group-Relative Actor Warm-up / GA-PPO 可行性门禁
在监督 Stage 1 真正改善空间导航行为后，使用当前 warm-up Actor 的真实 on-policy 采样建立同状态组，先小规模 K=4/8、16–32 个 Group anchors 验证。GT教师候选不可以冒充从当前策略采样的 on-policy group。
- 同一 anchor 依 Task231 已验证的 reset + deterministic prefix replay 精确恢复观测/历史/pose。每个分支从相同旧 Actor 独立采样完整0–4 Focus序列，执行一个导航 chunk 并计算环境任务奖励。
- 组内 Q 优先以实际短期导航 reward比较；Critic held-out EV 接近0，故不要默认其 bootstrap 给出可靠增量。若采用 bootstrap，应审计其偏差及方差，并提供 one-step-only 消融；正式 Task2.4 才建立完整 r+γV_old 的统一算法协议。
- A_i=Q_i-mean_{m≠i}Q_m，无组内小方差除法；Group anchor 的 Actor 不再叠加 GAE。每组先平均 K 条 loss，再按导航 anchor 平均；真实主分支必须在知道奖励前随机选择，不得选 best-of-K 来继续导航。Critic 若训练，只使用真正主轨迹的 return。
- 在任何候选采样、评分之前落盘主分支索引、旧策略hash与独立分支RNG。reset+prefix恢复须核验pose、观测/指令、模型历史缓存、融合状态、episode步数、终止/截断和任务度量；仅恢复相机pose不合格。完成旁支后重新恢复并重放预选主分支，核对其实际后继状态后才继续主轨迹。恢复失败整组拒绝，保留失败证据；旁支不可推进主轨迹cursor、RNG或预算。
- 数据显式标识 `teacher_candidate`、`on_policy_group_branch`、`main_transition`，记录父anchor及主分支身份。监督候选无论是否来自Actor proposal均不得进入on-policy组损失；旁支不得进入Critic、GAE或主轨迹return。Group Actor与导航GAE的anchor集合互斥，禁止同一anchor双重优化。bootstrap必须来自冻结旧Critic，并分别正确处理terminal与truncation。
- 先无或极低 Focus 成本测 Group 训练能否改善 camera/position 偏好、同预算导航进展和重复行为。不要把 group reward 与新增覆盖 reward 大量叠加。
若 Group anchor 间实际导航回报差远小于执行噪声，或恢复不一致、奖励几乎平局，则该 Group 机制信息不足；停止加大 K/anchor 比例。通过门禁后，再以 Task2.4 的正式设计（K=8、约12.5% anchors、GA-PPO与PPO-only同成本对照）开展完整训练。
6. Gate 3：导航级 PPO + 逐步学习“是否值得继续看”
只有在空间策略已可测地优于非学习的等预算参考时，才展开预算优化。保留原导航 terminal/真实进展 reward 作为最终任务目标；不把教师 imitation loss、coverage 或特征KL永久叠加为不透明的累计导航奖励。

建议分离两类信号：
- 在哪里看：来自专家/组内同状态的导航收益差；训练 camera/position 的条件动作分布。
- 何时停止：比较当前已融合状态选择 STOP 与再做一次 Focus 后的边际导航收益，与一次 Focus 真实计算成本比较。样本确实有净增益时，监督/Group相对优势推动继续；没有收益时推动 STOP。早期可在训练集获得边际对照作 STOP 教师标签，不向推理输入 GT。
成本做预声明的 curriculum：空间阶段 c_focus≈0（或极低），导航级对齐阶段 c≈0.0025，模型真正能选择有益 Focus 后再在单独版本逐步测试 c=0.005/0.01。这些是候选配置，不是推荐必定有效的超参；每次变化要重新计算真实 reward/return，并审计 Critic 对目标切换的适应，必要时重建 Critic/optimizer，不能复用不同reward版本的旧buffer。最优成本须按 SR/SPL—平均Focus/延迟的 Pareto 曲线选，而不是要求最终强制达到0.01。

注意 PPO sequence 中所有决策共享导航级 advantage 有原始策略梯度意义，不能直接把求和改平均。微步边际教师/同状态组相对奖励应作为独立且可溯源的训练信号，与导航级 GAE 在同一 anchor 不重复加权，防止伪精细信用。
7. 省时执行顺序、预算上限与早警
避免再次直接跑4096：
1. Gate0（纯评估+少量环境反事实）：先64 anchors，K≈8个感知候选；廉价 expert-action error 预筛选后，仅对较少候选做严格匹配的真实 chunk 比较。无可辨别标签则立刻停止，转诊融合和 frozen nav 作用机制。
2. Gate1（监督热启动）：先用 Gate0 高置信样本验证 loss 能降、保留集候选排序能升、确定性四Focus不再全重复；再扩充到约128–256 anchors，加入自主偏航状态专家再标注。教师生成调用数须单独报告。对照 update14、`Task22 init` 与 fixed/random同预算。
3. Gate2（Group pilot）：首批16–32 anchor，K=4先做方差/恢复门禁，通过才考虑K=8。不要把额外分支操作误计为主轨迹 transitions。
4. Gate3（短 PPO）：每个有希望的配置先采集256/512新主轨迹 transitions，必要时扩至1024；只有当空间、任务价值与成本指标都保持合理才安排2048/4096。每128–256 transitions 固定状态头探针，每512 transitions 小规模真实导航检查；少量验证集不单凭最高SR选模型。
并行观察四组核心指标：
维度	重要监测
Spatial	四相机 top1 分布、j=0→3条件变化、重复IoU、实际union coverage、同预算learned-vs-fixed/random的一步导航差
Task	SR/SPL/OSR−SR、nDTW、导航进展、接近目标不终止案例、分train/dev场景效果
Stop/Compute	p_stop P10/P50/P90、0–4 Focus分布、采样vs确定性差、平均视觉encoder调用与延迟
Optimization	teacher label margin/quality、分头梯度、loss/entropy/KL、Critic held-out EV/预测偏差、group间Q方差、主轨迹vs分支总成本


示例工程预警（须预注册、非统计结论）：确定性N4重复率连续两次仍100%；或独立matched one-chunk gain为零/负、可信标签比例过低；或 STOP零Focus比例快速越过50%；或 Critic held-out EV持续近0同时KL/位置崩溃。出现这些情况暂停扩预算，诊断而不是增加训练时长。
8. Task2.3.3 交付与停止条件
新增单独 Task233/，保留并引用Task23/231/232原始证据，不改写旧结果：
- TEACHER_PROTOCOL.md：expert导航动作在当前pose的定义、SimpleNav动作比较坐标、候选抽样、合法GT训练split、反事实环境恢复精确性、质量门槛及置信度。
- teacher_candidates.py / expert_scoring.py / spatial_warmup.py：可替换接口，复用 Task22/Task231/Task232。必须显式区分 candidate-ranking（监督）与 on-policy group（RL），不可把教师候选误当 on-policy。
- group_pilot.py：小规模 K=4/8 组内相对优势、one-step vs Critic bootstrap、主分支预选、同状态核验；若未过门禁，报告阻断原因而不是假装GA-PPO完成。
- ppo_recovery.py：Focus成本 schedule、STOP边际价值诊断、原PPO runner复用、actor/critic/optimizer/snapshot checkpoint可重现；严格记录新奖励版本。
- PROGRESS_GATE.json、`teacher_signal_audit.json`、`paired_effects.csv`、`training.csv`、`REPORT.md` 和全部配置/manifest/hash；含失败尝试与真实执行成本。
Gate 0 没有可学习导航增益 → 不进行大规模 SFT/RL，单独升级到视觉融合/动作敏感性诊断。Gate 1 只改善覆盖，不改善真实任务 → 不能声称空间 warm-up 成功。Gate 2 没有可靠组间差异 → 不推进正式 GA-PPO。Gate 3 失去 SR/SPL 或重新0/4塌缩 → 停止扩训、保留可解释checkpoint与负面结果。

9. 统一验收与失败关闭
- 门禁状态使用 NOT_RUN / BLOCKED / INCONCLUSIVE / FAIL / PASS；只有上游PASS且证据hash、协议版本一致才能启动下一阶段。停止必须保存最后有效checkpoint、失败样本、实际预算和原因；恢复执行需显式新版本审阅，不自动追加4096。
- learned/fixed/random在同一episode、anchor、真实历史与Focus预算下配对，随机种子先在状态内平均，再按episode聚合bootstrap并报告scene分层。no-Focus天然为零Focus，不能伪称与四Focus等计算量：以同导航chunk预算作为零成本基线，另报真实encoder调用、延迟与预算—收益曲线，不靠虚假调用填平预算。
- Gate1冻结STOP参数仍不足以保证STOP输出不变（共享trunk可改变输出）；应显式绕过感知STOP按预注册N执行，并从监督loss移除STOP。保留自然导航STOP，报告真实j=0..3相机条件响应和有效历史干预，不强制相机均匀。只有保留状态真实收益和空间行为同时过门，才能进行Gate2。
- Gate3每256新主轨迹transition做头探针、ROI重复/coverage、STOP与held-out Critic检查；每512步做固定导航集验收并报告SR/SPL、配对区间、终止失败和成本。256步已触发硬错误或价值退化时立即暂停，不等512步；1024及以上不是默认预算。Critic报告all/terminal-only EV、偏差与目标版本，验证样本不参与Critic拟合。
- 导航验收阈值、SR/SPL非劣容限、重复ROI及零Focus告警阈值、KL/梯度有限性、连续告警次数和成本升级条件均在run前写入配置；成本升级不能以reward变好替代SR/SPL保持。固定0/4策略只作为对照，不强制模型达到某个Focus均值。
- 本阶段Group只做K=4、通过后K=8的限量机制验证；任何K=8结果均不得标为Task2.4正式GA-PPO比较。未过门的下游脚本可规划但不得写成已实现或已执行交付。
