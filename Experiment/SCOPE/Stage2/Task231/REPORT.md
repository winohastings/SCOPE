# Task2.3.1 失败归因诊断报告

角色：执行者（含结果分析）。日期：2026-11-07。只写Task231，不修改Task23，不重跑4096训练。

## 核验与执行范围

- 真实5个checkpoint、原始固定32集及训练trace对照原ARTIFACT_MANIFEST逐项SHA256核验。
- train/val_seen各128个相同detached pre-Focus状态，5 checkpoint共1280次Actor探针；Actor输入无reward/GT。
- 全部4096训练transition、56次更新重算；原5轮160集配对；新3×2×32集对照（复用首次尝试已完成项）。
- matched-budget：16状态、128分支，128分支恢复检查有效；含learned/fixed/3随机种子/无Focus。
- 新增5项单元测试与原Task23的11项回归测试通过；未执行optimizer更新。

## 三个门禁问题

### 1. 后期forced4是否挽救性能？

| transitions | 诊断模式 | SR | SPL | nDTW | mean Focus |
| --- | --- | --- | --- | --- | --- |
| 1033 | forced4 | 56.25% | 42.19% | 0.4243 | 4.000 |
| 1033 | native | 56.25% | 42.19% | 0.4243 | 4.000 |
| 2054 | forced4 | 43.75% | 37.11% | 0.4454 | 4.000 |
| 2054 | native | 43.75% | 37.11% | 0.4454 | 4.000 |
| 4096 | forced4 | 53.12% | 44.43% | 0.5241 | 4.000 |
| 4096 | native | 34.38% | 26.05% | 0.3641 | 0.000 |

4096 native→forced4配对ΔSR=18.75%，95% bootstrap CI=[0.0, 0.375]，exact McNemar p=0.109375。
**观察到部分恢复，但SR配对检验未排除小样本波动。** forced4下最终模型已接近1033的SR，并不支持‘后期camera/position全面失效’。新native与历史差异见diagnosis.json/native_reproduction。

### 2. 1033→2054退化来自哪里？

forced4配对ΔSR=-12.50%，CI=[-0.28125, 0.03125]；两者原生均4 Focus，因此STOP次数不能解释这段退化。
历史6个成功→失败、2个失败→成功；失去的6集中，752/536/107曾有oracle success，206/111/338未进入成功范围。
752/536/206/338截断；107/111错误位置停止。536最终仍在3米内但未真实终止。碰撞、绕行和目标附近不终止共同出现，不能仅凭整体nDTW上升认定导航更好。
固定状态1033→2054的camera/position分布漂移（含部分left→right切换）见策略CSV；局部因果对照如下。

| 1-chunk配对进展差（米） | episode均值 | 95% episode-cluster bootstrap CI |
| --- | --- | --- |
| 2054_learned_minus_random3 | -0.012056 | [-0.08750637720028559, 0.042720874150594085] |
| u14_learned_minus_no_focus | 0.010068 | [-0.017959075421094893, 0.04979240894317627] |
| u28_learned_minus_no_focus | 0.007532 | [-0.020911216735839844, 0.04869341850280762] |
| u56_learned_minus_no_focus | 0.005522 | [-0.016156554222106934, 0.036459740996360776] |
| 2054_learned_minus_fixed | 0.050745 | [-0.037928004562854764, 0.1976300209760664] |
| 2054_learned_minus_1033_learned | -0.002537 | [-0.008024584501981735, 0.00149779170751571] |

同预算ROI只改变感知决策；reset+固定前缀重放后，姿态、图像/历史、指令、pre-Focus特征及BATS blocks逐项匹配。不一致分支不合并。
跨场景teacher-MSE、覆盖率及动作变化见matched_scenes.csv/matched_action_differences.csv；MSE下降并不自动带来chunk进展，更不等价于最终SR。

本次learned四次Focus覆盖总四相机面积约3.52%，近似在同一ROI反复查看；固定四相机中央覆盖14.06%，随机约13%。三模型learned的teacher-MSE均改善约4%，而fixed/random约14%。该差异不能单独证明Adapter失效，也不支持现在重校准。

### 3. 旧reward是否产生负面成本动机？

是：固定相同导航结果时4→0 Focus立刻增加0.04 reward；在progress未裁剪区，相当于0.6米进展。4 Focus加step成本需0.75米进展才能抵消。
4096训练累计progress +55.829，focus cost −52.89，step cost −40.96，terminal +121.5。成本量级足以形成省感知动机，但不证明是全部退化的因果来源。
历史1033→4096的每集Focus成本节省+0.6125，超过terminal奖励损失−0.515625；因此平均reward升高可以与成功数下降同时出现。完整分解见validation_reward_components.csv。
最终训练区间仍包含0/1/2/3/4 Focus，mean约0.984；最终固定val状态首步p_stop中位数约0.572，不是采样策略完全不Focus。0.5确定性阈值将此变为全部立即STOP。

注意：Actor的STOP只结束感知，不直接结束导航；导航waypoint_motion_stop与环境截断是另一条机制，不能混用这两种STOP。

## 数值、Critic和统计边界

- reward/return/normalized advantage/old logprob/未覆盖token最大误差：{'reward_max_abs': 0, 'return_max_abs': 0, 'normalized_advantage_max_abs': 0, 'old_logprob_max_abs': 0, 'uncovered_token_max_abs': 0}。
- head_gradients.csv为每更新每Focus count首个样本的整个head未裁剪梯度，不冒充全量optimizer时刻梯度；numerical_transitions.csv含全部样本长度、loss、KL、clip。
- gradient_components/head_gradient_components.csv进一步分离policy与entropy梯度；STOP-only样本上的camera/position非零entropy梯度不等于这些动作获得了reward信用。
- held-out Critic EV在5轮总体约0.00005/0.0418/0.0043/0.0026/0.0016；terminal-only亦近0。最终预测均值约−0.650，而目标均值约+0.080，说明基线质量弱且偏移。
- 历史0→1033的exact McNemar未校正p=0.015625，不应写成‘绝无显著差异’，但对10个探索比较Holm校正以及best-checkpoint选择后不支持推广结论。
- 1033→2054历史ΔSR=−12.5pp，95%CI=[−28.125pp,+3.125pp]，p=0.2890625。32集只提供诊断证据。
- Actor-only预期Focus数是静态首状态代理，未来Focus会改变输入，不能将代理列误读为真实序列期望。

- SR严格沿用原Task23的真实导航终止且distance<3m定义；截断时距离<3m不自动记成功。当前两套环境版本已记录，原run无完整package lock，不声称历史环境逐包一致。

## 假设判定与优先级

| 假设 | 判定 | 优先级 | 边界 |
| --- | --- | --- | --- |
| H1 | 部分支持 | 1 | 去掉成本能否改善训练必须重新采样，当前禁止重训。 |
| H2 | 部分支持 | 2 | 不能据此确证训练集过拟合或认定PPO方差是唯一病因。 |
| H3 | 仍不确定，存在可检验的局部证据 | 2 | 有限初始/一段前缀状态不能代表完整导航分布；不据此自动重校准Adapter。 |
| H4 | 反证数值实现断裂；信用分配仍不确定 | 4 | 序列求和的长度效应存在，但不是bug；不改为mean。 |
| H5 | 支持停止失败和小样本不确定性 | 1 | 单seed固定32集与best-checkpoint选择偏差不支持泛化结论。 |

**不启动正式优化。** 三问已有可审计的观测层回答，完整因果分解仍有限。先审阅诊断，再决定单因素成本/停止规则或Critic稳定性实验；不提前重写融合模块，不无依据把序列sum改为mean。

## 产物与复现

入口README.md；机器可读总判定diagnosis.json；所有新增文件列于FILES_MANIFEST.tsv。原始checkpoint/数据仍依赖本机可信路径，不声称独立离线可移植。
