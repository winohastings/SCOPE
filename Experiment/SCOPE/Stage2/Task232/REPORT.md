# Task2.3.2 定向短续训报告

角色：执行者（含分析）；2026-11-07。全部新内容仅在Task232，保留原视觉融合、Actor及冻结导航模型。

## 执行与门禁

- 从update_0014完整checkpoint分叉：1033 transitions、update14、下一train ordinal70；保留Actor/Critic/old、optimizer动量与RNG。
- U门禁新增64条真实transition；首episode13条状态/动作/logprob/value与历史逐位一致。U/A/B各自首更新restore重放逐位一致。
- A成本0.0025；B成本0.01、rollout minimum128、Actor LR3e-5/Critic LR1e-4；其余算法参数不变。每分支512新增on-policy数据，不复用历史buffer。
- 256/512真实预算边界probe；精确预算截断使用真实后继bootstrap。B的实际batch可能因episode边界及最小batch门禁大于128。
- 独立32集按seed232固定，排除历史32的episode及trajectory；原32保留为历史诊断。仅用新集配对效果和空间/Critic门槛共同决策，不取单次最高SR。

## 独立集基线与结果

BASE1033：SR=37.500%，SPL=30.262%，N=4覆盖=0.035168，重复比例=1.000000。

| 分支 | 集合 | SR | SPL | mean Focus | N=4联合覆盖 | N=4重复比例 | Critic EV / terminal-only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | historical | 34.375% | 27.461% | 4.000 | 0.03518688777261529 | 1.0 | -0.01011 / -0.00871 |
| A | independent | 28.125% | 21.936% | 4.000 | 0.03519319958157009 | 1.0 | -0.00078 / -0.00511 |
| B | historical | 37.500% | 28.229% | 4.000 | 0.035160112984572785 | 1.0 | 0.12755 / 0.10921 |
| B | independent | 34.375% | 25.778% | 4.000 | 0.03516128960005734 | 1.0 | 0.01701 / -0.02752 |

覆盖按四相机总面积归一化；重复定义为同相机且与历史ROI IoU≥0.8。真实N=4覆盖与固定参考状态几何代理严格区分。

## 配对证据与早期趋势

- A独立集ΔSR=-9.375%，95% episode-bootstrap CI=[-0.21875, 0.03125]，McNemar p=0.375。
- B独立集ΔSR=-3.125%，95% episode-bootstrap CI=[-0.15625, 0.09375]，McNemar p=1。

| 分支 | 新增步数 | 固定val p_stop中位数 | 首步零Focus比例 | camera entropy | Position KL相对fork | 条件N4覆盖代理 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 0.27060 | 0.000% | 0.92741 | 0.000000 | 0.035159 |
| A | 256 | 0.30569 | 0.000% | 0.97517 | 0.002160 | 0.035185 |
| A | 512 | 0.31404 | 0.000% | 0.88103 | 0.017430 | 0.035179 |
| B | 0 | 0.27060 | 0.000% | 0.92741 | 0.000000 | 0.035159 |
| B | 256 | 0.27645 | 0.000% | 0.82095 | 0.000214 | 0.035165 |
| B | 512 | 0.28366 | 0.000% | 0.87381 | 0.000435 | 0.035162 |

Position KL为相同pre-Focus状态上、按fork相机概率加权的latent位置分布KL，不是像素位移。各相机位置mean/sigma逐状态保存在runtime/probes的CSV；camera top1占比见early_probes.csv。低KL只说明较少漂移，不代表覆盖改善。

## 结果解释

独立基线Critic EV(all/terminal-only)=0.042795/0.047569。
- A真实四Focus覆盖相对基线变化=0.0727%；独立集零Focus比例=0.000%；原reward定义Critic EV(all/terminal-only)=-0.002062/-0.005394。
- B真实四Focus覆盖相对基线变化=-0.0180%；独立集零Focus比例=0.000%；原reward定义Critic EV(all/terminal-only)=0.017010/-0.027516。
- A降低了每次感知的直接惩罚，但未观察到空间探索改善；固定状态STOP概率仍上升，不能把降低成本本身当作学到了有效注视。
- B的STOP/Position漂移较小，但独立集terminal-only EV未改善；历史集较好的EV不能替代独立集证据，也不足以支撑A+B互补假设。
- 两支当前确定性均为4 Focus，只能说明这512步尚未出现零Focus塌缩。相机/ROI重复与覆盖停滞仍存在，不能外推长期不塌缩。
- 采样训练的四Focus覆盖约0.10，而确定性评估约0.035；前者包含随机探索，不是部署策略空间覆盖已改善的证据。
- 独立SR配对区间均跨0：本轮可据冻结门槛拒绝扩训，但不能据此宣称A/B的真实效果已被统计证明为负。

## 1024扩展与A+B决策

- A扩展资格：False；未通过：four_focus_coverage_gain_ge_25pct, repeated_roi_fraction_drop_ge_015, independent_SR_loss_le_one_episode, independent_SPL_loss_le_003, historical_SR_loss_le_two_episodes, terminal_only_EV_loss_le_005。
- B扩展资格：False；未通过：four_focus_coverage_gain_ge_25pct, repeated_roi_fraction_drop_ge_015, independent_SPL_loss_le_003, historical_SR_loss_le_two_episodes, terminal_only_EV_loss_le_005。
- A+B互补证据：False。不自动训练组合；do not combine unsupported fixes; consider an isolated lightweight repetition-constraint diagnostic。

**本轮没有达到预先冻结的1024扩展门槛，不宣称找到Task2.3.3可直接扩训的稳定配置。**
若SR保持或偶有上升但重复ROI/联合覆盖没有改善，不能把它等同有效空间注视学习。保留1033作为对照，不据单次小验证选择A或B。
A/B结果支持下一步单独设计轻量重复约束诊断，而非直接上复杂ROI奖励或改Adapter。建议先纯评估‘同相机IoU≥0.8时抑制再次选择’并审计动作分布；若进入训练，必须正确记录约束后的条件概率，不能沿用未经修正的旧logprob。此建议未执行，不计为已获改进。

## 预算、成本与限制

| 分支 | 新增train | optimizer steps | train wall秒 | train model秒 | 更新日志秒（每逻辑更新计一次） |
| --- | --- | --- | --- | --- | --- |
| A | 512 | 68 | 1328.73 | 484.97 | 42.33 |
| B | 512 | 64 | 1357.92 | 514.02 | 43.51 |

A实际更新batch：[80, 71, 105, 76, 74, 106]；新训练Focus分布：{'0': 139, '1': 117, '2': 78, '3': 51, '4': 127}；reward分量：{'progress': 8.56421706676483, 'terminal': 29.0, 'focus_cost': -2.335, 'step_cost': -5.120000000000001}。
A完整run wall（含启动、probe和评估）=4064.26秒；表中train wall为episode计时之和，不含模型启动/全部管理开销。

B实际更新batch：[256, 256]；新训练Focus分布：{'0': 120, '1': 113, '2': 94, '3': 35, '4': 150}；reward分量：{'progress': 8.971427243451279, 'terminal': 31.0, 'focus_cost': -10.06, 'step_cost': -5.120000000000001}。
B完整run wall（含启动、probe和评估）=4074.38秒；表中train wall为episode计时之和，不含模型启动/全部管理开销。

- 新训练总计1088=64门禁+512A+512B；恢复重复计算不算新增环境样本，原1033预训练预算另列。评估transition见各run analysis。
- GPU csv记录整卡采样功耗和显存；并发模型/renderer共享设备，不能把整卡能耗当某分支独占能耗。训练、评估wall和模型/感知时间分别保留。
- Critic EV：独立集真实rollout，branch和原reward两套定义；终止集单报以排除truncation bootstrap循环。确定性评估与采样训练存在policy差异。
- A沿用旧reward训练的Critic及optimizer动量，存在新目标适应期；B是工程包，不能归因于LR或batch单因素。
- 单seed、短续训、32集不证明长期稳定或泛化；未扩展不等于该方向在更大预算下一定无效。
- U只做64步工程一致性门禁，没有同预算512步不变配置训练组；A/B相对1033基线的变化，不能单独证明修复优于正常512步续训。新旧评估集难度不同，也不能仅比较两套SR就宣称过拟合。
- 固定probe读取历史detached状态仅用于观测，不进入optimizer；覆盖门槛使用新独立集真实轨迹。
- 原4096阈值敏感性另见threshold_summary.json；阈值修改不是策略学习成功。

## 交付

repair_decision.json、extension_eligibility.json、branch_comparison.csv、paired_evaluation.json、early_probes.csv；可复现脚本与完整原始运行见README及FILES_MANIFEST.tsv。

## 原模型STOP阈值纯评估

| STOP阈值 | 12集SR | SPL | mean Focus |
| --- | --- | --- | --- |
| 0.45 | 33.333% | 20.648% | 0.000 |
| 0.5 | 33.333% | 20.648% | 0.000 |
| 0.6 | 41.667% | 34.200% | 4.000 |

该12集结果仅说明阈值敏感性，未改变模型参数，不当作策略学习改善。原Task23全部归档产物与Task231诊断证据最终完整性复核通过。

### 进程退出与产物完整性

- branch_A512：评估记录数={'evaluation_independent.json': 32, 'evaluation_historical.json': 32}，新增训练=512；日志OpenGL context清理异常=True。
- branch_B512：评估记录数={'evaluation_independent.json': 32, 'evaluation_historical.json': 32}，新增训练=512；日志OpenGL context清理异常=True。
- thresholds：评估记录数={'threshold_results.json': 24}，新增训练=0；日志OpenGL context清理异常=False。

COMPLETE在清理前落盘，不等同进程零退出码。本次执行器观测到A/B均在全部结果保存后退出码134，日志末尾为OpenGL context错误；阈值评估进程退出码0。未重跑训练，也未掩盖此异常。所有预期评估记录、无episode failure及预算另行核验，见EXECUTION_HEALTH.json。

## 独立集相同初态真实Focus对照

此补充不是固定参考中间状态代理：直接配对独立集相同reset初态，逐位核验pre-Focus tensor/姿态/context grid，再比较实际顺序编码Focus的结果。仅解释首chunk，不外推整集因果。

| 分支 | 有效同态数 | 实际覆盖 | 重复比例 | 首chunk进展差均值/95%CI（米） |
| --- | --- | --- | --- | --- |
| BASE | 32 | 0.035173 | 1.000000 | 0.000000 / [0.0, 0.0] |
| A | 32 | 0.035196 | 1.000000 | -0.000748 / [-0.01674380600452423, 0.010821159929037082] |
| B | 32 | 0.035162 | 1.000000 | 0.001489 / [-0.006186605617403984, 0.00796760469675064] |

## 原文Adapter C条件分支审查

C修复未执行：现有H3证据不满足直接负面证据前提。已从128条原始分支记录重算配对统计并核验16个既有Glance状态，详见C_REVIEW.md/C_REVIEW.json。该项是适用性审查完成，不是Adapter修复完成；A/B未改善不能代替H3证据。

## 用户新增C：覆盖奖励探索（已执行）

该C不同于原文Adapter修复。新增覆盖奖励beta=0.01与beta=0的C0等预算对照均已完成512新on-policy transitions，详细结果见C_REPORT.md、C_DECISION.json。未加入KL/JS，未改Actor/Adapter，未组合A/B。
空间探索筛查通过=False；探索变化不等同导航收益，不自动扩训。Task232累计新增训练2112=原1088+C512+C0共512。
