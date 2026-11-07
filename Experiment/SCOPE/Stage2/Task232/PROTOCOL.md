# Task2.3.2 冻结短续训协议

日期2026-11-07；执行者（含分析）。角色文件缺失，遵循根AGENTS、Task2.3.0/2.3.2与用户补充。
只写Task232，不改Task21/22/23/231。依据Task231已核验的诊断，不声称H1/H2唯一因果。

## 分支与完整状态迁移

- 可信来源：Task23/ppo4096/update_0014.pt，1033 transitions、update14、下一个train ordinal70、空buffer。
- U门禁：原成本/学习率/rollout配置，新增64真实transition；首episode与历史同episode对比前向；
  对同批新数据做一次真实更新，再restore重做验证参数/optimizer/RNG/cursor逐位一致。
- A：Focus cost 0.01→0.0025，其他算法超参数不变。
- B：原Focus cost0.01；rollout minimum64→128，Actor LR1e-4→3e-5，Critic LR2e-4→1e-4。
  minibatch16、epochs2、clip0.2、熵系数、梯度裁剪和Actor/Critic结构不变。
- 三者都保留完整Actor/Critic、old snapshots、AdamW moments/step、随机数和下一episode顺序。
  B只覆盖optimizer参数组LR，不清零动量；A保留旧Critic和其动量，明确存在目标尺度适应期。
- 用新manifest/config/provenance/实验目录生成迁移checkpoint；不冒称Task23原样续训。
- train列表及顺序原字节语义保持；独立val增补只改变授权评估集合，绝不加入训练数据。
- 只更新当前新run采集的training-sampling轨迹；历史trace仅可读用于探针/一致性分析，不进入buffer。
- A/B第一阶段各512新增transition；256与512在真实episode边界执行更新和固定状态probe。
  为精确预算而结束的episode标记truncation并用实际后继Glance bootstrap；不将其标terminal。
  episode级buffer合并保证每次更新>=rollout minimum；实际batch可能超过配置下限。

## 评估与空间诊断

- 原固定32集作为历史可比诊断，保留原SR定义：导航真实终止且距离<3m。
- 新固定32集val_seen：seed232，从全val_seen按scene平衡抽样，排除原32的episode与trajectory。
  相同场景并不声称scene-disjoint泛化；新集只评估，使用后称独立开发集，不冒称最终测试集。
- BASE(1033)先评估新32；A/B在512各评估原32+新32，均不选中途最高SR checkpoint。
  原32中的预固定12集可同时作为快速诊断，但不替代新集判断。
- 固定Actor probe：train/val_seen各128相同detached pre-Focus状态；无GT/reward输入。
  STOP分位数/阈值、camera熵/占比、position mean/std及相对fork KL；256前后共享状态。
- 固定状态几何probe：在相同历史4-decision状态链上重算各模型ROI并统计union覆盖/同相机IoU。
  这属于冻结参考状态链条件下的几何诊断，不伪装为实时重编码闭环。另统计新真实rollout的
  Focus分布、覆盖和N=4条件覆盖，避免把静态代理当真实行为。
- 重复ROI定义：同相机且与之前任一ROI IoU>=0.8；覆盖按四相机总面积4×512²归一化。
- Critic在独立集真实rollout上计算discounted-return EV和MSE；另报terminal-only，截断bootstrap
  与训练采样/确定性评估的policy差异明确保留；A同时报告原reward与branch reward。
- 原4096模型STOP threshold0.45/0.60在原快速12集纯评估，0.50使用Task231已复现原结果。
  这是解码敏感性对照，不算策略学习成功，也不据此重选独立集。

## 扩至1024的预设门槛

只有512的新真实结果满足全部条件，才继续该分支至累计1024新增transition：

1. 固定val pre-Focus p_stop>=0.5比例<=10%，新独立集确定性零Focus比例<=10%；
2. 独立集N=4联合覆盖相对BASE至少提高25%，且重复ROI比例降低至少0.15；
3. 独立集SR不低于BASE超过1/32，SPL不低于BASE超过0.03；原32不出现大于2/32的SR下降；
4. 独立集terminal-only Critic EV不低于BASE超过0.05，所有finite/freeze/ratio/restore门禁通过。

这些为工程筛选条件，不是统计显著性检验。配对episode bootstrap/McNemar另报。
若没有分支同时改善空间行为与导航/稳定性，停止于512，不因单次SR最大而扩展。
仅当A在感知行为/成本上有证据且B在Critic稳定性上有互补证据，才提出A+B；不自动训练组合。
若A/B都不能改善空间行为，单独讨论轻量重复约束后续协议，不直接加复杂ROI奖励。

## 禁止事项与成本

不启动GA-PPO，不重训Adapter，不改Actor/融合，不完整4096重训，不访问val_unseen。
新增训练预算分U/A/B列示；重复恢复更新不计新增transition。评估transition单列。
记录wall/model/perception时间、optimizer步数及GPU采样；共享GPU系统功耗不得伪作分支独占能耗。
全部执行以run artifacts为准；找不到合格配置也是有效结论，不强行宣称Task2.3.3门禁通过。
