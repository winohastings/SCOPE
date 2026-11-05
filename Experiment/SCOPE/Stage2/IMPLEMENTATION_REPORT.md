# Task4 执行实现与交付报告

日期：2026-11-05。角色：用户指定执行者。根入口/任务书与核心文档已读，缺失的
`agents/exec-agent.md`未被伪造。按用户补充先将审阅独立归档至
`../../../docs/Logs/Task3/STAGE2_REVIEW_REPORT.md`，再继续真实实现/验证。

## 六个最终问题

| 问题 | 实测回答 | 证据与边界 |
|---|---|---|
| Focus是否有可测潜在收益？ | **有局部潜在收益，不保证整体有益** | 54次同状态分支中train起始fixed相对Glance多0.13704m progress；fixed总体6条件均值-0.16993m，正负混合；不是导航改善证明 |
| 空间对齐是否需要adapter？ | **当前证据不足以证明必须** | 几何/grid/scatter正确，空间文本作用不一致；提供零初始化可选adapter但默认关闭、未训练 |
| 完整状态条件Actor–Critic是否正确运行？ | **在本轮小规模合同内通过** | 同一Actor五开关，顺序Focus、增量记忆、独立Critic、State GRU、感知STOP；A/B/C真实更新与前向反向通过，不代表大规模稳定 |
| GRU是否用真实导航反馈且无泄漏？ | **是，按实现与测试合同** | 实际执行后的body位移/误差才更新；B/C真实State GRU梯度非零；PolicyObservation无GT，train/dev权限与scene/ID隔离；只有长度1截断BPTT |
| PPO能否有效更新？ | **是** | 主验证24transition/4update，旧联合ratio首epoch=1、参数变化0.245–0.342等、主干grad=None；额外恢复训练2transition/1update；固定target/掩码/RNG恢复测试通过 |
| 实测吞吐与完整成本？ | **C为0.6103 transition/s（episode）或0.3592（含phase启动）** | 参考199680transition/seed线性外推约90.89–154.41墙钟小时/seed，三seed双卡约545.32–926.49 GPU小时；不含长期历史/额外评估等，非预算上限或授权 |

## 执行结果

- 第一阶段只读逐行核对256episode、3768step，确认G1>G0>G2/G3的描述性SR排序。
- 正常导航82个完成策略步；四个训练rollout24transition，开发训练前/后/恢复各两集。
- 同状态诊断：train/dev各1episode、2物理状态，共6个状态×历史条件×9分支=54次。
  主诊断重测只修正native对照中的历史数值混杂，不改变模型配置或训练结果。
- 当前缓存严格每图一次：Glance四图一起、Focus各一次，导航复用同一postmerge特征。
  相同调度原生重编码与缓存特征逐位一致；匹配历史后的原scatter动作也逐位一致。
- 独立Actor/Critic与反馈闭环通过。训练预算实际覆盖0、1、2 Focus，并记录明确感知STOP。
- 固定开发SR为0.5→0.5，mean nDTW为0.47005→0.58335；两episode成败交换，**不宣称
  导航改进**，没有依据结果调参或选择checkpoint。保存恢复后开发指标全部精确一致。
- 真实训练恢复从cursor继续episode3，两步后第3个C update通过，不重复训练episode1。
- Stage2单元20项、全部SCOPE34项、现有导航5项回归通过。根全套有1个既有模块导入
  缺失，未擅自修复，详见测试报告。
- 9013个历史保护文件（含原冻结checkpoint）SHA256全部不变；另核对Stage1列出的10项
  外部依赖/旧scope入口hash全部不变，Task1列出的238个legacy运行时文件hash也全部
  不变。不存在修改第一阶段历史或预训练权重的行为。

## 相对旧版本的代码变化

全部实现集中新增在`Experiment/SCOPE/Stage2/`，不补丁旧视觉/外部导航源文件：

| 文件 | 相对第一阶段新增 |
|---|---|
| features.py | typed当前特征bank、原图ROI/相机几何/射线、候选低分辨率摘要、可选空间残差adapter |
| policy.py | 同一ROIActor五开关、增量Focus GRU、独立Critic、真实反馈后State GRU与episode reset |
| trajectory.py | 真实body位移、预测执行误差、执行反馈类型与成本感知reward |
| training.py | 独立optimizer、联合顺序PPO、GAE、固定target/合法mask、原子checkpoint和RNG恢复 |
| data_guard.py | 复用原train/dev划分、拒绝heldout/跨scene、annotation/checkpoint及历史保护hash |
| runtime.py | 扩展现有ScopePolicy、原融合入口当前缓存复用、真实反馈/训练/恢复、诊断重编码审计 |
| server.py | 原session RPC兼容入口，新增feedback/finish/configure/restore，不改导航STOP |
| real_rollout.py | 复用原worker/backend/actions的真实导航、step_filter接触测量、同状态多分支诊断 |
| run_pilot.py | 受限八case编排、GPU采样、冻结保护、失败保留与自有服务清理，拒绝占用他人端口 |
| learning_sanity.py | 真实train特征的8样本Actor过拟合健全性，不当作导航性能 |
| audit_results.py | 第一阶段逐行复核、最终训练/诊断/恢复/吞吐/GPU/保护审计 |
| ../../../tests/test_scope_stage2.py | 20项单元测试，覆盖32种Actor开关组合、训练/泄漏/恢复/几何合同 |

文档：本目录八份任务报告、README与完整文件清单；新增Task3审阅报告并更新
`../../../docs/index.md`索引。源码、文档、原始日志、配置、episode清单、Actor/Critic
checkpoint逐文件路径/哈希均列于`FILES_MANIFEST.tsv`，避免仅列几个关键文件而遗漏。

## 数值与失败的透明记录

第一次混用旧全batch与新顺序特征的诊断被一致性断言拦截，未删除失败尝试；像素差0
而特征不同，最终同调度重编码特征差0。旧BF16分组差异最高relative-L2约0.07275，
不伪称新调度与第一阶段任意旧batch数值等价。新的默认路径从未为导航重新编码当前图。

随后native-scatter对照只保持当前路径相同但混入历史batch差异，首次完整对照保存到
`release/diagnostic_attempts/`。修正对照为“历史同一特征，仅替换当前特征来源”后54次
重测，特征和动作最大差均0。额外GPU开销记入总体窗口，不混入主训练吞吐。

更早接线/奖励测量修订（`previous_step_collided`不保证由自定义连续动作更新，改为
原生step_filter请求/过滤位移）均保留旧日志，最终主运行使用修正后实际测量。
这没有修改冻结模型、数据划分、ROI候选、导航动作或外部实现。

## 交付和复现

最终实测入口：`artifacts/release/AUDIT.json`；主运行完成门禁：`release/COMPLETE.json`。
后续诊断对照和真实训练恢复的独立完成记录：`artifacts/FINAL_VALIDATION.json`。
资源总账：`artifacts/resource_accounting.json`，历史保护：`release/protection_audit.json`。
含失败/冷启动/调试的全任务双卡保守预约窗口上界约1.59473 GPU小时，不是利用率积分。

```bash
/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope_stage2.py'

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/audit_results.py \
  --output Experiment/SCOPE/Stage2/artifacts/release

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/run_pilot.py \
  --output Experiment/SCOPE/Stage2/artifacts/recheck \
  --port 10096
```

详细分步服务/训练/恢复命令见README。执行完成后关闭本轮自有服务与采样进程，不结束
无关进程。没有启动全规模PPO，也没有val_unseen评估；本次架构及小规模验收通过
**不自动授权完整训练**，后续仍需独立实验协议和长历史预算验证。
