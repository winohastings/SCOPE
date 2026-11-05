# Task4 执行前审阅报告

日期：2026-11-05。按用户要求，审阅部分归档至 `docs/Logs/Task3/`；对应任务仍为
`docs/Tasks/Stage2/stage2_task4_focus_benefit_judge.md`，不将其误标为第一阶段的新结果。

## 1. 角色与审阅边界

- 角色：用户指定的严谨执行者，按 Exec Agent 执行。
- 已读根 `AGENTS.md`、`index.md`、`docs/index.md`。`agents/` 实际为空，引用的
  `agents/exec-agent.md` 缺失；不编造不存在的角色流程，遵守当前任务书和用户约束。
- 只读审阅第一阶段协议、结果汇总和相关实现；不遍历加载全部历史日志，不读取留出评测
  结果来选择模型或参数。基线冒烟日志只用于确认已验证的服务、模型、动作和环境接口，
  其中历史 val_unseen 成绩不作为本轮设计依据。
- 仓库根不是 Git 工作树；用源文件哈希和保护清单记录本轮变更，不提交 commit。

## 2. 依据与来源

已审阅的核心材料：

1. Task1 主报告、R2R 专项报告，以及 `Experiment/SCOPE/README.md`。
2. Stage1 的 `STAGE1_EXPERIMENT_PROTOCOL.md`、`IMPLEMENTATION_AUDIT.md`、
   `SELECTOR_DESIGN.md`、`REMAINING_IMPLEMENTATION_TASKS.md`、
   `METRICS_AND_STATISTICS.md`、`COST_MEASUREMENT_PROTOCOL.md`。
3. Stage1_Feasibility 的实现报告、可行性报告、计算成本、失败分析、导航指标、执行清单、
   数据划分和环境修订；结果来自实际 G0–G3 聚合产物，而非聊天记忆。
4. `docs/Logs/2026-11-05-simplenav-r2r-rxr-baseline-smoke.md` 的现有启动接口。
5. 当前 `scope_server.py`、legacy RXR 服务、Qwen 视觉预处理/当前特征融合/历史缓存、
   视觉块构造，以及 SimpleNav worker、Habitat backend/runtime/action adapter。

指定上游为 `winohastings/SCOPE`，执行开始解析到 HEAD
`e52fe5780aea3c076d5226d285c15a1e357068ff`。已读取该提交 README 和发布文件清单；
该仓库是本项目第一阶段的定向研究归档，并非独立的新版 Actor/PPO 代码。
本地已有对应实现，采用发布清单逐文件核对来确认继承关系，核对记录见
`Experiment/SCOPE/Stage2/artifacts/upstream_provenance.json`。完整 clone 下载较慢，
不把尚未完成的 clone 声称为已完成，也不因此重建或覆盖已验证运行时。

## 3. 第一阶段事实

R2R 原 train 来源中隔离的 64 个开发 episode、7 个开发 scene、固定单 seed；不是正式
val_unseen，也不是多 seed 显著性试验。

| 组别 | SR | SPL | nDTW | 记录失败数 |
|---|---:|---:|---:|---:|
| G0 原始兼容输入 | 0.4688 | 0.3941 | 0.4103 | 0 |
| G1 仅 Glance | 0.5625 | 0.4396 | 0.4133 | 0 |
| G2 Fixed-Cyclic-2 | 0.3750 | 0.2952 | 0.3328 | 0 |
| G3 Random-2 | 0.3750 | 0.2919 | 0.3477 | 0 |

结论仅限：现有固定/随机 Focus 未在这批开发样本上优于 Glance。不能直接推出所有裁剪
无信息、顺序学习策略无效、或空间适配器必需。闭环轨迹与历史长度不同，不能从总耗时
差异单独推出当前视觉编码的因果加速。

原开发/训练划分为 54 个训练 scene、7 个开发 scene；训练 9837 集，开发来源 982 集，
开发评估固定 64 集。新训练严格复用该划分，不用开发样本做 PPO 或小型过拟合测试。

## 4. 实现审阅发现

| 审阅项 | 事实与风险 | 后续处理 |
|---|---|---|
| 当前视觉特征 | 原流水线先选图再统一编码；图像选择器没有编码后 Glance/Focus 接口 | 增加 typed 当前特征 bank，先 Glance，再按选择获取 Focus，并在原融合处复用 |
| 空间语义 | Focus 继承源相机 TVI，但局部 ROI 坐标没有自动变成冻结 VLM 理解的空间语义 | 保留原 grid/M-RoPE/scatter；用固定状态配对比较有/无明确空间文本 |
| 视觉分布 | 原始导航训练主要是 256 全图；128 Glance 与局部 256 裁剪均可能改变视觉/上下文分布 | 比较 Glance、固定/跨相机裁剪、同裁剪低细节对照；不预设有益或有害 |
| 裁剪价值 | 固定或随机 ROI 不保证与指令相关，第一阶段闭环比较又混入状态差异 | 同 episode、同实际 pose、同像素、同扩散 seed 的单 chunk 受控比较 |
| 历史交互 | Glance 当前输入并不意味着全部历史也变成 Glance；旧合同仍是兼容 256 历史/pool4/BATS | 明确保持该合同，并做有/无历史干预，不将历史 token 与当前 token 混算 |
| STOP | 旧 learned_adaptive 只是脚本接口冒烟，不能当已学习的状态策略 | 独立感知 STOP=16，最多两 Focus；不替代原导航运动阈值 STOP |
| 动作反馈 | 预测的 waypoint 不是实际执行位移，碰撞会使两者不同 | 在 Habitat 真正执行后计算体坐标位移/误差，才更新跨步状态 |
| 坐标系 | worker pose 的 yaw 与模型 body_pose 相差 pi/2 | 使用 body_pose 计算反馈；实际 waypoint pose 显式转换 yaw |
| 碰撞 reward | 当前标准测量列表没有 collisions 项；自定义连续动作不保证更新 `previous_step_collided` | 临时包装实际 `sim.step_filter`，比较请求与过滤后位置，记录原生动作是否触发阻挡/滑移，不改外部实现 |
| Actor/Critic | Stage1 没有可训练完整状态条件策略 | 同一 Actor 核心支持五开关，Critic 独立参数；GRU 接收真实反馈而非 GT |
| PPO | 未实现旧 logprob、固定 GAE target、掩码和恢复 | 联合顺序 logprob 求和后计算 ratio；正确区分终止/截断，并保存 RNG/优化器 |
| 环境恢复 | Habitat 原接口未提供中途精确快照恢复 | 限制为已完成 rollout 的 episode 边界原子 checkpoint，不伪称支持中途续跑 |

## 5. 执行门禁与验证计划

1. 只新增 Stage2 实现、测试和报告，不改 Stage1 历史产物、冻结 checkpoint 或外部动作实现。
   已保存 `artifacts/protected_before.json`，交付时逐项复核。
2. 模型用 SimpleNav 的 `.venv/bin/python`，Habitat 用已验证的 `correctnav_hab033`；
   一份冻结模型驻留 GPU0，真实 Habitat 在 GPU1，独立本轮服务端口 10096。
3. 核心测试覆盖特征/空间、缓存复用、五开关、独立参数、GRU/实际反馈、GT 隔离、
   GAE 终止/截断、联合 PPO 比率、固定目标、参数变化及 checkpoint 精确恢复。
4. 诊断预先限定少量 train/dev episode 和起始/一 chunk 后固定状态；按同状态配对比较
   单 chunk 实际 geodesic progress。这只是潜在局部收益，不等于完整导航改进。
5. 小规模真训练分别验收 A 固定两 Focus 无历史、B 有真实跨步反馈、C 自适应 STOP。
   固定开发集检查与训练数据隔离，不依据两集结果调参，不启动全规模 PPO。
6. 吞吐与 GPU 成本使用真实观测/执行/更新计时，报告冷启动、环境重置、仪器开销及
   外推限制；不将合成单元测试吞吐冒充闭环训练吞吐。

## 6. 报告写入时状态

- 上述审阅已完成；新增核心模块和第一轮 18 项单元测试已通过。
- 真实冻结模型服务已启动；初始化 Actor 完成固定开发前两集闭环，失败 0、成功 1/2。
  该结果只证明接口可运行，不是训练改进，也不是原 G1 的重新评估。
- 此时真实 PPO 训练及 Focus 受控诊断仍未完成；本审阅报告不提前宣称通过。
- 后续结果以 `Experiment/SCOPE/Stage2/` 的八份任务交付报告和原始 artifacts 为准。

继续执行不需要另行等待用户确认；仍严格禁止全规模训练和 val_unseen 评估。

## 7. 后续交付入口

审阅完成后已继续执行，最终报告见
`../../../Experiment/SCOPE/Stage2/IMPLEMENTATION_REPORT.md`，实际状态与各项门禁结果
见同目录`artifacts/FINAL_VALIDATION.json`；本节仅添加交付链接，不改写第6节的当时状态。
