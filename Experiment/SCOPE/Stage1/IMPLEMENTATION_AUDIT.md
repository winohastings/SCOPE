# Task1实现与证据审计

日期2026-11-05；范围仅Stage1相关代码、配置与指定产物。不是全仓库审计，未重新启动模型。
历史数据校验与新运行严格区分。执行角色文件缺失，按用户执行者授权完成本次任务。

## 1. 已验证证据

| 审计项 | 当前代码/产物 | 结论与边界 |
|---|---|---|
| 权重/原始兼容性 | Task1 `fixed_equivalence.json`、`fixed_habitat_equivalence.json`及R2R副本 | 历史合成/真实固定图连续两步动作 `[8,4]`，最大差0；本次只读检查，不冒充新推理 |
| 共享checkpoint | SimpleNav官方本地 `docs/guides/VLNCE_TRAINING_AND_EVALUATION.md` | RXR step-8000同时评测R2R/RxR是官方口径，不缺R2R独立权重 |
| raw/compat/Glance/Focus | `legacy_runtime/starVLA/model/modules/input_context.py` | raw512先copy，256兼容、128Glance，四相机各四256原图ROI，共16候选 |
| 原生token/特征 | `qwen35_vision.py`、`QWen3_5.py`、`framework/VLM4A/navvla_qwen35_cpm.py`；Task1实际scope JSON | grid `[1,8,8]`→16、`[1,16,16]`→64；hidden2560；scope禁resize；使用既有scatter与M-RoPE |
| Fixed/Random/顺序 | `input_context.py` 与 `tests/test_scope_sampling.py` | Fixed=0/3；Random SHA256/NumPy无放回；顺序selector最多2次，已选列表递增，缺selector拒绝 |
| 历史合同 | `scope_server.py::_observation`与历史构建、BATS配置 | 当前传入history为256兼容图，按原合同pool4；Glance变化只应用当前sample；不能说全部历史都是128 |
| 每步成本/轨迹 | 外部 `benchmark/common/runner/worker.py`、Habitat backend/runtime；Task1 `scope_*.json` | 预测/实际位姿、token/ROI、vision/LLM/DiT、采集、总耗时、峰值allocated均有；不足以当稳态报告 |
| 已有冒烟 | Task1 `smoke_verification.json`、R2R同名文件 | 12任务/模式组×2集及独立R2R6组×2集；两步上限，不是完整split性能 |
| 已有完整基线 | Task1 `before_baselines/`和旧baseline日志 | RXR权重在两任务各2集完整闭环，R2R已有数据/GT/nDTW适配，无需重写 |
| 入口/选择 | `run_smoke.py`、现有R2R/RxR eval及input adapter | smoke固定每模式2集；可选task；R2R正式默认全量，RxR本机config默认50，必须正式编译覆盖 |

本次只读枚举确认为：R2R val_unseen1839集/11scene，RxR English guide3669集/11scene。
manifest保留adapter原始顺序与ordinal，不只保存两条smoke episode。
R2R train只做annotation划分：54/7scene、9837/982episode，dev固定64，与heldout scene无交集。

## 2. 实现缺口（不得静默掩盖）

1. **循环Fixed缺口**：现有fixed只取front.R1/R4；本轮提出8步16候选循环并离线校验，未改旧代码。
2. **G4不是现成实现**：现接口缺instruction、编码Glance/Focus、权重/训练状态；脚本选择器不是learned。
   候选列表预先crop图像不代表允许actor访问全部高分辨率内容；新接口应只提供已获取Focus特征。
3. **顺序复用缺口**：现有先选图再一次batched视觉编码；真正特征条件G4必须先编码Glance，
   再逐Focus编码且复用到导航，否则重复计算。G2/G3需同调度以免不等计算预算。
4. **历史成本与认知边界**：G1仍可从共同256历史得到信息；Task1保留历史合同不是纯Glance全轨迹。
   固定历史不改是本候选协议的明确选择，需审查是否符合论文主张。
5. **计时缺口**：timer初始化在select_focus后；新增selector feature encoding可能被清空分项计时。
   缺独立selector、current/history视觉、GPU events、reserved/NVML峰值及日志I/O分解。
   worker的显式 `response.stop=True`路径没有现成scope日志（motion-stop路径有），需后续覆盖。
6. **FLOPs缺口**：Task1只有token/时间，没有完整视觉算子计数；不能以token比代替GFLOPs实测。
7. **种子与分母风险**：session seed是base＋episode global ordinal，不是所有episode都固定42；
   分片重排会变seed。现有metrics排除failure，正式paired统计须另实现分母合同。
8. **正式执行与审批缺口**：现run_smoke只支持两集聚合、包括六模式且有adaptive，不能直接作为
   五组Stage1正式运行器。声明式YAML也不是eval原生配置，需要门禁编译器，禁止仅改名运行。
9. **环境时限**：既有R2R policy cap200/Habitat cap200、RxR500/500均保留；
   单policy step执行8waypoint可能先触Habitat cap，须原样披露而非擅自改为8倍cap。
10. **有限证据**：固定两步输出/单元测试不是所有历史长度、所有分辨率组合或全量episode的兼容证明。

## 3. 本轮安全动作与修改范围

仅新建 `Experiment/SCOPE/Stage1/` 的协议/配置/候选manifest/离线校验器，增加协议单元测试，
更新文档索引，新增 `docs/Logs/Task2/` 审计日志。未改Task1采集/模型/旧Fixed/Random分支，
未改外部SimpleNav仓库、动作协议、检查点或Task1既有产物；未启动训练、完整评估或新服务。

`validate_protocol.py`检查五组预算、八步循环、授权false、manifest来源哈希、train/dev隔离、
训练transition预算一致性与历史等价性/smoke记录，输出候选包哈希。
它不是模型forward测试，也不是审批证书。模型实时再次等价性、完整GFLOPs和G4需待后续P0实现。
