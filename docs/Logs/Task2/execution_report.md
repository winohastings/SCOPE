# Task2：第一阶段实验设计与协议审计交付

日期：2026-11-05。角色：严谨执行者（Exec Agent）。
任务：`docs/Tasks/stage1_task2_verify_adapt_vision_useful.md`。
已读取根AGENTS、项目/文档索引、项目协议、Task1必要代码、R2R报告与对应实测产物。
`agents/`仍为空，缺失 `agents/exec-agent.md`，本次按用户明确任务执行，未伪造角色要求。

## 交付结论

已产出可供审查的第一阶段协议包，位于 `Experiment/SCOPE/Stage1/`。
状态为 **DRAFT_VALIDATED_NOT_APPROVED**：候选方案完成、离线校验通过，
但**未审查、未冻结、未授权训练或完整评估**。本次未加载模型、启动新服务或执行完整episode评测。
五组配置只是协议声明，不能直接传给现有eval.py；执行门禁编译器仍是后续任务。

| 要求交付物 | 文件/目录 | 本轮交付内容 |
|---|---|---|
| 第一阶段协议 | `Experiment/SCOPE/Stage1/STAGE1_EXPERIMENT_PROTOCOL.md` | 组定义、研究假设、共享权重/历史/动作、数据隔离、seed、审批冻结门禁 |
| 分组配置 | `Experiment/SCOPE/Stage1/STAGE1_GROUP_CONFIGS/` | G0–G4、common、两个benchmark、训练配置、配置说明和3份候选manifest |
| 选择器设计 | `Experiment/SCOPE/Stage1/SELECTOR_DESIGN.md` | 轻量pointer、指令/Glance/首个Focus条件、两次不同ROI、独立PPO actor/value、reward与复现 |
| 指标统计 | `Experiment/SCOPE/Stage1/METRICS_AND_STATISTICS.md` | episode分母、失败处理、三seed配对、cluster bootstrap、scene sign-flip、Holm及SR安全性 |
| 成本协议 | `Experiment/SCOPE/Stage1/COST_MEASUREMENT_PROTOCOL.md` | 预热/稳定检查、当前与历史/selector成本、完整FLOPs覆盖、模型/渲染分离及显存口径 |
| 剩余任务及成本 | `Experiment/SCOPE/Stage1/REMAINING_IMPLEMENTATION_TASKS.md` | P0/P1验收、人力/训练/评测GPU时间范围、上界及估算假设 |
| 实现审计 | `Experiment/SCOPE/Stage1/IMPLEMENTATION_AUDIT.md` | 已有证据与实现缺口分开，避免脚本选择器/两步smoke被当正式结果 |

## 关键设计决定

- G0原模型256兼容输入，G1仅当前四Glance，G2固定循环，G3随机两Focus，G4学习顺序两Focus。
  第一阶段不含感知STOP或adaptive；原有导航STOP保持不变。
- G2/G3/G4每步恰好两个不同ROI，当前视觉token均192。G0/G1分别256/64，历史另计。
- 现有front.R1/R4只能算狭窄固定位置消融；提出跨相机8步遍历16个ROI的循环：
  `(0,4),(8,12),(1,5),(9,13),(2,6),(10,14),(3,7),(11,15)`。
  每集从第0对开始、phase0固定。本次只提出并校验，不修改Task1旧Fixed分支。
- R2R/RxR共用官方RXR step-8000权重。统一512渲染；G0不是原生256渲染逐像素复现。
- 所有组共同保留Task1的256历史图→pool4，G1不能被描述为“全轨迹只看低分辨率”。
- G4需编码后Glance、instruction、已选Focus及camera/ROI embedding。现有脚本不满足，
  当前selector API也没这些完整输入，必须后续增加特征接口和当前块编码复用。
- R2R作为主基准、RxR作为selector不微调泛化；已物化候选列表1839/3669集，分别11scene。
  RxR旧配置默认50集是正式运行风险，协议编译必须覆盖为全量manifest。
- 只用R2R train训练selector：54scene/9837episode；开发7scene/982episode，固定开发64集。
  与两个val_unseen scene无交集。主干原先在RXR训练，不声称整个系统对RxR训练数据零接触。

## 审计证据及新校验

`implementation_evidence.json`：读取Task1来源哈希，对持久化运行时238个文件逐一SHA256复核，
**全部仍与Task1记录一致**；逐一核对独立R2R六模式原始scope日志的grid/token、8-waypoint和计时。
同时登记Task1全部文件哈希用于保护性复核，本次没有改Task1产物。

`protocol_validation.json`：五组schema/预算、八步覆盖/不重复、授权false、
eval manifest来源哈希/global ordinal、训练/开发/评估scene隔离、训练量一致性，
以及历史固定观测两步0差异证据和smoke通过记录均检查通过。
其中历史兼容性是**读取并验证已有证据**，不是本轮新的GPU推理。
校验包内保存候选协议文件哈希，不等同于最终审批freeze manifest。

| 本轮检查 | 结果 |
|---|---|
| Stage1协议新增测试 | 4项通过：循环覆盖、重复候选拒绝、同相机pair拒绝、候选包保持未批准 |
| Task1 SCOPE回归 | 9项通过，未修改旧测试或旧实现 |
| Python编译 | 新校验器及测试通过 |
| 新训练/完整评估/服务 | 均未启动 |

初次离线校验中，校验器把dev_episode_count读取到错误YAML层级，导致1项测试报错。
已修正为checkpoint_selection.dev_episode_count并重新运行全部新增测试，最终通过。
初次失败日志保留为 `*_initial_schema_error.log`，没有改配置或删测试掩盖问题。

## 剩余阻塞项

1. 审查者批准组/共同历史/G0口径、统计假设、资源预算后，才可冻结协议；当前授权保持false。
2. 循环G2接线、G4独立训练器与指令/特征接口、顺序编码重用、门禁配置编译器。
3. 稳态测量/FLOPs覆盖、selector与history计时拆分、所有早停/失败日志、正式配对统计器。
4. 审批后先train/dev验收与成本pilot，复核粗估预算，再按批准范围训练/评估。

预算量级：三seed训练599040导航transition，约140–220串行模型GPUh，另加开发检查；
R2R正式五组×三seed约28–166h、RxR约138–991h串行墙钟，均为基于明确步数/秒每步假设的
规划范围，不是实测或执行承诺；两卡预约GPUh约为墙钟两倍。详见预算文档的上界和误差说明。

## 修改/新建文件与执行命令

本轮新增Python：`Experiment/SCOPE/Stage1/validate_protocol.py`、`tests/test_stage1_protocol.py`。
新增协议、YAML/manifest及本目录审计日志；仅修改 `docs/index.md` 添加入口。
逐文件完整列表见 `files_manifest.txt`，保护性哈希见 `implementation_evidence.json`。

以下命令只做离线校验和单元测试，不会启动训练、模型或Habitat：

```bash
/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage1/validate_protocol.py \
  --output docs/Logs/Task2/protocol_validation.json

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_stage1_*.py'

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope_*.py'
```
