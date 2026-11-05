# SCOPE 第一阶段实验协议

版本：stage1-v0.1，日期2026-11-05，状态 **DRAFT_REVIEW_REQUIRED / 未审查、未冻结**。
这是供审查的完整候选方案，文件存在、配置校验通过都**不等于获得实验授权**。
本次仅审计、只读枚举数据、编写协议及运行单元测试，不启动策略训练或完整评估。

## 1. 研究问题与成功条件

在冻结SimpleNav、保持相同导航和历史契约的条件下，当前步给定相同的两个Focus预算，
根据指令和观测顺序选择ROI，是否比固定循环或随机选择改善R2R导航质量？

确认性比较：G4−G2、G4−G3的episode级nDTW差；统计检验与多重比较见
`METRICS_AND_STATISTICS.md`。两比较均须显著且方向为正，同时SR非劣性检查通过，
才称“学习选择在等预算下有益”。如果只优于G1或仅耗时不同，不满足该结论。
成本下降与原始G0质量保持是另行检验的主张，不由token下降自动推导。

第一阶段不研究何时STOP或动态分配预算；每步G2/G3/G4均恰好两个不同ROI。
导航原有STOP规则保留，它与被禁用的感知STOP不是同一件事。

## 2. 已有实现审计与可执行边界

详见 `IMPLEMENTATION_AUDIT.md` 与 `docs/Logs/Task2/`。Task1已验证原始输出一致、
原生16/64-token块、2560维特征、scatter/M-RoPE和逐步轨迹/成本记录。
但目前的 `fixed_2`只取front.R1/R4，脚本learned没有训练；特征条件选择与编码复用、
FLOPs完备计数、正式统计器及审批门禁尚未实现。

不重写视觉采集流水线；后续只在既有selector接口、当前块编码复用和统计/执行层补齐。
Task1基线代码、权重、已有日志及产物保持原样。本次配置是协议schema，不能直接当eval.py配置执行。

## 3. 冻结模型与观测合同

- R2R与RxR按官方指南共用RXR step-8000 checkpoint，不需要独立R2R权重。
  SHA256和路径在 `STAGE1_GROUP_CONFIGS/common.yaml`；全部视觉、LLM、TVI、DiT-B参数冻结，
  仅新增独立selector/value参数可训练，不使用LoRA或主干微调。
- 四相机front/left/right/rear统一512×512渲染，方向、高度1.25m、HFOV90°一致。
  所有ROI均在原512图上定义；不得先缩小再裁剪。
- G0输入相同512图缩放出的256兼容图，属于**统一渲染的原始模型路径**。
  不将它宣称为Habitat原生256渲染的逐像素官方复现；Task1旧服务等价性只比较相同有效输入。
  原生256基线可单列校准结果，不纳入G0–G4确认性组，也不能据此调参。
- G1是“仅当前Glance”，不是全程只读低分辨率：所有组沿用Task1的共同256历史图→pool4。
  历史BATS配置、pool4、TVI、动作后采样频率均不变；不能用当前token节省擅自增加历史帧。
  实际选中历史帧可随轨迹变化，不能要求不同策略的历史图内容相同。
- 当前图和历史图的实际encoder工作量分开计数；历史上下文4-token不意味着encoder仅算4个token。
- 不对未选择ROI调用视觉encoder；CPU网格crop不等于可让策略偷看未选Focus特征。

## 4. 实验组

| 组 | 当前视觉块 | Focus选择 | 当前合并后token |
|---|---|---|---:|
| G0 | 四个256兼容视图 | 无 | 256 |
| G1 | 四个128 Glance | 无 | 64 |
| G2 | 四Glance＋两个256 Focus | Fixed-Cyclic-2 | 192 |
| G3 | 四Glance＋两个256 Focus | 无放回Random-2 | 192 |
| G4 | 四Glance＋两个256 Focus | 真正训练的顺序Learned-2 | 192 |

总视觉上下文还包含实际历史，必须另列。G2/G3/G4必须具有相同历史规则、编码调度和embedding复用，
不能让G4额外重复编码一次Glance再以相同192-token标签掩盖其额外成本。

### Fixed-2的评估与确定性循环方案

front.R1/R4可作为旧固定位置消融，但长期忽略另外三相机和front的R2/R3，不能单独代表
有全域覆盖的固定预算基线。没有实测证明它“性能差”；本结论只是覆盖性审计。
保留旧实现，不将它重新命名为循环基线。

相机ID为0–3，ROI的R1/R2/R3/R4为0–3，候选ID=4×camera_id+region_id。
设固定次序
`O=[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]`，导航步从0开始。
第t步取 `O[(2t)%16]`、`O[(2t+1)%16]`；每集重置t=0，固定phase=0。
前八步为 `(0,4),(8,12),(1,5),(9,13),(2,6),(10,14),(3,7),(11,15)`。
每步不同相机、恰好两个不同ROI，连续八步覆盖全部16个ROI，之后重复。
短于八步的episode不保证全覆盖，不通过额外采集补齐预算。
这是**提出并经数学校验的方案，尚未接入现有fixed_2运行分支**；后续接线须增加测试。

## 5. 训练与评估数据隔离

只用R2R `train`训练selector；主干本来是RXR训练模型，因此“泛化”明确指selector
没有在RxR上训练/微调，不声称整个导航系统从未见过RxR训练数据。

训练scene按SHA256(scene_id)升序、scene_id作tie-break，最后ceil(10%)作为dev。
已物化为54个训练scene/7个dev scene，9837/982条episode；dev池按
SHA256(`909:episode_id`)升序取前64条。训练/开发scene与两个val_unseen均无交集。
scene和episode列表、来源annotation哈希均在 `STAGE1_GROUP_CONFIGS/manifests/`。
主干原始RXR训练可能含这些R2R训练scene，不承诺主干层面的新数据隔离。

主基准：R2R-CE val_unseen，**1839集、11scene**，200个policy step与现有Habitat
episode step cap=200不变。泛化：RxR val_unseen guide English en-US/en-IN，
**3669集、11scene**，policy/Habitat cap均为500。
这里只读枚举annotation，不启动Habitat；当前RxR本机配置默认50集，正式编译必须覆盖为全量manifest。

所有组使用相同的有序manifest。保留annotation顺序与全局ordinal，禁止随机挑集、
按scene难度筛掉失败集，禁止用val_unseen选模型、选循环phase、选reward或学习率。
Task1两集只做过技术验收，已经不是全新的盲测样本；必须披露，后续不得再据其结果调参。

## 6. 随机性与执行合同

三个配对run block：导航seed 42/43/44，G4分别使用训练seed101/202/303的一份dev选定权重。
每个训练seed只训练一次并按固定dev规则选权重，不挑最好的seed。
G0–G3也分别跑这三个导航seed；G4不是偷偷取九种组合中的最好结果。

继承当前runner：session seed = run seed + **完整manifest全局ordinal**；
每次预测的diffusion seed继续沿用session seed＋累计观测数。
Random-2使用Task1的SHA256(session_seed:episode_id:navigation_step)→前8byte little-endian
→NumPy default_rng无放回选择；固定NumPy版本。selector采样独立generator，不消耗diffusion RNG。
G4评估用两次masked argmax，tie选最小ROI ID；不采样、无STOP、无重复。
G2不依赖任何随机源。不同组轨迹不同造成后续观测数不同不称为seed错配。

固定一worker，模型GPU1、Habitat GPU0；新实验前停止冲突服务或改资源映射并重新登记，
禁止为了省时把正式成本测试与别的模型服务同卡并行。不同机器/驱动/内核需单列平台。
跨worker扩展须先证明全局ordinal和seed不因分片变化，否则继续单worker。

每block使用固定拉丁轮转顺序：42为G0,G1,G2,G3,G4；43为G2,G3,G4,G0,G1；
44为G4,G0,G1,G2,G3，降低整组顺序偏差，不在看到结果后更改。

## 7. 导航与指标锁定

动作规格、反归一化key、8-waypoint、左右/朝向符号、碰撞控制、相机几何完全沿用
`common.yaml`及Task1既有入口；导航STOP仍是相邻waypoint平均平移≤0.03。
不开启到成功半径就自动停止；NE与成功判定沿用当前euclidean evaluator，不静默切为geodesic。
训练reward可使用训练环境geodesic teacher，仅用于训练，不改变eval。
GT nDTW仍用现有稠密locations和轨迹连续重复位置去重。
保留原始metric产物；对异常结果的最终分母合同见统计协议，不能只引用跳过failure的现有summary。

## 8. 审查、冻结与分阶段执行门禁

本候选协议、配置、manifest与代码审计哈希形成审查包；请审查者明确确认：组定义、
共同高分辨率历史解释、G0渲染口径、统计假设、预算/训练量、资源和成本。
审批前 `training_authorized=false`、`full_evaluation_authorized=false`；不能自行改为true。

审批后仍须按 `REMAINING_IMPLEMENTATION_TASKS.md` 补齐代码、训练/评估入口门禁、
train/dev小样本验证、FLOPs计数与稳态测试，然后生成带审批记录的freeze manifest：
协议/配置/数据/代码/依赖/模型/selector的哈希、reviewer、时间、资源预算、允许命令。
任何空审批或缺权重时正式运行入口应fail closed。当前既有eval.py不理解本schema，
**这项门禁尚未实现**，不能依赖配置中的false自动阻止旧入口。

审批后先进行train/dev上的实现验收与成本pilot，再训练selector，锁定三份权重；
R2R先做G0–G4配对实验，达到预算批准后才进行同权重RxR泛化。
若必须改协议，先登记新版本及原因，再运行；不可看完heldout结果后回填“预注册”。

## 9. 交付物

`STAGE1_GROUP_CONFIGS/`、`SELECTOR_DESIGN.md`、`METRICS_AND_STATISTICS.md`、
`COST_MEASUREMENT_PROTOCOL.md`、`REMAINING_IMPLEMENTATION_TASKS.md`与本文件组成完整协议。
`IMPLEMENTATION_AUDIT.md`列证据与缺口；`validate_protocol.py`只做离线校验，
既不训练也不评估；Task2执行日志位于 `docs/Logs/Task2/`。
