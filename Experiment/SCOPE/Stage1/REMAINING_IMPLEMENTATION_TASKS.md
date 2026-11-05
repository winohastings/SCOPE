# 剩余实现与预计执行成本

状态：预算草案，不是授权或已发生费用。除离线Task2审计/校验外，以下均未执行。
人力按熟悉当前代码的一名工程人员估算；GPU时间为规划区间，须在审批后train/dev pilot重新校准。

| 优先级 | 待完成任务与验收 | 人力估计 | GPU/机器时间估计 |
|---|---|---:|---:|
| P0 | 审批包与freeze manifest、显式门禁配置编译器；只允许G0–G4、缺审批/权重拒绝，保留global ordinal | 0.5–1天 | CPU <1h |
| P0 | G2循环调度接入现有fixed分支，保留Task1默认0/3；8步覆盖、episode reset、无重复测试 | 0.25–0.5天 | CPU <10min；dev小样本约0.2 GPUh |
| P0 | typed current feature bank、Glance/Focus顺序编码复用、instruction+feature selector接口，保持原scatter与grid | 1.5–3天 | 2–6 GPUh技术验收 |
| P0 | 独立actor/value、PPO joint-action训练器、resume/GT隔离/冻结参数审计、缺权重拒绝 | 2–4天 | 2–8 GPUh短train/dev验证 |
| P0 | 所有早停/异常路径计时、current/history/selector拆分、独立GPU、FLOPs coverage与显存采样 | 1–2天 | 2–6 GPUh profile/重放 |
| P0 | 全manifest执行器、failures分母、配对bootstrap/sign-flip/Holm、episode CSV与总成本 | 1–2天 | CPU 0.5–2h；dev小样本0.5–2 GPUh |
| P1 | 通过train/dev验收后正式selector训练三seed，每seed199680导航transition | 监督0.5–1天 | 单串行模型等效约140–220 GPUh，未含开发评估 |
| P1 | 每seed20次dev checkpoint检查×64集，平均规划30导航步/集 | 监督0.25–0.5天 | 约29 GPUh模型等效；若每集60步则翻倍 |
| P1 | R2R五组×三seed×1839完整episode | 监督0.5–1天 | 规划约28–166 GPUh模型+Habitat串行墙钟 |
| P1 | RxR泛化五组×三seed×3669完整episode | 监督0.5–1天 | 规划约138–991 GPUh串行墙钟 |
| P1 | 固定观测/三history层×五组×三重复×200 timed状态及warmup | 0.25–0.5天 | 约2–5 GPUh，依history与profile开销而变 |

## 估算依据与公式

Task1 R2R独立smoke的客户端闭环约0.68–0.82秒/导航步，峰值torch allocated约9.66–9.70GiB，
来自模型和Habitat同GPU、每模式仅4次预测，**只能作粗略容量参考，不是稳态基准**。
长历史、scene切换、顺序特征编码及selector训练会改变成本。

正式评估预算公式：`episode数 × 5组 × 3block × 平均导航步 × 秒/步 / 3600 × 1.2`。
预留20%给reset/scene与可恢复基础设施开销；低/高范围不是置信区间：
- R2R假设平均10–30步、0.6–1.0秒/步，约28–166h。
- RxR假设平均25–90步、0.6–1.2秒/步，约138–991h。
它们不是实测任务平均长度，不能引用为完成时长承诺。极端cap的规划上界还需单列：
R2R全部200步、1s/步约1839h（含20%），RxR全部500步、1.2s/步约11007h。
一次完整比较因此必须先审批时限与停止条件，不能默认“几小时跑完”。

训练导航transition总计599040（3seed）；按约0.7–1.1秒/步再加20%训练/更新overhead，
串行模型等效约140–220h。4个环境不意味着单模型4倍吞吐；吞吐需pilot实测。
dev检查20次（update20,40,...,380,390）每seed，共3840episode；以30步、0.9秒/步得约28.8h。
actor反向成本预计较小但未测；多环境渲染、模型调用和checkpoint I/O都必须计入训练总成本。

时间单位：上表对单模型部署给墙钟/模型GPU占用近似，若模型与Habitat各占一GPU，
两卡预约GPU-hours≈2×墙钟，而不是把一小时两卡写成1GPUh。可并行多个seed需额外模型卡和
模拟器卡，不能套用串行耗时又声称总GPU成本减半。
未计电价/云报价；本机资源成本以GPUh、CPUh、磁盘GB记账，不凭空给美元价格。

日志体积按 `步数×平均每步JSON字节数` 从pilot测量；禁存全量RGB，固定观测重放集单独压缩。
Task1实际日志不含完整稳态FLOPs或selector训练，不能拿其体积推断最终全量日志成本。

## 执行门禁与停止条件

所有P0验收通过、批准协议和预算并冻结代码/manifest后，才可训练/评估。
先用train/dev小样本测秒/步、最长episode、OOM/失败率和总日志体积，更新本预算并由审查者确认；
不得读取val_unseen结果来调整训练或预算优选组。
关键算子FLOPs未覆盖、出现原始输出漂移、G2/G3/G4不等预算、GT泄漏、主干梯度非空、
未解释的episode/seed差异，任一项均阻塞正式运行。
