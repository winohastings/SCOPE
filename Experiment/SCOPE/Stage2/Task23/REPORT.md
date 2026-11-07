# Stage2 Task2.3 执行报告

## 当前状态

执行日期：2026-11-06至2026-11-07。角色：严谨执行者（Exec）；仓库`agents/`为空，
`agents/exec-agent.md`不存在，因此按用户要求、根AGENTS和Stage2任务文档执行。

**4096真实主轨迹预算和全部周期验证已完成；最终策略出现确定性零Focus塌缩，
不能宣称已得到性能稳定、有效主动注视的论文基线。**
工程交付与预算完成事实见`artifacts/ppo4096/COMPLETE.json`、`summary/SUMMARY.json`。
模型代码已实现真实导航级PPO-only；不加入Group Sampling/RLOO/GA-PPO。

## 实现与固定协议

- Actor直接使用Task2.2 SpatialQueryActor与`evaluate`，参数2,293,073。
- Critic独立874,113参数：hidden128、一层空间Transformer、语言attention pooling、
  MLP；只估计每导航步Focus前长期价值。不修改任何冻结导航或视觉参数。
- Task2.1 Adapter SHA256：`44174dc777ee21fbed5d7469d5cdcb9007f58c5531647708a0637bea05d21b51`。
- 当前64视觉token、192×192 ROI、sigmoid-Gaussian latent和BATS历史接口全部沿用已交付实现。
- `learning.py`实现导航reward、GAE(.99,.95)、terminal零bootstrap、truncation真实后继
  Glance bootstrap、逐决策联合概率ratio、序列内求和/导航状态平均，以及独立优化器。
- 64+主轨迹transition后于episode边界更新，2 epochs，minibatch16导航状态，
  Actor/Critic LR1e-4/2e-4、clip.2、gradclip1；KL>.05停止该轮余下minibatch。
- 训练清单9837条授权train；固定32集scene-balanced val_seen周期验证；不使用val_unseen。
  56个训练/验证涉及的场景文件均已确认存在，实际布局为`mp3d/mp3d/scene/scene.glb`。
- seed233，正式预算4096；预先冻结的奖励和选模规则见`PROTOCOL.md`。
  完成4096不等同10k–20k正式扩展预算、全量778集val_seen评估或多seed论文结论。

## 真实更新和恢复门禁

通过产物：`artifacts/gate_rng_isolated/`。

| 项目 | 真实测量 |
| --- | --- |
| Habitat主轨迹 | 5集、64 transitions；不计正式4096 |
| 逻辑PPO更新 | 1轮、8次optimizer minibatch step |
| Actor参数L2变化 | 0.40277270757781797 |
| Critic参数L2变化 | 0.7027067261104973 |
| 初始ratio最大误差 | 0.0 |
| STOP/camera/position/value head梯度 | 均非零；详见TRAINING_GATE.json |
| Backbone/Adapter/old snapshots | 全部冻结审计通过 |
| 完整checkpoint恢复 | 参数、优化器、Python/NumPy/Torch/CUDA/minibatch RNG及cursor逐位一致 |
| 独立新进程恢复 | 另起CUDA进程读取更新前checkpoint，重建64条真实buffer并更新，与原结果逐位一致 |
| 轨迹审计 | 全部64条真实feedback、reward公式、唯一ID、64token通过 |
| Focus次数0/1/2/3/4 | 17/8/6/9/24；平均2.234375 |
| 相机0/1/2/3 | 38/47/34/24 |

恢复测试重复同一64条数据的更新，**不是新增64个环境transitions**。
门禁SR=.4、SPL=.24793仅为5集训练数据的工程测量，不是正式性能结果。
门禁GPU采样能耗约8.42Wh、峰值GPU0 12219MiB；包含重复更新，不并入正式训练成本。

## 失败尝试与修复

失败目录全部保留，不计入正式训练有效预算，不以修复后的结果覆盖历史证据。

| 目录 | 问题与处理 |
| --- | --- |
| `artifacts/gate` | 指令首尾空白造成manifest校验误拒绝；只在一致性比较时strip，不修改模型输入 |
| `artifacts/gate_verified` | RPC直方图整数key被msgpack strict_map_key拒绝；改为字符串key |
| `artifacts/gate_release` | Habitat场景加载触发WebSocket默认心跳超时；延长客户端等待 |
| `artifacts/gate_final` | 64真实步和参数更新成功，但恢复审计发现server自动ping消耗Python全局RNG |
| `artifacts/gate_rng_isolated` | server禁用自动ping，消除通信对训练RNG干扰，真实门禁通过 |

另有一次端口TIME_WAIT导致输出创建前启动失败，probe使用SO_REUSEADDR修复；
不占用或终止其他监听者。曾误假设场景少一层目录，随后检查真实嵌套路径确认无缺失。
模型环境没有pytest，使用已有unittest测试，不为本任务安装额外依赖。

## 验证范围与限制

### 正式训练结果

正式运行：`artifacts/ppo4096/`。276集train、4096 transitions、56轮更新、570次
Actor optimizer.step和570次Critic optimizer.step。buffer最终为空。
另有5轮×32集val_seen，共2452个验证transitions，**不计入4096**。
全部6548条trace逐文件核验唯一ID、真实feedback、64当前token、奖励公式；
train有217次真terminal、59次truncation，截断均保存有效后继state/旧Critic bootstrap。

| 已训练transitions | 固定val_seen SR | SPL | 每transition平均Focus |
| --- | --- | --- | --- |
| 0 | 34.375% | 28.9439% | 4.0000 |
| 1033（最佳） | 56.250% | 42.1852% | 4.0000 |
| 2054 | 43.750% | 37.1141% | 4.0000 |
| 3100 | 37.500% | 28.6767% | 0.2744 |
| 4096（最终） | 34.375% | 26.0498% | 0.0000 |

最佳权重`runtime/checkpoints/best_val_seen.pt`按预声明SR/SPL规则来自update14；
原评估点完整权重为`update_0014.pt`。最终完整恢复点`latest.pt`。
最佳点相较初始化SR增加21.875个百分点、SPL增加13.2413个百分点，但后续回落；
32集单seed选择噪声大，不能把最佳点当作全量或跨域有效性的证明。

训练集合计SR=33.3333%、SPL=25.5930%，每episode平均奖励0.302461；
这是训练期间不同策略的混合统计，不是最终模型评测。
训练Focus 0/1/2/3/4分别1724/940/541/297/594次，平均1.291260；
相机0/1/2/3分别479/3038/1437/335次。详细按policy_version直方图见SUMMARY。

### 数值稳定性与行为退化

- 56轮Actor/Critic参数变化均非零，初始ratio最大误差全部0，所有执行loss/gradient/value
  有限；冻结审计全通过。实际minibatch最大KL=0.0285608，KL早停0次。
- 首轮最后minibatch Critic loss=.243577，末轮最后minibatch=.041059；这些不是同一
  样本上的泛化误差，不能直接据此认定价值学习已收敛。末轮explained variance=-.001268。
- 最后minibatch STOP/camera/latent-position熵为.682207/.522300/1.801071 nats。
  STOP采样熵仍高不等于确定性策略有用：deterministic阈值越过0.5后可整体偏向STOP。
- 3100步验证448/481状态零Focus，最终498/498状态零Focus。训练仍有0–4次Focus，
  因此是**确定性部署策略塌缩**，不是动作空间或梯度路径被代码禁用。
- 奖励归因：train progress合计+55.829144、Focus成本-52.89、步成本-40.96、
  terminal bonus+121.5，总计83.479144。最终验证奖励均值高于初始化，但SR不增、
  SPL下降，说明节省感知成本可改善标量奖励却不保证导航改进。
- Task2.1仅在两个训练场景做短校准，跨场景收益未被证明；Task2.2同状态N=1..4
  已证明下游动作有非零变化（最大约.1264），只证明连通性，不证明变化有益。
  现有证据支持检查感知成本与长期导航信用分配、确定性阈值、相机集中和Adapter
  泛化；不能仅凭本次相关性确定唯一原因。本任务未改奖励、重校准Adapter或追加实验掩盖退化。

### 成本与退出异常

主运行训练+验证约16959.84秒（4.711小时），其中train episode约10212.94秒，
val_seen约6266.90秒，PPO更新合计277.33秒。GPU0/1实测峰值12766/5970MiB，
5秒整卡功率采样积分约544.80/168.55Wh，合计713.34Wh；含空闲/桌面，不是进程独占能耗。
门禁、失败尝试、独立恢复审计成本另存，不并入这次正式run统计。

完成全部验证、冻结审计、COMPLETE和latest落盘后，Habitat退出阶段出现
`GL::Context::current(): no current context`，进程exit134。保留完整`driver.log`，
**不声称进程干净退出**。模型server已退出，无遗留训练进程。
之后在新trainer加载最终checkpoint，四套模型及两套optimizer逐位相等，计数4096/570、
buffer空、cursor validation_round=5且next_validation_at=5120；见`FINAL_CHECKPOINT_AUDIT.json`。
所有6548个完整trace亦通过独立审计，故退出清理异常不抹消已完成预算，但仍属运行限制。

另发现PROGRESS/COMPLETE的cursor曾取save前status，原始JSON显示验证游标落后一轮；
真实latest中的游标正确。已修复后续run的状态报告顺序，**未重写本次原始产物**，
恢复以checkpoint和FINAL_CHECKPOINT_AUDIT为准。训练数据、优化和选模不受此展示缺陷影响。

相关SCOPE测试78项全部通过，包含新增非单位ratio clipping、truncation隔离和fresh trainer恢复。
独立新进程实测证据见`artifacts/gate_rng_isolated/FRESH_PROCESS_RESTORE.json`。
最终全仓库发现83个测试条目，其中82项通过、1项为既有导入失败：
`tests/test_perception_sampler.py`依赖不存在的`scripts.plan_perception_margin_eval`。
未修改无关文件，也不声称全仓库测试全部通过。

正式运行必须同时报告初始化/最终/最佳固定val_seen SR和SPL、训练奖励、Focus/相机、
entropy/KL/Critic loss、原始更新曲线与成本。若结果未改善，不能宣称方法有效；
需要结合奖励组成、早停分布、Task2.1校准限制及Task2.2动作敏感性证据分析。
完整恢复状态引用原episode边界文件，保留原运行目录才可恢复未更新buffer。

## 文件与复现

组件职责、逐条执行命令见`README.md`，数据/恢复字段见`TRACE_SCHEMA.md`。
冻结训练配置为`config.json`；参数及数据来源哈希见每次运行的`runtime/training_contract.json`。
源码清单与SHA256见`SOURCE_MANIFEST.json`；全部门禁/失败/正式运行产物见
`artifacts/ARTIFACT_MANIFEST.json`，包括checkpoint、trace、曲线和日志，不含清单自身。
