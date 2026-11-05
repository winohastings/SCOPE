# 成本测量协议

状态：待审、待实现稳态测量器。本次没有生成新的模型FLOPs或稳态性能数字。
Task1/R2R每组4次预测的小样本是功能smoke，不是充分预热后的成本结论。

## 1. 平台与隔离

冻结GPU型号/UUID、驱动、CUDA、PyTorch、Transformers、NumPy、Habitat、attention backend、
BF16、batch size1、CPU线程与CPU型号。规划使用一张4090作模型、另一张作Habitat，
模型卡上无其他进程。每次测试记录 `nvidia-smi` 快照、温度、功率与时钟状态；
无权限锁频就如实报告，不声称已经锁频。禁止模型/模拟器同卡的旧smoke数字与本协议稳态数字混合。

模型加载、首次权重读取、CUDA context、首次kernel初始化为cold-start独立统计，不纳入steady-state。
不在每个测量步清空allocator缓存；仅在测量区间开始前reset peak memory计数。
在确定性设置中记录实际可启用项，遇不支持kernel时报错/登记，不能默默退回另一attention实现。

## 2. 测量集合与可复现预热

### A. 固定观测模型microbenchmark（解耦轨迹差异）

审批后从训练来源dev的固定64集采集20集，每集首10个G0参考状态，共200状态。
以dev episode选择顺序的前20集作为来源；状态不足10时循环已有状态但记录重复，
不得按耗时或“视觉有用”程度挑选。轨迹采集只用于cost重放，记录其成本，不用于heldout调参。
每状态保存raw四图、指令、pose、真实history选择与token长度、frame count和固定导航step。
五组使用相同状态，且按真实step运行cyclic/random/learned。已存raw512不意味着要把render延迟算作0：
该microbenchmark明确是模型-only，渲染单测于B。

为区分当前块成本与历史合同，用三个明确分层：history=0、4、8个观察帧（每帧四相机）。
来自同一状态的history按最近帧截取；不足所需历史的状态只进入可满足层，绝不补造pose。
若层内<200状态，则在train/dev继续采集到200，缺层时标“未完成”，不能复制历史伪装真实缓存。
各层实际history mask/所选frame ID必须跨组一致。另保留在线BATS自然history重放作补充。

每组每层启动新session/恢复同一重放状态，先20个forward预热；随后按固定状态序列计量200次。
三个独立process重复（run seed42/43/44），每次都重新加载并预热。组顺序采用主协议轮转。
预热使用train/dev中的固定状态，与正式timed状态分离；预热不推进正式重放session。
记录全量预热时间而非只说“已warm up”。

预热后将最后10次与前10次中位数比较，若漂移>5%，继续10步，最多100步；
未稳定则标platform_unstable，不发布steady-state speedup。先20步是最小值，不是可随结果缩减的参数。

### B. 渲染与模拟器成本

在同一20个dev episode上，固定512四相机、相同状态/动作记录，以Habitat GPU独立执行。
每scene加载时间与首次reset独立统计；每scene最少20次观测调用预热，随后记录200次观测调用。
使用重复的确定状态观测调用时记录其重放特性，不声称是200个独立导航transition。
计时 `get_sensor_observations`/`get_observations_at` 的同步接口边界，去重嵌套调用。
它包含观测采集/拷贝等开销，标名 `render_observation_wall_ms`，非纯光栅化GPU kernel延迟。
step物理/碰撞耗时另计 `sim_step_wall_ms`，若无法拆分则报告combined并注明，禁止双重相加。
不得对不同组改渲染尺寸、相机数量或只渲染选中ROI：本阶段512四图渲染是共同成本。

### C. 真实闭环总成本

每个正式任务/block开始在独立train/dev session预热到稳定，然后reset所有model/Habitat session，
再开始锁定manifest。不得把heldout episode的前20步删掉作为预热；首步与短episode也进入分母。
所有episode记录模型、渲染、物理、RPC、CPU预处理及日志成本；episode结束前的停止预测亦计。
session reset、scene load、模型cold start分别列，episode inclusive总成本还应给出包含reset/scene的口径。

## 3. 计时边界与插桩字段

CUDA异步计算：阶段GPU时间用同stream CUDA events与synchronize后读取；
阶段墙钟使用perf_counter配合明确边界同步；同时记录，不能将二者混为一列。
正式少插桩端到端与细粒度profile分开重放，避免逐模块同步扭曲吞吐。

最低字段：
- `vision_current_ms`、`vision_history_ms`、`selector_first_ms`、`selector_second_ms`、
  `current_feature_reuse_ms`、`llm_ms`、`dit_b_ms`、`preprocess_ms`、`model_wall_ms`。
- `render_observation_wall_ms`、`sim_step_wall_ms`、`rpc_wall_ms`、`artifact_io_ms`、
  `policy_cycle_wall_ms`、`episode_wall_inclusive_ms`、cold/reset/scene时间。
- 当前块与历史块逐图raw size/native grid、premerge patch数、postmerge token、LLM context视觉数；
  实际encoder invocation count、编码image count、duplicate-encode标志、cached/recomputed来源。
- 每步预测与实际waypoint、Focus ID/ROI/顺序、选择器hash、seed、全局ordinal。

`model_wall_ms`从服务收到已解包请求后到动作和元数据就绪，必须含两次selector及所需特征计算。
`policy_cycle_wall_ms`从客户端本次prepare/predict前到8-waypoint执行返回，不包含下一个cycle；
完整 `episode_wall_inclusive_ms`从reset前到最终结果写盘完成。不能用所有部分相加假设无重叠，
重叠或同步时序需记录timeline，优先报告直接测量的墙钟总时间。

当前Task1 `timings={}`发生在选Focus之后，**新selector耗时可能落在server_total里却不在
vision_ms/llm_ms**，甚至新增特征提取可能被清空。因此后续应把阶段计时初始化放到请求最前，
新增专用selector计时，而不是重写视觉pipeline。`stop=True`早停分支、异常预测都须落成本日志；
只在动作执行后生成step文件会遗漏这些路径，需对应补测。

## 4. FLOPs定义与覆盖率

约定一次乘加=2 FLOPs，报告per-policy-step视觉encoder **GFLOPs**，包含patch embedding、
各attention/MLP、merger与所有本步重算的历史图。分别列current/history/selector/total；
LLM/DiT-B本任务至少计延迟，若另外计FLOPs也声明coverage，不与视觉列混淆。

Qwen本机vision config：patch16、merge2、hidden1024、depth24、MLP4096、输出2560。
128/256单图premerge为64/256，postmerge为16/64。FLOPs不能按postmerge token简单线性缩放，
更不能把64个历史context token当作只编码64个patch。

后续计数应追踪实际visual.forward图形状与算子：linear/matmul/bmm/einsum、patch conv、
SDPA/attention与merger等，登记unsupported ops、动态分支及所有调用次数。
对融合attention注册可审查的解析计数：每层attention主项约4×N²×D，QKV/输出linear约8×N×D²，
MLP按实际gate/层数与矩阵形状逐项计；各图N分开计算，不把多图拼接总N当跨图全注意力。
RoPE、激活、norm/softmax若未计，应列excluded_elementwise，不把部分计数称全模型精确FLOPs。
确认实际block/packing后再使用公式，不能只凭配置替代真实forward计数。

可用torch.profiler的 `with_flops=True` 作为交叉检查，但其支持算子有限，
不能保证对融合attention等完整计数。必须导出coverage表、算子形状与reference matmul测试，
若关键算子未覆盖则标incomplete，禁止填0或按token比例推估成实测。
在独立profile pass重放，profile pass延迟不混入正式timed pass。

## 5. 显存与汇总

torch峰值同时给allocated/reserved bytes；按测量segment reset峰值，记录模型驻留baseline与增量。
device-level显存另用NVML/驱动采样（至少20Hz，记录采样周期与可能漏峰），不冒充精确kernel峰值。
model GPU与Habitat GPU分别报告，不能只报torch allocation而声称整个系统峰值。
共享checkpoint内存不因组不同重复加载两个模型；selector权重/activation、feature bank计入G4。

每层/组给N、warmup实际次数、mean、p50/p90/p95、population描述性std、三process重复均值与
样本std，显存给最大值与每process峰值。不要对吞吐和延迟互取倒数而不注明batch/并发。
闭环还给每episode总模型/渲染/端到端成本，按统计协议生成配对cluster CI。
计数若发现重复当前图编码，先阻塞等计算预算效率结论，但保留实测原始成本，不只修CSV。
