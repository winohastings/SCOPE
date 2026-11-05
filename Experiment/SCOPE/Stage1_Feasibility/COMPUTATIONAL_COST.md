# 计算成本

## 真实闭环测量

CUDA events为设备流时间；CPU enqueue不冒充同步墙钟。模型墙钟与闭环墙钟直接测量。

| group | model_wall_ms_p50 | model_wall_ms_p90 | policy_cycle_wall_ms_p50 | policy_cycle_wall_ms_p90 | vision_combined_ms_p50 | vision_combined_ms_p90 | history_visual_tokens_p50 |
|---|---|---|---|---|---|---|---|
| G0 | 1596.7160 | 1623.9003 | 2700.5704 | 3293.8817 | 416.6781 | 421.4970 | 368.0000 |
| G1 | 1332.4699 | 1573.4018 | 2596.4786 | 3201.6079 | 351.5225 | 415.1421 | 368.0000 |
| G2 | 1325.0084 | 1359.6166 | 2379.4630 | 2768.2337 | 353.3873 | 358.1123 | 368.0000 |
| G3 | 1326.6571 | 1359.3921 | 2349.8937 | 2763.0244 | 353.6279 | 358.3041 | 368.0000 |

| group | peak_gpu_memory_bytes_max | peak_reserved_bytes_max | render_observation_wall_ms_p50 | render_observation_wall_ms_p90 |
|---|---|---|---|---|
| G0 | 11193114112 | 11882463232 | 53.8691 | 90.5866 |
| G1 | 11153553920 | 11882463232 | 54.2932 | 90.7564 |
| G2 | 11180339712 | 11882463232 | 55.3297 | 91.5246 |
| G3 | 11180339712 | 11882463232 | 55.3528 | 89.6771 |

policy_cycle优先读取既有benchmark worker从prepare前到waypoint返回的直接墙钟；RPC到返回另列，不能混用。


## 同状态独立拆分重放（非闭环计时）

| group | samples | profile_vision_current_ms_p50 | profile_vision_current_ms_p90 | profile_vision_history_ms_p50 | profile_vision_history_ms_p90 |
|---|---|---|---|---|---|
| G0 | 8 | 43.7217 | 44.0613 | 398.1594 | 400.1252 |
| G1 | 21 | 43.5524 | 44.0748 | 398.0882 | 401.7326 |
| G2 | 9 | 48.7107 | 48.9783 | 403.0229 | 404.6365 |
| G3 | 9 | 48.4547 | 48.9226 | 402.7382 | 404.8302 |

真实在线执行为当前/历史混合batch，保持不变；动作生成后独立拆分重放以测量两个部分，二者不可相加冒充原混合batch成本。四组完整episode逐步动作等价性见COST_VALIDATION.json。

独立512四相机固定状态渲染：20次预热、200次timed，观测接口墙钟p50=3.092ms、p90=4.618ms，含读回，不是纯GPU光栅kernel。原始数据见RENDER_PILOT.json。


## 固定开发观测试点（history=0）

| group | warmup | timed | model_wall_ms_p50 | model_wall_ms_p90 | rpc_wall_ms_p50 |
|---|---|---|---|---|---|
| G0 | 20 | 100 | 631.9866 | 639.8205 | 644.1490 |
| G1 | 20 | 100 | 587.3548 | 590.6985 | 598.8562 |
| G2 | 20 | 100 | 635.1837 | 640.8753 | 646.4180 |
| G3 | 20 | 100 | 635.8151 | 639.6623 | 646.9088 |

同一dev首帧、相同指令/pose、每次重置session；每组至少20次独立预热并检查≤5%漂移，100次timed。不是任务2三历史层、三进程完整microbenchmark，不能推广到长历史稳态。预热不删除正式episode首步。

## FLOPs与显存边界

每次真实visual.forward的grid、输入shape和current/history来源保存在逐步CSV；用VISION_ARCHITECTURE.json实际24层模块尺寸估计patch Conv3d、QKV/输出Linear、两层MLP、attention及merger。一次乘加=2 FLOPs；attention按每图/每时间帧的4×N²D，非跨图全注意力。未计norm、softmax、激活、RoPE、索引/拷贝，状态incomplete；这些是明确形状估计，不是完整实测FLOPs。不按合并token比例推断算力；历史context pool4不代表只编码4个patch。

GPU驱动以请求50ms间隔采样，可能漏峰；gpu_samples.csv含设备总占用（包括桌面）。正式评估两环境worker共享单模型，RPC/闭环含排队及并发竞争，不能称单请求独占延迟；独立profile/固定观测试点无并发排队。四组使用相同并发设置。torch allocated/reserved为模型进程峰值，不含Habitat。模拟器reset/scene合并报告，物理非渲染量为差值估计，不把组件之和当直接测量的闭环总耗时；日志IO与reset在逐episode CSV另列。

检测到3512次current/history混合视觉调用；若非零，相应分项设备时间不能解释为纯当前/纯历史，需要独立profile。
在线纯current/history分项的缺失值保留为空；各项CSV附_n，不能把只在history=0时测得的纯current分位数当全轨迹成本。


| gpu | peak_device_mib | interval_p50_ms | interval_p90_ms | occupied_device_hours_estimate | utilization_weighted_device_hours_estimate |
|---|---|---|---|---|---|
| 0 | 12103.0000 | 52.0000 | 52.0000 | 1.7274 | 0.4017 |
| 1 | 2495.0000 | 52.0000 | 52.0000 | 1.6361 | 0.0503 |