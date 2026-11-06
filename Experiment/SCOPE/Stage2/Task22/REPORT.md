# Stage2 Task2.2 执行报告

日期：2026-11-06。执行范围：Spatial Query Actor、增量主动注视闭环、可重放轨迹、真实
Habitat 导航推理。**不进行正式 RL 训练，不更新任何权重，不用随机导航指标证明方法有效。**

## 最终结论及入口

Task2.2 本轮验收通过。最终实测依据：

- `artifacts/verified_validation/VALIDATION.json`：`passed=true`。
- `artifacts/final_runtime/actor_initial.pt`：2,293,073 参数的完整 Actor 初始化权重。
- `artifacts/final_runtime/traces/`：30 个真实导航 transition 的完整感知轨迹。
- `artifacts/final_sampling/results.json`、`artifacts/final_deterministic/results.json`：实际环境执行结果。
- `artifacts/final_runtime/episode_boundary_*.pt`：4 次 episode 运行的边界 cursor、RNG、轨迹顺序和冻结审计。
- 代码/命令：`README.md`；数据/概率/重放边界：`TRACE_SCHEMA.md`；完整文件哈希：`FILES_MANIFEST.tsv`。

最终服务已关闭，未结束其他用户进程。未启动 Task2.3 或 PPO/Critic 正式优化。

## 规范和计划依据

开始时按执行者职责读取根 AGENTS.md、index.md、docs/index.md、Stage2 固定约束和
用户指定的 Task2.2 文件，并按需读取 `Plans/AGM_method.md` 中 Actor 与闭环定义。
仓库没有名为 Stage2 AGM Execution Plan 的独立文档；Task2.2 文档自身标题为
“Task 2 — AGM Actor 与完整主动注视闭环”，故结合该文档与 AGM 方法计划执行，不猜测
额外未提供任务。`agents/exec-agent.md` 仍缺失，已在执行中说明，没有假称读取。

## 实现与复用边界

| 文件 | 交付或修改 |
|---|---|
| `../Task21/incremental.py` | 新增 IncrementalPerception：逐次 acquire、融合状态、精确像素并集 coverage、坐标历史、四次上限与关闭状态 |
| `../Task21/runtime.py` | 原 FusionPolicy 增加 controller 回调；沿用原 Glance 编码 batch，不重编码当前图、不增加图片块；固定 Focus 接口也改用同一增量引擎 |
| `actor.py` | 两层空间 Transformer、Fourier 位置、相机 embedding、G/M/L/B 条件 Query cross-attention、分层 STOP/camera/position 分布 |
| `trajectory.py` | 决策状态及 latent/probability 保存、原子落盘、概率重放、episode 顺序加载、显式 next_state/bootstrap、原 Task4 Transition/GAE 桥接 |
| `actor_runtime.py` | 复用 Task21 FusionPolicy；直接复用 Task4 Stage2Policy.language 和 ExecutionFeedback，加载冻结 Adapter、Actor 权重、闭环反馈与冻结审计 |
| `server.py` | 复用 SimpleNav session.v1、msgpack、reset/predict/feedback/finish/close 契约，显式模式配置，无训练更新端点 |
| `run_habitat.py` | 复用原 Task4 navigate、FeedbackModel/FeedbackBackend、worker、Habitat backend、训练场景划分、episode manifest |
| `../real_rollout.py` | 仅改反馈 token 数读取：优先响应显式 current_visual_tokens；旧 Task4 无该字段时仍用原 64+64N 行为 |
| `validate.py` | 重载权重、真实轨迹概率/梯度重放、同状态 0..4 Focus 对照、无 Focus BATS 对照、两模式复现与冻结审计 |
| `config.json` | 固定初始架构、动作范围、最大 4 Focus、200 导航 transition 上限、已有 Adapter checkpoint、禁止 RL 标记 |
| 项目根 `tests/test_scope_task22.py` | 19 项新测试，包含 CUDA inference/gradient 路径及跨设备轨迹回放 |

没有更换旧语言编码、导航网络、视觉编码器、环境动作转换、训练场景划分或冻结
Adapter checkpoint。旧 Task4 的 17 类离散 Actor 和追加 Focus token 路径没有被调用。
仅导入旧 Stage2Policy 的语言方法，不实例化其旧 Actor/Critic/CurrentFeatureBank。

## Actor 与几何细节

G、M 分别是当前步原 Glance 和逐次融合网格，形状均为 `(4,16,2560)`；L 复用冻结
Qwen token embedding 均值（原 Task4 256-token 首尾截断规则）。视觉与语言投影
2560→192，G/M 使用共享视觉投影和独立融合映射；四相机 embedding、各自 4×4 的
Fourier 2D 坐标和 coverage 编码后，经过两层 Transformer（4 heads，FFN 768）。
指令与 Focus 历史生成 query，经 cross-attention 读取 64 个视觉位置。

STOP 192→128→1，初始 p=0.25；camera 192→128→4；position 选中相机条件
384→192→128→4，输出 mu 与 log_std 修正，另有每相机二维 learnable log_std。
初始 mu=0、latent std=0.8，标准差限制 0.1–1.5。实际参数 **2,293,073**，不是估计数。

坐标严格为 `96+320*sigmoid(u)`；除以 512 后进入 Task2.1 原 Focus，裁剪调用处统一
舍入到 192×192 ROI，再缩放为 256 编码。center、latent 和实际 ROI 一并保存。
四视角从未拼成伪 8×8 图；camera ID 0/1/2/3 始终对应 front/left/right/rear。

每次 Focus 编码后立即调用冻结 Adapter 并更新 M、coverage、history、count，再次
调用 Actor。允许重叠和重复 Focus，coverage 是实际像素并集而非简单相加。第四次
Focus 后直接 budget 终止，不添加训练 STOP probability；从 j=0 即允许零 Focus。

## 轨迹、训练衔接与不可假设事项

每个 decision 保存当时 G/L/M/coverage/history/count、camera、latent u、中心、ROI、
STOP、旧联合 logprob、旧分布参数、STOP/camera/latent-position/joint entropy、时间
和 transition ID。序列总 logprob 为实际 decisions 求和，旧值 detach，原精度 CPU 保存。
环境 feedback、终止/截断、oracle reward 数据与 ActorState 分开存储。

使用 latent 空间 Gaussian 概率，不用舍入后的坐标反解 u，也不混入像素 Jacobian。
确定性路径标记 `ppo_eligible=false`。采样路径也必须有真实反馈才可转入 PPO transition。
原 Task4 `Transition`、`gae_targets`、`FixedTargets` 可直接接收桥接结果；新分层
`joint_log_probability` 代替旧离散 evaluator。旧 PPOTrainer.update 的 GRU/17类假设
仍不兼容，不假称无需改训练循环即可调用；Task2.3 应复用本模块 evaluator。

完整 episode loader 按边界文件中的顺序连接 next_state，末端截断要求真实后继图像
bootstrap；不存在默认置零的截断 value。bootstrap 单独四图编码，与含历史的在线
batch 调度差异已明确记录。trace 不等同 Habitat snapshot；不支持假定可从任意
特征恢复中途环境，不支持从缓存推断未观察过的新 ROI。完整协议见 TRACE_SCHEMA。

## 真实 Habitat 闭环

同一冻结 backbone/Adapter/Actor，R2R train episode 1、2（场景 7y3sRwLe3Va）。
GPU0 RTX4090 运行模型，GPU1 RTX4090 运行 Habitat。每种模式两集，最大 8 个导航步。
使用原 SimpleNav action block、实际环境执行、碰撞探测、BATS action observation
历史和后继状态，非静态像素回放。

| 模式 | episode | 完成 transition | 实际终止 | 路径长度 |
|---|---|---:|---|---:|
| training-sampling | 1 | 6 | waypoint_motion_stop | 4.400 |
| training-sampling | 2 | 8 | max_steps，保存 bootstrap | 8.683 |
| deterministic | 1 | 8 | max_steps，保存 bootstrap | 6.590 |
| deterministic | 2 | 8 | max_steps，保存 bootstrap | 7.647 |

共 **30 个真实 transition**、4 份边界记录。导航动作已实际推进环境；其中存在感知
policy_stop 后仍执行导航的 transition，证明感知 STOP 不混同导航终止。没有基础设施
失败被包装为奖励样本。三份步数截断均明确保存 bootstrap。

这些是未训练 Actor 的短闭环检查，不是导航性能实验。原始 results 包含 SR/nDTW 等
环境指标，但不以其随机波动作为效果验收或 checkpoint 选择依据。

### 分支与熵监测

| 模式 | N=0 | N=1 | N=2 | N=3 | N=4 |
|---|---:|---:|---:|---:|---:|
| training-sampling（14步） | 7 | 1 | 0 | 2 | 4 |
| deterministic（16步） | 0 | 0 | 0 | 0 | 16 |

采样小样本没有自然出现 N=2，不把该样本缺口隐瞒为完整分布覆盖：N=2 在真实观测
下的受控导航调用与单测中单独覆盖。确定性初始化 p_stop=.25，所以按阈值必然继续
到上限；这不是 STOP 死分支。STOP 采样与梯度测试正常。

两模式合计 camera 0/1/2/3 的 Focus 次数为 **12/66/2/7**；所有四个相机分支均出现。
均值熵（nats）：STOP **0.562335**，camera **1.364139**，latent position **2.391590**，
joint latent **3.379131**。确定性 argmax 会偏向初始化占优相机，不宣称均匀或最优。

## 核心验收数字

- 30 条真实轨迹在权重重载后，no-grad 与可反向梯度重放的旧 sequence logprob 最大
  误差均为 **9.5367431640625e-7**，低于预先采用的 1e-5 门禁。
- 两个真实初始观测、每个三种历史长度，共六次无 Focus 对照：token 最大差 **0**、
  导航动作最大差 **0**、上下文 grid 相同、导航 STOP 字段相同。
- 两个初始观测各做 N=0..4 干预，所有当前视觉 token 为 **64**。N>0 时融合特征与
  下游 SimpleNav 动作均发生非零变化；不预设这些变化改善导航。
- 相同状态/种子，两模式各重复两次，动作差与序列 logprob 差均 **0**；采样复现使用
  seed=223，实际执行四次 Focus，避免只验证立即 STOP 的退化情形。
- Actor reload 参数逐项相等；真实闭环与验证结束时 Actor 未更新、backbone version
  counter 未变、backbone requires_grad=False/grad=None、Adapter 冻结且逐参数相同。
  验证脚本只做 Actor 诊断 backward，不执行 optimizer.step，优化次数为 **0**。

同状态 N=1/2/3/4 的下游动作最大绝对变化：

| 初始观测 | N=1 | N=2 | N=3 | N=4 |
|---|---:|---:|---:|---:|
| episode 1 | 0.038720 | 0.031916 | 0.107916 | 0.126402 |
| episode 2 | 0.021571 | 0.024181 | 0.038437 | 0.024103 |

实际动作值以最终 VALIDATION.json 为准；这些数值仅证明感知已连通导航决策。

## 测试及失败透明记录

新增 19 项测试；`test_scope*.py` 共 **67 项通过**，包括 Task2.1 的全部回归和 CUDA
inference/gradient 重放、CPU/CUDA 轨迹转换。全量发现 72 项，71 项通过，1 项因
既有 `scripts.plan_perception_margin_eval` 缺失导入失败，未修改无关模块。

保留以下尝试，不与最终成功产物混用：

1. `artifacts/runtime/`、`artifacts/sampling/`：首次真实调用发现旧 ScopePolicy.mode
   覆盖 Actor 采样模式。分离为 actor_mode 后修复；失败 episode 无有效训练 transition。
2. `artifacts/release_runtime/`、`release_sampling/`、`release_deterministic/` 和
   `artifacts/validation/`：闭环已完成，但启用梯度后的 logprob 最大误差约 2.098e-5，
   超过门禁。原因是 PyTorch inference fused Transformer 与训练路径不同；实现
   ReplayStableSpatialLayer 统一算子路径，重跑真实闭环，而非放宽 probability 阈值。
3. `artifacts/final_validation/`：新增 CPU episode loader 后，float64 sigmoid 在
   CPU/CUDA 的极小差别导致 center 严格逐位相等检查失败。改为 1e-10 像素容差，
   同时 ROI 整数仍要求完全一致；latent、旧概率不改。补充跨设备测试后通过。
4. 最终成功：`final_runtime/`、`final_sampling/`、`final_deterministic/`、
   `verified_validation/`。模型服务日志出现的无关 HTTP 探测 WebSocket 握手失败不计
   导航结果；真实客户端请求及最终 episode failure 均为空。

Task2.1 的历史报告与旧 FILES_MANIFEST 不重写；本次允许修改的 runtime 源码哈希已
记录于 Task2.2 manifest，原冻结 Adapter 文件未变。项目根不存在 Git 仓库，未提交、
未创建分支。所有新增/修改项目文件及持久化 artifacts 都列入本次 FILES_MANIFEST，
排除 Python 自动缓存和 manifest 自身。

## 后续门禁

Task2.3 可使用当前完整 Actor、增量感知和可重放数据协议，独立实现 Critic/GA-PPO
训练逻辑；应保持旧概率固定、按导航 transition 处理 advantage、明确 group 分支环境
重放及 bootstrap 调度，不能复用旧追加 Focus token 路径或旧离散概率函数。
本轮通过不自动授权正式训练预算或 heldout 评估。
