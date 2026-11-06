# Task2.1 执行报告

日期：2026-11-06。范围：冻结导航接口上的视觉融合与短校准。

## 结论与门禁

Task2.1 本轮验收通过，最终依据为 `artifacts/release/validation.json` 的 `passed=true`。
已交付可复用 crop/geometry/adapter/runtime/calibration、配置、单测、真实校准 checkpoint。
没有开始 Task2.2、Actor/RL、大规模训练或 val_unseen 评估；不宣称导航性能提升。

开始时重新读取根 AGENTS、项目索引、docs 索引及用户指定的两份任务文件。
`agents/exec-agent.md` 仍不存在，因此按用户明确指定的严谨执行者职责与 Task2.1 要求执行；
没有自行补写或假称读取缺失角色文档。

## 代码审计与真实接口

| 项目 | 现有实现及审计结论 |
|---|---|
| 原始观测 | Habitat `vlnce031_runtime.py` 的 rgb/rgb_left/rgb_right/rgb_rear 对应 front/left/right/rear；原图 512×512、HFOV 90°、高度 1.25m |
| 相机坐标 | Habitat sensor yaw 为 0、+π/2、−π/2、π；不把它与旧 `Geometry.vector()` 的另一坐标约定混用。本模块只在各自图像像素坐标内计算 |
| Glance | `legacy_runtime/.../input_context.py:prepare_scope_images` 原图直接 bicubic 到 128；保持这一预处理，不从兼容的 256 图再次缩放 |
| 旧 Focus | Stage1/Task4 从原图取 256×256 象限；`ScopePolicy` 把 Focus 追加为图片块。旧当前 token 为 64+64N，而不是固定 64 |
| 新 Focus | 192×192 原图 ROI，bicubic 到 256×256 编码；不追加任何 Qwen 图片块 |
| Qwen | `navvla_qwen35_cpm.py:_encode_online_postmerge` 是接入点。沿用 `_build_qwen35_inputs`、`_fuse_qwen35_visual_tokens`、TVI、scatter、语言模型和动作头 |
| 当前特征缓存 | 旧 Task4 `CurrentFeatureBank` 按离散 Geometry 缓存，并拒绝重复候选；新连续几何不能冒充旧候选。新运行时复用原 forward 内已编码的 Glance，只额外编码选中的 Focus |
| BATS | 原 `LegacyRXRPolicy` 使用 BATS、1024 token 预算、每历史相机 4 token。当前修改不改变历史选择或历史池化 |
| 历史缓存 | 原 Qwen 提供 postmerge→4 token→BF16 bits 缓存记录，新 hook 不绕过该出口。实际 Legacy session 追加 RGB 观测，不自动把返回的缓存写回历史；本次实测的是在线历史分支，未宣称离线缓存命中加速 |
| checkpoint | Task4 `training.py` 仅允许 episode 边界保存，含 optimizer、RNG、cursor、provenance。此契约不变；新 Adapter 为独立 checkpoint，不改旧 Actor/Critic 权重 |

以下是**实际视觉编码张量**，不是通过配置猜测：

| 输入 | 原生 premerge grid_thw | 编码后张量 |
|---|---|---|
| 每个 Glance 128² | `(1,8,8)` | `(16,2560)`；四路堆叠 `(4,16,2560)` |
| 每个 Focus 256² | `(1,16,16)` | `(64,2560)` |
| 每个 teacher 全图 512² | `(1,32,32)` | `(256,2560)`，即 16×16 网格 |
| 历史 Qwen 上下文 | `(1,4,4)` | 每个历史相机 4 token；不是原始历史图片的编码 grid |

N=0..4 的真实导航调用均得到当前 64 token。六次无 Focus 抽查的总视觉 token 是
64、80、96（两个 episode 各三个历史长度），额外部分来自 BATS，不来自 Focus。
纯视觉编码输出量仍为当前 Glance 64 + Focus 64N，故固定 Qwen token **不代表**固定 ViT FLOPs。

## 几何与 Adapter

相机 ID 顺序严格固定为 front/left/right/rear。归一化 `(x,y)` 表示原始 512 图像上的
请求中心：`left=clip(floor(512*x-96+0.5),0,320)`，top 同理；right/bottom 加 192。
边界采用平移整个窗口而非补零或缩小 ROI。有效中心可能因此与请求中心不同，
`Focus.roi` 是裁剪与映射共同使用的唯一整数几何依据。

每张相机独立建立 4×4 的 128 像素目标格子。Focus 的 8×8 token 对应 ROI 内 24 像素
网格，用精确的二维矩形交面积将特征平均映射到有交集的目标单元，并产生面积覆盖比例。
不存在把四视角展开为伪 8×8 图像的操作。

门控残差：`M = G + sigmoid(gate_logit) * coverage * (F_aligned - G)`。
只有一个可训练标量，初始 gate=0.02；无 Focus 返回原 G 对象；其他相机、非覆盖单元
精确保留原 dtype/数值。顺序重复 Focus 使用前次输出作为当前网格，可累积，最多四次。
运行时保留 BF16，残差运算使用 FP32 后写回 BF16。

## 校准协议与实测

- 数据：由现有 `data_guard.episodes/check_episode` 校验 R2R train 划分并验证 annotation
  SHA256；8 个真实 Habitat 初始观测，每个 NPZ 单独保存 SHA256。没有加载 dev/heldout 数据。
- 校准 episode 1–6，诊断 episode 7–8；覆盖两个训练场景。诊断与校准 episode 不同，
  **场景并不独立**，因此仅为短校准健全性检查，不是跨场景泛化评测。
- 每个状态选四相机×四位置：中心、左上边界、右下边界、(0.37,0.63)。共 96 校准样本、
  32 诊断样本。每个样本只有一次 Focus；多次融合另做运行时及单测验证。
- 教师为相同冻结视觉编码器的 512² 全图输出。仅将与 ROI 相交的 teacher 网格面积
  映射到可见区域；目标为 `G+coverage*(T_aligned-G)`。其他单元的目标严格为 G。
- 只在覆盖单元计算特征误差，并显式加入非覆盖区 Glance 保持损失，权重 1.0。
  teacher 只出现在校准脚本，不存在于运行时 policy observation。
- 全图 teacher 本身具有全局感受野；ROI 监督掩码约束的是监督位置，不能宣称教师特征
  完全不包含 ROI 外上下文。未对 ROI 外单元施加恢复高清细节的目标。
- Adam LR=0.05，120 步、固定 seed=211；只优化 gate。没有按诊断集挑选 checkpoint。
  最终 gate=0.7967281342。校准后冻结，保存后重载逐参数精确相等。

下表为覆盖单元所有通道的 MSE；目标是上述部分可见区域混合目标，不是不可见全图细节：

| 数据 | Glance | 初始 gate | 校准 FP32 | 部署 BF16 |
|---|---:|---:|---:|---:|
| 校准 96 样本 | 0.005211812 | 0.005018723 | 0.001304358 | 0.001304455 |
| 诊断 32 样本 | 0.006026036 | 0.005791937 | 0.001231157 | 0.001231112 |

诊断集部署误差相对 Glance 下降约 79.57%。两组的非覆盖区最大绝对误差和 MSE 均为 **0**。
训练前后所有导航模型参数的 version counter 未变，且全部 requires_grad=False、grad=None；
因此 Qwen、视觉编码器、指令相关参数、SimpleNav 动作头均未参与校准优化。
导航 checkpoint 在模型加载前与既有冻结 SHA256 比对成功；没有写入该文件。

## 编码成本与导航一致性

GPU0 RTX4090，模型 BF16；每个 N 预热一次、同步计时五次。每次测量重新编码四个 Glance，
然后按运行时顺序逐个编码 Focus。总时间含 CPU 图像预处理、geometry 与 Adapter，
纯 ViT 时间来自视觉模块 forward hook；不含 Qwen/动作头、teacher 或 BATS 历史。

| Focus 数 | 纯 ViT 平均 ms | 编码+预处理+融合平均 ms | 当前 Qwen token |
|---|---:|---:|---:|
| 0 | 26.49 | 28.67 | 64 |
| 1 | 47.80 | 53.63 | 64 |
| 2 | 69.79 | 79.34 | 64 |
| 3 | 90.50 | 103.48 | 64 |
| 4 | 112.14 | 128.94 | 64 |

计时是该次短跑样本，不是跨设备吞吐保证。原始逐次计时保存在 validation.json。

无 Focus 抽查：episode 7、8 各三个输入历史长度；同权重、同图像、同指令、同种子，
关闭新融合与开启新融合 N=0 对比。原 `_encode_online_postmerge` 的 batch 调度不变；
六次当前特征最大差 **0**、预测动作最大差 **0**、上下文 grids 一致、导航 STOP 字段一致。
这里是同状态 RGB 回放并逐次增加历史，不是完整 Habitat 闭环 SR/SPL 对照。

另外实际调用 N=0..4 的导航推理，覆盖区内最大变化分别为
0、0.8125、1.5546875、2.0625、2.0625；每次非覆盖区最大差均为 0，证实不是只改形状
或永远不生效的 identity 实现。感知请求为空和导航 STOP 仍为不同概念。

## 测试、交付与限制

- Task2.1 新增 14 项单测：边界及舍入、相机身份、行列方向、面积守恒、只更新覆盖区、
  无 Focus 对象等价、重复累积、上限/形状/非法输入、teacher 掩码、冻结、BF16 保真、
  未校准运行时拒绝 Focus、拒绝旧追加块模式、包导入类型一致性。
- `test_scope*.py` 共 48 项通过。全量发现 53 项，52 项通过，1 项因既有
  `scripts.plan_perception_margin_eval` 缺失而导入失败（`test_perception_sampler`）；
  与本次改动无关，未擅自修复。完整结果见 `artifacts/logs/tests_all.log`。
- checkpoint：`artifacts/release/adapter.pt`，SHA256：
  `44174dc777ee21fbed5d7469d5cdcb9007f58c5531647708a0637bea05d21b51`。
- 运行目录不存在 Git 根；未提交、未创建分支。未改旧导航实现、Task4 源码或历史权重。
- `FILES_MANIFEST.tsv` 列出此次项目内源码、配置、文档、测试及全部持久化 artifacts；
  不包含解释器自动生成的 `__pycache__` 与 manifest 自身。
- 缺失角色文件、短校准场景重叠、静态导航一致性抽查和缓存分支覆盖范围均已明示，
  不扩张为完整导航性能结论。下一任务只能复用新融合入口，不能继续使用旧追加 token 路径。

完整执行命令与 Python 调用见 README.md。
