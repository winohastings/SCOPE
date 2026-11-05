# SCOPE Stage2 核心架构

## 范围与继承

本轮依据 Task4 实现，而非改写已完成的第一阶段。审阅报告位于
`../../../docs/Logs/Task3/STAGE2_REVIEW_REPORT.md`。指定上游为 winohastings/SCOPE，
锁定提交 `e52fe5780aea3c076d5226d285c15a1e357068ff`；发布清单中本地存在的 629 个
SCOPE 文件 SHA256 全部一致。详见 `artifacts/upstream_provenance.json`。

保留 Task1 原生像素预处理、视觉编码器、image grids、M-RoPE、TVI、原始 scatter、
Qwen3.5 和 DiT-B；仅在当前特征编码入口插入缓存复用。冻结模型 4,728,515,844 参数，
Actor 578,497 参数，独立 Critic 573,569 参数。未实现另一套一次性 Selector。

## 每个导航步的数据流

1. 四相机原始 512×512 → 四个 128×128 Glance，编码一次，得到 `[4,16,2560]`。
2. 同一冻结 Qwen 文本 embedding 对真实指令编码，最多首128+尾128 token，masked/有效
   token 均值，得到 `[2560]`。不需要目标位置、距离或成功标签。
3. Actor 根据 Glance、语言、当前已获取 Focus、上一个真实动作反馈和跨步状态，依次选
   ROI 或感知 STOP。最多获取两块 Focus，每次仅编码所选 256×256 crop → `[64,2560]`。
4. 当前特征 bank 将已编码块提供给原 `_encode_online_postmerge` 的输出位置；历史仍按
   Task1 的兼容 256/pool4/BATS 合同编码，不将当前 bank 假装成历史缓存。
5. 原 `_fuse_qwen35_visual_tokens`、原 M-RoPE/scatter、LLM 和 DiT-B 预测八个 waypoint。
6. Habitat 按原动作协议实际执行，返回真实位姿、动作观测和过滤响应，才提交反馈并更新
   State GRU。导航 STOP 由原运动阈值/环境终止判据决定，不由感知 STOP 决定。

## 当前特征缓存

`CurrentFeature` 是带类型记录：观测 SHA256、encoder/checkpoint hash、预处理版本、
camera/ROI/resolution、native grid、dtype 和 frozen tokens。观测 key 含精确 body pose
及四相机实际像素，bank 仅存在于当前请求，不跨不同位姿或 episode 复用。

- Glance grid `(1,8,8)`；Focus grid `(1,16,16)`。
- 相机顺序 front/left/right/rear，body-FRD yaw 为 0/-pi/2/+pi/2/pi。
- HFOV=90°，相机高1.25m；ROI 为512原图中的精确整数象限坐标。
- 几何对象提供归一化 ROI、相机方向和每个 merged token 中心射线。
- `read_focus` 拒绝读取未选择 ROI；重复选取拒绝；Actor 输入不包含候选高分辨率特征。
- 原采集层会生成 PIL 裁剪候选，但这不等于编码所有 Focus，Actor 从不取得全部候选图。
- 正常导航严格断言：当前图编码数=4+已选Focus数，复用数完全相等。额外重编码只允许
   明确标记的独立诊断，不能计入正常训练/导航吞吐。

## 同一个 Actor 的五开关

`ActorConfig` 包含 `use_language`、`use_focus_memory`、`use_navigation_state`、
`use_action_feedback`、`enable_perception_stop`，所有组合共享 `ROIActor`。

Glance 4×4 token 网格每个2×2象限均值构成16个低分辨率候选 key。全局 Glance、语言、
顺序 Focus GRU 记忆、跨步 State GRU 组成 query；候选 key 加相机/ROI 几何投影。
有效动作掩码禁止重复 ROI。STOP=16，导航最多两 Focus；达到预算后环境自动结束感知，
不额外伪造一个无随机性的 PPO 决策。

Focus GRU 在单导航步内按获取顺序递增；State GRU 跨导航步，仅在执行反馈提交后推进。
Actor 和 Critic 分别拥有自己的 encoder/GRU/heads/optimizer，无共享 Parameter。
当前采用长度1截断反向传播：历史隐状态 detach，但上一真实反馈的 GRU 转换在 PPO
重放时重新计算，从而 State GRU 本身可获得梯度。不是长序列完整 BPTT。

## 可选空间融合

`SpatialAdapter` 为8维相机/ROI几何→2560维的零初始化残差投影。默认关闭；显式提供
`server.py --spatial-adapter` 的 state_dict 才启用。零初始化逐位恒等，包括 BF16。
本轮没有训练适配器，不能把空间文本干预结果当作已学习 adapter 的收益。

## 数值一致性的重要限制

真实检查发现：像素完全相同，但 BF16 的旧“当前+历史一次大batch”与新顺序调度可产生
约百分之几的特征差异。因此不能声称缓存输出与旧任意批处理逐位相同。正确的缓存一致性
基准是同样的“四Glance batch→Focus1→Focus2”调度：该路径特征与像素要求逐位相等，
且同调度原生编码输出经原 scatter 后动作也单独比较。旧 batch 差异作为诊断证据保留，
不通过放宽误差门限隐藏。最终值见 `artifacts/release/AUDIT.json`。
