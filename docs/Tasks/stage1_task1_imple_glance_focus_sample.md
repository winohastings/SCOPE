# 任务 1：实现 SCOPE 的多分辨率 Glance–Focus 视觉采集路径

## 目标

扩展现有的 SimpleNav-RXR Qwen3.5-4B 推理流水线，以支持多分辨率、四视图 Glance 和高分辨率局部 Focus 裁剪，同时保持与原始 SimpleNav 检查点和评估协议的兼容性。

## 要求

### 1. 保留原始观测

- 将 Habitat 的四个 RGB 传感器配置为渲染 512×512 图像。
- 在任何缩放之前保留原始 512×512 图像。
- 保留现有的 256×256 SimpleNav 兼容图像路径。
- 不要在 Focus 裁剪之前静默缩放原始观测。

### 2. Glance

- 从原始观测生成四张 128×128 的 Glance 图像。
- 确保 Qwen3.5 实际以 128×128 处理它们。
- 验证每张 Glance 图像产生 16 个合并后视觉 token。

### 3. Focus

- 将每个 512×512 视图划分为四个 256×256 候选区域,也即每个相机画面划分为：
\[
\begin{bmatrix}
R_1&R_2\\
R_3&R_4
\end{bmatrix}
\]。
- 跨四个相机构建 16 个候选。
- 支持确定性的 Fixed-grid 和带种子的 Random 选择。
- 支持顺序策略选择，每个导航步最多两个 Focus 裁剪。
- 为自适应策略添加 STOP 动作。

### 4. Qwen3.5 接口

检查并扩展：
- starVLA/model/modules/vlm/QWen3_5.py
- starVLA/model/modules/qwen35_vision.py
- starVLA/model/modules/input_context.py

要求：
- 支持可变分辨率视觉块。
- 保留每张图像的 image_grid_thw 和 M-RoPE 信息。
- 支持 16-token Glance 块和 64-token Focus 块。
- 保留 2560 维合并后特征接口。
- 通过现有多模态 scatter 路径插入视觉嵌入。
- 保留原始历史缓存契约和检查点加载。

为每个 Focus 块添加相机 ID、ROI 坐标和 focus 索引元数据。

### 5. 统一评估模式

实现共享的 policy_mode 接口：

- original
- glance_only
- fixed_2
- random_2
- learned_2
- learned_adaptive

所有模式必须使用相同的 episode、相机几何、动作归一化、waypoint 执行协议和随机种子。

### 6. 插桩

记录每步：
- 视觉 token 数量
- Focus 数量和 ROI 坐标
- 视觉编码器延迟
- LLM 延迟
- DiT-B 延迟
- Habitat 渲染延迟
- 总延迟
- 峰值 GPU 内存
- 预测和实际 waypoint 位姿

保存原始 episode 结果并生成适合论文表格的聚合 CSV。

### 7. 验证

在完整评估之前：

1. 在不做修改的情况下运行原始 SimpleNav-RXR / SimpleNav-R2R 检查点。
2. 在固定观测上验证原始输出等价性。
3. 验证 128×128 Glance 每视图产生 16 个 token。
4. 验证 256×256 Focus 每个裁剪产生 64 个 token。
5. 验证两个 Focus 裁剪总共产生 192 个当前观测视觉 token。
6. 验证所有六种模式在相同的两 episode Habitat 冒烟测试上运行。



本任务中不要启动策略训练。不要修改原始检查点。不要静默更改原始导航动作协议。

实际执行时找到现有能启动SimpleNav R2RCE RXRCE的现有入口。

生成实现报告，列出修改的文件、张量形状、token 数量、兼容性检查和实测推理成本。
执行Logs保存至docs/Logs/Task1
