## 创新点一：Incremental active gazing Module (AGM)

### 方法核心
#### 确定设计配置
设计项	                推荐确定方案
- Focus Patch：	    原图 512×512，裁剪 192×192，编码输入 256×256
- 动作空间：         感知 STOP + 四相机 Categorical + 二维连续位置
- Actor：           192 hidden、两层空间 Transformer、Cross-Attention、分层输出，预计约 2–3M 参数
- Critic：	        独立的轻量空间编码器、128 hidden、约 1M 参数
- Visual Adapter：	几何对齐、局部特征写回、门控残差，保持下游 64 个当前视觉 Token
- 优化算法：     	 同状态 Group-Relative Actor Warm-up → 导航级 PPO/GAE，Critic 只在后者训练

*注意*：选用patch：192²	占单个视角的14.1%	细节与局部上下文比较均衡	仍需处理重叠将 192 个真实源像素放大到 256 个输入像素，不会创造新的物理细节，只是改变局部内容在预训练视觉编码器中的尺度

*可选*：通过历史 Focus Coverage Map，让 Actor 知道当前导航步骤内部哪些区域已经看过，而不是硬性禁止两个 Patch 重叠。

可以定义：
\[
\operatorname{IoU}(b_1,b_2)
\]

作为可选的软惩罚，且只对同一相机的两个 Patch 计算

### 整体流程


### Actor

**Actor采用分层决策的设计**
区分三种随机决策：
1. 是否继续感知；
2. 如果继续，选择哪个摄像头；
3. 在该摄像头内部注视哪里。
定义感知状态：
\[
s_{k,j}=(G_k,L,M_{k,j},B_{k,j})
\]

其中：
- \(G_k\)：四视角 Glance 的空间特征；
- \(L\)：导航指令特征；
- \(M_{k,j}\)：本步已经累积的视觉表征；
- \(B_{k,j}\)：本步已经观察的空间区域和 Focus 次数。

#### Actor 流程 形式化
先输出：
\[
z_{k,j}\sim
\operatorname{Bernoulli}(p_{\mathrm{stop}})
\]

其中 \(z=1\) 代表结束当前感知。

如果 \(z=0\)，则：
\[
v_{k,j}\sim\operatorname{Categorical}(p_0,p_1,p_2,p_3)
\]

然后在对应相机中采样：
\[
u_{k,j}\sim
\mathcal N(\mu_{v},\operatorname{diag}(\sigma_v^2))
\]

连续中心位置为：
\[
\boxed{
(x,y)=
\frac{P}{2}
+
(512-P)\operatorname{sigmoid}(u)
}
\]

对于 \(P=192\)，实际合法中心为：
\[
x,y\in(96,416)
\]

像素裁剪时再按统一规则进行整数舍入。

因此，一次继续观察的联合概率是：
\[
\boxed{
\pi_\theta(a\mid s)
=
(1-p_{\mathrm{stop}})
\pi_\theta(v\mid s)
\pi_\theta(u\mid v,s)
}
\]

而 STOP 的概率就是 \(p_{\mathrm{stop}}\)。


**最大Focus次数**
 \[
\boxed{N_{\mathrm{focus}}\in\{0,1,2,3,4\}}
\]
Actor 在每次注视之前决定是否继续。
Focus 数量	当前视觉特征编码量	输出给 Qwen 的当前视觉 Token
0	64	64
1	128	64
2	192	64
3	256	64
4	320	64

这里假设采用上一版确定的 Spatially Aligned Fusion。编码量包含四个 Glance 的 64 Token 和每次 Focus 的 64 Token；输出 Qwen 的 64 Token 不包含另计的 BATS 历史。

允许四次 Focus 并不会直接使 Qwen 的当前视觉序列增长到 320 Token，但会增加 Vision Encoder 的实际计算量。


*注意*
S采用igmoid-transformed Gaussian，保留可训练的标准差。

默认：
\[
\mu=0,\qquad \sigma=0.8
\]

其中初始 \(\sigma=0.8\) 是潜在空间的标准差，不是实际图像中心坐标的标准差。

建议训练期间限制：
\[
0.1\le\sigma\le1.5
\]

并保留采样的 \(u\)、camera ID、STOP、旧策略 log probability 和当时的观察特征。

这样在 PPO 和 GRPO 更新时可以正确重建旧动作概率

#### Actor结构
Actor v1 · Spatial Query Policy

\(G\in\mathbb R^{64\times2560}\)四视角，每视角 4×4 Glance Token

                      ⬇️

Visual / Language Projection
\(2560\rightarrow192\)

                      ⬇️

Spatial Transformer × 2
192 hidden · 4 heads · FFN 768

                      ⬇️

Instruction-Conditioned Query Cross-Attention
读取视觉网格、Focus 累积特征和 Coverage

                      ⬇️

STOP（Bernoull i）｜ Camera(4-way) |  Position(μx, μy, σx, σy)



**具体组件配置**
组件	输入 → 输出	结构配置
Visual Projection	2560 → 192	Linear + LayerNorm + GELU
Language Projection	2560 → 192	Linear + LayerNorm + GELU
Spatial Position	2D 坐标 → 192	Fourier Features + Linear
Camera Embedding	4 → 192	Learnable Embedding
Spatial Encoder	64×192 → 64×192	Transformer Encoder ×2，4 Heads，FFN 768
Query Fusion	192 → 192	Cross-Attention，4 Heads
Coverage Encoding	每相机 4×4 + Focus 坐标	小型 MLP
STOP Head	192 → 128 → 1	GELU + Linear
Camera Head	192 → 128 → 4	GELU + Linear
Position Head	相机条件 384 → 192 → 128 → 4	GELU + Linear
Exploration Scale	2D/相机	可学习 log_std

Position Head 输出选定相机的二维均值和标准差；可以为四个摄像头分别生成四组参数，但实际仅采样选定相机对应的分布。

关键是保持四个摄像头身份，不把四个 4×4 网格无几何说明地当作一个普通 8×8 图像。

参数量预计约 2–3M，正式实现时应该打印实际参数数目，不用估算值作为报告结果。


*这套架构参考了什么？*
- AdaptiveNN：小型决策网络与高容量视觉表征网络解耦，连续注视位置。
- Seeker（2026）：冻结预训练视觉特征，通过任务条件化 Query 和空间注意力学习与行动相关的视觉区域。
  arxiv.org+1
- Task 4 SCOPE：复用已完成的语言编码、特征缓存、训练器、独立 Actor/Critic 与 STOP 接口

### Vision Adapter
 **目的** 
 Stage 1 和 Task 4 已经发现，额外 Focus 可能导致导航退化，因此task4的直接拼接：
 \[
[G;F_1;F_2]
\] 会改变下游 Qwen 的视觉序列组织
 **因此新版本始终保持：**
\[
\boxed{
M_{k,j}\in\mathbb R^{4\times16\times2560}
}
\]

也就是四个视角，每个视角 16 个当前视觉 Token。Focus 只能更新这些 Token 的内容，不增加下游当前视觉序列长度

以下是推荐设计，如果有更少参数量的更简单的网络能完成adapt的功能就更好，adapter可能需要被当做相对独立的视觉表征训练

*TODO真实实现需要寻找更加简便有效的设计方法，考虑到算力限制，可能无法直接设置一个需要训练的视觉层然后用导航指标反传，同时此处也不是核心创新*


#### 简单的 Adapter 
一个短暂的 Adapter 校准步骤，使用 R2R train 图像上的冻结全图视觉特征作为蒸馏教师，训练 Spatially Aligned Fusion，然后冻结它

1. Spatial Fusion Calibration

用训练场景中的冻结视觉教师特征，简短训练几何对齐和门控融合。确认 Focus 不会因为融合接口持续破坏已有 Glance 表征。

完成后冻结 Adapter。本步骤不训练 Actor/Critic。


2. Unified Group-Assisted Actor–Critic Training

Actor 从第一轮就同时支持 Camera、Continuous Position 和 STOP，最多四次 Focus。

普通状态使用导航 GAE，少量状态执行同状态分组比较并使用 Group Advantage。Critic 统一学习主轨迹长期回报

### Critic
只保留一个真正用于长期回报估计的 Critic：
\[
\boxed{
V_\phi(s_k^{0})
}
\]

其中 \(s_k^0\) 是第 \(k\) 个导航步骤的 Glance 完成后、Focus 选择之前的状态。

它估计：
在当前指令和粗略视觉状态下，遵循当前感知策略并继续导航，预期可以获得多少后续导航回报？

Critic 不直接预测 Focus 的二维中心，也不直接预测导航 waypoint。
5.1 Critic 逐层配置
组件	配置
Visual Input	4×16×2560
Visual Projection	2560 → 128
Language Projection	2560 → 128
Spatial Encoder	Transformer ×1，4 Heads，FFN 256
Attention Pooling	64 Tokens → 128
Value MLP	128 → 128 → 1
激活	GELU
输出	单个标量 \(V_\phi(s_k^0)\)
预计参数	约 0.8–1.2M
Optimizer	AdamW，独立于 Actor

Critic 可以比 Actor 小，因为它只估计预期回报，不需要构造精确的空间位置分布。

Actor 和 Critic 使用同一个冻结 Qwen Vision Encoder 输出的特征，但各自拥有独立的可训练投影、Transformer 和输出头

### RL Training by Group-assisted PPO (GA-PPO)的分层架构
1. 感知动作：Actor 自主决定 0–4 次 Focus，每次是 Camera + 连续中心；STOP 由 Actor 输出。

2. 唯一环境奖励：导航进展 + 成功/失败终止 + Focus 成本。不再叠加复杂的 ROI、动作损失和重叠奖励。

3. Group 采样：同一导航状态下从旧策略独立采样八条完整感知序列，采用 Leave-One-Out 相对优势，不强制每个相机出现。

4. Critic：独立网络，只估计导航长期价值，用实际主轨迹的 GAE/λ-return 更新。

5. Actor 更新：同一个 Actor、同一轮优化；普通状态使用 GAE，分组状态使用 Bootstrap Group Advantage，不在同一个状态上机械叠加两套奖励。

6. 训练结构：短暂校准 Adapter，随后只执行一个统一的 On-policy Actor–Critic 训练循环。

#### 统一奖励

\[
\boxed{
\begin{aligned}
r_k={}&
0.2\operatorname{clip}
\left(\frac{d_k-d_{k+1}}{3},-1,1\right)\\
&-0.01-0.01N_k^{\mathrm{focus}}\\
&+r_k^{\mathrm{terminal}}
\end{aligned}
}
\]

其中：
\[
r_k^{\mathrm{terminal}}=
\begin{cases}
+2,&\text{导航成功终止}\\
-0.5,&\text{导航失败终止}\\
0,&\text{尚未终止或人为截断}
\end{cases}
\]

\(d_k\) 为训练环境中到目标的 geodesic distance，\(N_k^{\mathrm{focus}}\in\{0,\ldots,4\}\)。

这套奖励只回答：

当前主动感知最终促成了多少导航进展，以及为此付出了多少额外视觉成本

#### Group part 采样方法

**采样设置**
Group-assisted 发生在一步Focus动作执行时，采样一组Focus动作，我们设置：

- 每组 8 条独立采样的完整感知序列；
- 每条序列自主选择 STOP、Camera、Position；
- 可以重复选择同一个相机，但不强制重复相同像素区域；
- 不人为保证四个方向各有一条；
- Camera Head 初始分布尽量均匀，并监控探索熵。
4 条可用于吞吐试验；8 条作为正式训练的起始推荐

**采样方法设计**

同状态采样的 \(K=8\) 条感知序列，先计算各分支一个导航动作块之后的收益。

为了让局部比较不仅看立即进展，还能考虑后续状态，可以使用冻结旧 Critic 的一步 Bootstrap：
\[
\boxed{
\widehat Q_{k,i}
=
r_{k,i}
+
\gamma(1-d_{k,i}^{term})
V_{\phi_{\mathrm{old}}}(s_{k+1}^{(i)})
}
\]

这里 \(d^{term}\) 为真实终止指示，而不是到目标的距离。

当 Critic 最后一层初始为零时，早期 \(\widehat Q\) 主要依赖实际导航奖励；随着 Critic 学习，它逐步引入对后续导航结果的估计。

组内优势建议使用：
\[
\boxed{
A_{k,i}^{\mathrm{group}}
=
\widehat Q_{k,i}
-
\frac{1}{K-1}
\sum_{m\ne i}\widehat Q_{k,m}
}
\]

属于 RLOO（REINFORCE Leave-One-Out）风格的组相对优势，而不是标准的组均值/std 归一化 GRPO。

它的优点是第 \(i\) 条轨迹的基线不包含自身结果；在各条轨迹独立按同一旧策略采样时，这是一个更干净的控制变量设计。

不建议每组强制除以组内标准差，因为不同 Crop 可能取得几乎相同的导航进展，这时极小的组方差会放大噪声

#### Actor-Critic 架构 训练更新

核心：同一批导航数据中，一部分状态用 Group Advantage，其余状态用 GAE Advantage。Critic 统一从真实导航轨迹学习。

**分别计算优势Advantage**
设一次采集得到 \(B\) 个真实导航 transition。

其中一小部分状态执行 \(K\) 次同状态分支试验。

对于正常状态：
\[
A_k^{actor}=A_k^{GAE}
\]

对于分组状态的第 \(i\) 条候选序列：
\[
A_{k,i}^{actor}=A_{k,i}^{group}
\]

也就是说：
\[
\boxed{
A^{actor}=
\begin{cases}
A^{GAE},&\text{普通导航状态}\\
A^{group},&\text{同状态分组样本}
\end{cases}
}
\]

*注意* 它不是将两个优势分别优化一遍，更不是直接对同一个导航奖励重复加权。

Critic 仍然在真正继续运行的主导航轨迹上，用 GAE/λ-return 更新。

分组中的其他候选只作为反事实对照，不会凭空构造它们未来完整 Episode 的真实回报

#### 损失计算

对每个感知决策 \(j\)，计算：
\[
\rho_{k,i,j}
=
\exp\left(
\log\pi_\theta(a_{k,i,j}\mid s_{k,i,j})
-
\log\pi_{\mathrm{old}}(a_{k,i,j}\mid s_{k,i,j})
\right)
\]

然后使用：
\[
\ell_{k,i,j}
=
-\min
\left(
\rho_{k,i,j}A_{k,i},
\operatorname{clip}(\rho_{k,i,j},1-\epsilon,1+\epsilon)A_{k,i}
\right)
\]

其中：
\[
\epsilon=0.2
\]

建议采用逐感知决策的 PPO Ratio

对于一条序列，保留所有实际随机决策的损失之和；对 Group 内的 \(K\) 条序列取均值，再对导航状态取均值，避免 Group 数量多的状态在整个训练 Batch 中被过度加权。

这样一轮 Update 只包含：
\[
\boxed{
\mathcal L_{\mathrm{train}}
=
\mathcal L_{\mathrm{Actor}}
+
0.5\mathcal L_{\mathrm{Critic}}
}
\]

Actor 和 Critic 仍使用各自的独立优化器。熵正则如需加入，应按 STOP、Camera、Position 三类动作分别监测和缩放，而不是未经校准直接求和

#### 完整训练流程和配置参考

1. 初始化 Episode 和旧策略快照
   
   冻结本轮采样所用的 Actor \(\pi_{\mathrm{old}}\) 和 Critic \(V_{\mathrm{old}}\)，加载 R2R train 指令与 Habitat 起点。
2. 正常执行 Glance → Focus → SimpleNav → Habitat
   
   Actor 每步选择 0–4 次 Focus，然后由 SimpleNav 执行导航动作。记录旧概率、旧价值、实际奖励和后继状态。
3. 少量导航状态启用 Group Sampling
   
   对预先选中的导航状态，从同一个旧 Actor 独立采样 8 条完整感知序列，并分别执行一个导航 Action Chunk。
   
   使用重置与前缀重放检查同状态一致性。
4. 选择主导航分支继续 Episode
   
   在执行分支奖励比较之前，预先随机指定其中一条策略采样作为真实主轨迹。
   
   其他分支仅用于相对优势估计，不能选择奖励最高的分支继续导航。
5. Episode 结束，写入 Rollout Buffer
   
   保存真实主轨迹；另存同状态分支和它们的收益。所有训练数据保留原采样策略概率和状态快照。
6. 积累足够数据后统一更新
   
   累计约 64 个真实导航 transition，并在 Episode 边界结束采集。计算普通状态的 GAE、分组状态的 Leave-One-Out 优势和 Critic 的固定 λ-return。
   
   使用同一轮更新分别优化 Actor 和 Critic。
7. 保存 Checkpoint，开始下一批 Episode
   
   清空 On-policy Buffer，使用更新后的策略重新采样，不重复使用旧的 Group Reward 或旧 GAE。

继承 Task 4 已验证的 Episode 边界 Checkpoint 约束，不要求 Habitat 支持运行中任意时刻的精确状态快照。

**统一训练配置** 
参数	推荐值
Patch Size	192×192
Max Focus	4
Group Size	8
Group State 比例	约 12.5%
Rollout Batch	≥64 个真实导航 transitions
PPO Epochs / Update	2
Actor Minibatch	16 个状态等价批量
Actor LR	1×10⁻⁴
Critic LR	2×10⁻⁴
Gamma / GAE Lambda	0.99 / 0.95
PPO Clip	0.2
Gradient Clip	1.0
初始 STOP 概率	约 0.25
Camera Head 初始化	近似均匀分布
初始连续位置 std	0.8（Gaussian latent）
初次训练预算	4,096 个主轨迹 transitions
下一档预算	10,000–20,000 transitions

其中初始 STOP 概率约 0.25，是为了让训练从一开始就出现 0、1、2、3、4 次 Focus 的不同路径，而不是强制先训练一个固定四次的 Actor。

这些仍是建议起始参数，不是已经通过超参数搜索的结果。

**Group Sampling 不必每个导航状态都执行**
假设每次更新收集 64 个真实导航 transition，其中约 8 个状态进行 K=8 的 Group Sampling。
64个主轨迹 transitions 
8个Group Anchor 状态
56个额外分支执行


合计约 120 次单 Chunk 分支执行

64 + 8×(8−1) = 120。未包括同状态恢复所需的前缀重放，也未计 0–4 次 Focus 造成的变动视觉编码成本。

如果所有状态都执行八分支，则成本会大幅增加，而收益未必同比例提高，因此建议先用 12.5% 的分组比例

#### GA-PPO 训练方法的风险边界
**注意**
要明确其统计含义：
- 普通状态使用较长时间尺度的 GAE 估计。
- 分组状态使用一步导航回报加 Critic Bootstrap 的相对优势。
- 两者的信用分配时间尺度并不完全一致。
因此这是一个有待验证的混合优势估计器，不是已经证明优于 PPO 或 GRPO 的新算法。

论文必须保留 PPO-only 消融。如果 Group-Assisted 版本没有带来增益，就不应该为了方法复杂度继续保留它

### 专家监督辅助 + RL优化的整体训练 思路考虑

1. GT 用于提供稳定的导航决策监督
在专家轨迹的某个导航状态，比较 Focus 前后的 SimpleNav 动作损失：
\[
\boxed{
r^{GT}_{k,j}
=
\mathcal L_{\mathrm{nav}}^{\mathrm{before}}
-
\mathcal L_{\mathrm{nav}}^{\mathrm{after}}
}
\]

如果 Focus 后更接近专家动作，就给予正奖励。

这能直接回答：这次注视是否帮助 SimpleNav 做出了更符合专家的导航决策？

它与 AdaptiveNN 利用分类损失改善构造自奖励的思想相似。
2. Rollout 用于验证真正的导航收益
继续保留环境奖励：

因为动作预测损失降低，并不必然代表真实导航成功率提高。
3. 最简洁的训练结合方式
- 训练初期：利用 GT 路径和动作损失改善，帮助 Actor 学会选择有价值的注视区域，降低探索难度。
- 正式 RL 阶段：使用真实 Rollout 的导航进展和成功奖励训练 Actor/Critic。
- 可选：保留一个小权重的 GT 辅助损失，防止 RL 训练使注视策略退化。
关键限制是：GT 动作监督应在专家轨迹状态使用，或在偏离专家路径后重新查询有效的专家动作，不能直接把原专家路径上同一时间索引的动作当成偏航状态的 GT。

### Experiment

#### 消融实验——设计的训练方法 GA-PPO有没有用
为了确定 Group Sampling 是否真正有贡献，建议主训练只保留三个学习算法对照：
训练算法	Actor	Critic	同状态分组
PPO-only	✓	✓	—
Group-only（RLOO）	✓	—	✓
Group-Assisted PPO	✓	✓	✓

三组使用相同 Patch、融合、最大 Focus 次数和奖励。比较时不仅对齐训练主轨迹数，还应报告总导航模型调用量、分支重放成本和总 GPU 时间，否则 Group-Assisted 方法可能只是在使用更多计算预算。

再与 Glance-only、固定 Focus、随机 Focus 做导航性能和感知成本比较即可。

需要重点报告的不是 Group Loss 是否下降，而是 SR、SPL、平均 Focus 次数，以及相同视觉预算下的实际导航表现。