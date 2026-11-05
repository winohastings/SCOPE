# Focus 受控诊断

## 实验约束

只使用既有 train/dev 隔离清单中的 R2R train episode1（7y3sRwLe3Va）和固定开发
episode9791（jh4fc5c5qoQ）。每集取起始状态，以及执行预先固定的 Glance 导航chunk后
的状态；后者再比较有/无历史。共4个物理状态、6个状态×历史条件、每条件9个分支，
合计54次受控预测/执行。没有使用val_unseen结果设计或选择模型。

每个分支重置相同episode并重放完全相同的prefix动作，检查四相机像素SHA256和起始
geodesic距离一致；固定扩散seed4242，关闭Actor选取，强制同样ROI。比较一个真实八
waypoint chunk后的 `geodesic_distance_before - geodesic_distance_after`（米）。
这不是SR、完整episode收益或统计显著性试验，6个条件并不独立。

## 分支与结果

最终原始数据位于 `artifacts/release/diagnostics_train/diagnostics.json` 和
`artifacts/release/diagnostics_dev/diagnostics.json`；配对审计为 `release/AUDIT.json`。

| 分支 | 干预 | 相对Glance平均progress差(m) | 正/负条件数 |
|---|---|---:|---:|
| fixed | front.R1 + front.R4，ROI0/3 | -0.16993 | 2 / 4 |
| fixed_spatial | 同样ROI，加显式相机/原图ROI文本 | -0.17659 | 3 / 3 |
| cross_camera | front.R1 + left.R1，ROI0/4 | +0.01852 | 3 / 3 |
| cross_spatial | ROI0/4，加空间文本 | +0.03544 | 2 / 4 |
| right_rear | right.R1 + rear.R1，ROI8/12 | +0.02897 | 3 / 3 |
| fixed_low_detail | 同样ROI0/3，先64再放回256 | -0.22708 | 3 / 3 |
| single_front | 只获取ROI0 | -0.01576 | 2 / 4 |

均值只作描述，不能据此挑冠军配置。native_scatter是缓存/原生编码接口对照，不是额外
独立策略候选。初次native对照混入历史重编码batch差异，原始数据留在
`release/diagnostic_attempts/`；最终重测保持历史特征不变，只替换当前特征来源。

## 四个假设的证据

### H1：空间/相机位置歧义

支持存在风险：代码审阅确认Focus的TVI继承源相机，ROI原图坐标并未自动表达为冻结VLM
能理解的语义。真实pixel/geometry检查则反对“裁剪坐标实现错误”：原图象限与相机朝向
精确对应，native grids和现有M-RoPE/scatter保持合法。

空间文本对ROI0/3的配对作用在6条件中3正3负，平均约-0.00666m；对跨相机ROI0/4是
5正1负，平均约+0.01692m。说明相机/位置解释会影响输出，但效果不一致，不能认为
“仅缺空间文本”已经解释全部回退。**尚无证据证明必须启用或训练空间adapter**；默认
关闭，保留可选零初始化残差接口，未经实测不得宣称有效。

### H2：冻结 VLM 输入分布偏移

支持“敏感且可能有偏移”，不能证明单一因果：128全图、256局部裁剪、额外图像块与原
训练全图上下文不同；相同真实状态下仅添加不同ROI即可改变导航输出和progress。
真实诊断进一步发现BF16旧全batch与顺序调度在像素完全一致时仍有特征数值差异。
同调度原生重编码则特征逐位一致，排除“bank错误复制特征”这一解释。

低细节对照并未稳定改善固定Focus：完整细节相对低细节仅1/6条件更好，另5/6更差，
平均被一个约+0.644m条件主导。低细节干预也会改变分布，故不能将其解释为纯因果
“高频细节有害”或“高分辨率本身无用”。本轮没有微调VLM或测分布距离，不宣称证实。

### H3：低价值或干扰裁剪

支持明显的ROI条件性：固定0/3均值为负，而跨相机/右后相机均值为小正值，正负条件
混合。train初始状态Glance progress=0.63383m；同状态fixed=0.77087m（+0.13704m），
cross_spatial=0.79851m（+0.16468m）。train后继状态无历史时fixed相对Glance
+0.23935m，但另一个dev无历史条件fixed下降约0.87539m。

因此可以回答：**Focus在受控条件下有可测的局部潜在收益，但固定裁剪并不可靠**。
这些例子不是oracle训练标签，也没有据其改Actor超参或挑开发最优策略；两episode
样本远不足以证明泛化收益、平均导航改善或收益显著。

### H4：历史视觉特征交互

同样后继像素、同样固定Focus，有/无历史时相对Glance的收益差发生明显变化：

| 状态 | 有历史Focus增量(m) | 无历史Focus增量(m) | 交互差(m) |
|---|---:|---:|---:|
| train episode1，state1 | -0.00525 | +0.23935 | -0.24461 |
| dev episode9791，state1 | -0.11420 | -0.87539 | +0.76119 |

支持强条件交互，且方向不同。移除历史同时改变VLM上下文长度/TVI等，因此不能单独
归因某一种历史融合机制；不能为了改善两个诊断状态而关闭正式导航历史。

## 一致性与限制

- 相机/ROI、像素身份、grid/shape、冻结参数、原scatter的检查通过。
- 正常导航不重编码当前图；诊断显式允许额外编码，计时不混入训练吞吐。
- 同调度缓存特征与原生特征最大差0；最终匹配历史后的动作最大差也为0。
  最终诊断中旧BF16全batch特征relative-L2最大约0.07275，差异如实保留，不能把
  新调度称作与Stage1逐位等价。
- 没有证明空间adapter必要，没有训练适配器，没有完成Focus价值的多seed/多scene估计。
- 诊断支持开展后续研究，不构成启动全规模训练的自动授权。
