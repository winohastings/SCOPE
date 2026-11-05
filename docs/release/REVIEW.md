# SCOPE Stage1 发布审查

## 三项任务到方法实现

| 任务 | 核心文件 | 发布方式 |
|---|---|---|
| Task1：采集与接口 | `scope_server.py`、`run_smoke.py`、`capture_observation.py`、`verify_fixed.py` | 保留原文件与测试 |
| Task1：原始512、Glance128、Focus256、metadata、可变token | legacy `input_context.py`、`navvla_context.py`、`qwen35_vision.py`、`QWen3_5.py`、`navvla_qwen35_cpm.py` | 精确补丁，不复制233个未修改官方文件 |
| Task1：动作执行、Habitat渲染与日志 | SimpleNav benchmark/runner/Habitat runtime 与 legacy server | 基于锁定官方提交的补丁 |
| Task2：协议与选择器设计 | `Experiment/SCOPE/Stage1/` 的协议、审计、统计、成本、G0–G4 配置和清单 | 全部保留；草案并非已执行训练 |
| Task3：缩减可行性 | `Stage1_Feasibility/` 的11个执行脚本、配置、manifest、最终报告和图表 | 保留原版本；本轮不改变实验实现 |
| Task3：执行证据 | 每组64集 episode结果、eval_info、唯一结构化逐步轨迹、CSV、成本/资源/一致性审计 | 逐步与采样大文件无损压缩，删除打包中的冗余日志 |

## Review 结论

1. 原512图先保留，再生成兼容256/Glance128；Focus为原图2×2网格精确裁剪。视觉输出保持2560维，变分辨率块沿既有scatter/M-RoPE路径插入。
2. Task1 固定0/3与旧Random只用于旧验证；Task3独立策略满足八步16ROI覆盖、两ROI不同相机及SHA256随机可复现，不能用Task1服务默认行为代替Task3。
3. 历史路径保留原BATS选择与混合编码后pool4，不能把历史context token数当作实际encoder patch数。成本分项重放与在线混合batch分别记录，FLOPs是声明覆盖范围的估计而非精确全模型实测。
4. 256集分母、失败、token预算和配置证据可独立核查。原始完整任务2尚含G4设计，但任务3明确缩减为G0–G3；不训练Actor/Critic/PPO，不启动第二阶段。
5. 学习型选择器尚未实现，现有learned模式脚本仅验证接口。当前Focus没有提供优于Glance的导航证据，不能宣传已证明SCOPE学习策略有效。

## 保留与排除

保留兼容性JSON、冒烟聚合、最终回归日志、原始执行状态、各失败尝试的状态/异常日志、完整评估episode级证据、唯一逐步轨迹、成本profile、固定输入试点数据和资源汇总。

排除完整官方runtime、重复server JSONL、按步重复scope/action/位姿文件、每组重复控制台进度、图片NPZ、Python缓存、全量GPU采样的未压缩副本、旧的全文件清单；GPU样本和结果逐步CSV仍有无损压缩档。历史文档中未打包的原路径保持原文，它们不是发布文件存在性承诺。旧 `FILES_MANIFEST.tsv` 由新的发布清单替代，避免把机器上数千个冗余产物当发布内容。

不引入与三项任务无关的 Assumption、NaVid、AdaPerc-Random、早期边际收益实验，也不打包 habitat_extensions 动态相机旧原型。

## 可复现性与发布限制

- 原实验路径绝对化，源码原样归档，不宣称新机器直接启动。历史legacy源没有upstream提交，逐文件hash是现有追溯上限。
- 冻结报告与输入manifest含R2R开发指令和路径信息；持有者后续已创建公开远端，保留其可见性及Apache-2.0许可证，不将该许可证扩展解释为上游数据授权。
- 新增工具只负责依赖重建和证据核验，不改变实验结果；发布时运行的是CPU回归/完整性检查，不是新一轮GPU评估。
- GitHub SSH 身份校验、仓库可访问性、实际push与远端clone验证分别记录，未成功的步骤不会写成完成。
