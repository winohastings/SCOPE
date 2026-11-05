# Stage2 Task4 增量发布

## 范围

本次只整理与验证既有 Task4 产物，不新跑 GPU 训练或导航评估。新增 `Experiment/SCOPE/Stage2/` 的11个Python模块、README及八份交付文档、原Task4任务书、执行前审阅、20项Stage2单元测试与关键实测证据。根README新增阶段入口，不改Stage1实验文件、历史数据或既有依赖补丁。

关键实现为 `features.py`、`policy.py`、`trajectory.py`、`training.py`、`runtime.py`：typed当前特征缓存、五开关统一Actor、独立Critic、执行后更新的State GRU、增量Focus记忆、独立感知STOP、联合顺序PPO/GAE及checkpoint恢复。

## 结果与边界

- 最终 `artifacts/release/`：54次固定状态配对诊断，train/dev各27次；匹配历史后原scatter与缓存动作最大差0。
- 主运行24个真实训练transition/4次PPO更新；补充恢复2个transition/1次更新，总计26/5。正常导航完成82个策略步。
- 开发检查固定2集，SR为0.5→0.5；nDTW描述性均值0.47005→0.58335，成败交换，不宣称导航改善或统计显著。
- 既有根全套测试存在无关模块缺失；保留 `regression_all.log`，不冒充全套通过。相关SCOPE/协议/归档测试单独记录。

## 保留证据

- 最终 `FINAL_VALIDATION.json`、`release/AUDIT.json`、`COMPLETE.json`、环境与资源总账、依赖及历史保护审计。
- 主运行与恢复的config/episode/result、原始eval_info/trajectory、27+27条诊断、逐次PPO更新标量、学习健全性结果。
- `release_runtime/`、`diagnostic_runtime/` 的唯一结构化轨迹和scope流采用无损gzip；GPU采样与保护快照同样压缩，源hash可验证。
- 保留首次历史混杂诊断及native特征不一致记录，避免只发布成功日志；删除打包中的重复逐步scope/data和过期pilot。

不上传Actor/Critic权重、tensor rollout、图像、预训练权重、官方仓库、场景或环境。权重和rollout的路径、字节数与SHA256保留在 `STAGE2_EXCLUDED_FILES.json`；因此本归档可审查更新结果，但直接checkpoint恢复或学习健全性重放仍需原机器上的tensor文件。原报告引用的未打包路径保持历史原文，不承诺这些文件已上传。

所有新增原始文件见 `STAGE2_SOURCE_MAP.json`；完整发布清单见 `FILE_MANIFEST.json`，累计源映射见 `SOURCE_MAP.json`。源码未做跨机器绝对路径迁移，不直接执行pilot覆盖历史结果。Stage2复用Stage1原依赖，未新增第三方源码补丁。

## 验证命令

本次发布重新运行34项SCOPE测试（含20项Stage2）和4项Stage1协议回归，均通过；输出保存在同目录 `STAGE2_SCOPE_REGRESSION.log`、`STAGE2_PROTOCOL_REGRESSION.log`。无GPU归档验证重新从原始训练、诊断、开发恢复数据检查26/5/54分母，并与最终完成门禁及源码hash比较，不只读取报告中的“通过”结论。新增2项Stage2归档测试覆盖真实产物与错误训练分母拒绝；另复用既有4项归档回归。

没有重跑原根目录全套的无关缺失模块测试，没有宣称修复该历史问题；本次不启动模型服务或训练。

```bash
python3.11 tools/verify_stage2_release.py

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope_*.py'
```
