# Stage2 / Task233 发布说明

本次整理Task233导航价值教师、全train扩采与Gate1空间监督热启动的实现、文档、关键日志和数据，不重跑训练/导航，不修改此前任务。相关Task21/22/23/231/232顶层Python依赖与现有发布版本一致。

## 以最终产物为准的状态

**Gate0=PASS；Gate1=INCONCLUSIVE；Gate2/3=NOT_RUN且锁定。** 任务计划中的“未执行训练”、Gate0/扩采报告中的“Gate1未运行”描述各自当时阶段，不能覆盖后续Gate1结果。最终状态见 `artifacts/gate1_v1/PROGRESS_GATE.json`。

- Gate0 v3完成72 anchors（含8个校准）、1006候选分支；教师相对STOP/fixed/random的一步进展通过原冻结门槛。它不是整集SR/SPL改善，也不是on-policy训练。
- 全train扩采保留64个旧非校准anchor并新增192个，共256个、61场景、128条轨迹；teacher_train/heldout为176/80，可信标签81/37。扩采新增2640分支，合并非校准3520分支；优化器更新0。`PASS_PARENT`只引用原Gate0，不等于扩采重新通过全部统计门槛。
- Gate1两个初始化候选各27次空间监督更新，选择update14_reset；最终用81个可信train anchors做132次更新，共186次。STOP参数冻结，loss不含STOP；没有训练Critic、Adapter、Qwen/SimpleNav，没有Group或PPO。
- 可信heldout加权NLL从3.6030588升至4.0181371，候选top1从32.43%降至10.81%，触发预注册停止。没有进行learner-induced追加标注、真实Gate1导航验收或SR/SPL验收。
- split与微步j存在既有不均衡，初始空间输出并列使top1采用first-argmax。不能将结果解释为均衡泛化证明，也不能单凭训练损失下降宣称有效。

## 上传内容与关键日志

下列路径相对于 `Experiment/SCOPE/Stage2/Task233/`：

| 内容 | 入口 |
|---|---|
| 代码、预注册与测试 | 顶层Python、README、TEACHER/EXPANSION/GATE1_PROTOCOL |
| Gate0真实执行日志 | `gate0_v3.log`、`artifacts/gate0_v3/server.log` |
| 扩采执行日志 | `fulltrain256_v1.log`、`artifacts/fulltrain256_v1/server.log` |
| **Gate1训练日志** | `gate1_v1.log`、`artifacts/gate1_v1/training.csv`（186次loss/gradient记录） |
| 候选与运行事件 | `artifacts/*/candidate_results.jsonl.gz`、`runtime/scope.jsonl.gz` |
| 教师状态数据 | gate0_v3的72个、fulltrain256_v1的192个 `runtime/states/*.pt.gz` |
| 热启动结果 | before/after_predictions.csv、stratified_supervision.csv、OFFLINE_AUDIT、INITIALIZATION_SELECTION、TRAIN_MANIFEST |
| 门槛与审计 | 各run的PROGRESS_GATE、teacher_signal_audit、VERIFICATION、EXPANSION_VERIFICATION、INDEPENDENT_VERIFICATION及报告 |
| 历史失败证据 | gate0_v1/v2的INCOMPLETE、配置、服务日志及已有候选记录；Gate1首次核验记录不删除 |

**ActorState是监督输入数据，不是模型checkpoint。** 保留256个非校准状态及8个校准状态，哈希与候选manifest对应；不得将它们送入on-policy Group/Critic/GAE。仅排除早期失败gate0_v2的12个状态tensor及缓存/空日志，不整包复制官方仓库、场景资产或基础模型。

## Checkpoint与来源追溯

五个Gate1模型文件（两个初始化、两个pilot末状态、最终warmup）全部留本机，仅上传 `*.pt.placeholder.json`。占位包含原路径、字节数、实测SHA256与 `uploaded:false`。相关Task23父权重继续沿用已有占位；占位不能由torch加载，恢复需从可信原机器取得权重并核验哈希。

执行后 `spatial_warmup.py` 增加了拒绝覆盖已有输出保护。已保留 `artifacts/gate1_v1/source_snapshot/spatial_warmup.py` 的真实执行版本，独立核验记录明确列出旧/新hash，发布验证器同时核验两者，不覆盖历史证据。

`TASK233_SOURCE_MAP.json`记录本次原样复制/压缩前哈希，`TASK233_EXCLUDED_FILES.json`记录排除范围，累计 `FILE_MANIFEST.json`校验实际上传文件。原DELIVERY_FILES与输入hash清单仍包含原机器路径，不代表发布包全部文件名未变。

压缩状态/候选JSONL/大型DATA_MANIFEST需要在独立实验副本解压，并处理原绝对路径，才能供原脚本读取；完整导航还需真实模型、场景、双环境。部分旧Task产物曾在源机器清理，本发布不承诺那些历史路径仍可用，也不删除已发布的历史证据。不要直接运行会改写原run分析/验证文件的主程序来验收发布包。

## 只读发布校验

```bash
python3.11 tools/verify_task233_release.py
python3.11 -m unittest discover \
  -s tests \
  -p 'test_*archive.py'
```

标准库验证器不加载pickle、不用GPU；检查累计hash、候选分母/数据边界、256个状态hash、轨迹隔离、186次更新日志、CSV重算加权NLL/top1、门禁结论和checkpoint占位。保留的原CPU权重复核是历史证据，不冒称本次重新训练或验收导航。

发布时16项Task233单元测试、78项既有模型回归测试、29项归档测试通过，共123项。另在CPU使用原评分/几何函数重算3646条候选，最大评分误差0；使用 `weights_only=True` 读取归档内可信监督状态，256个非校准ActorState与实际执行状态指纹逐一一致。此项只读重放不使用模型checkpoint，不进行训练或导航。记录在 `docs/release/task233_validation/`。

全部本次新增/修改文件见 `docs/release/TASK233_CHANGED_FILES.txt`。
