# Stage2 Task2.2 增量发布

本次按仓库整理发布职责执行，角色规范文件仍缺失，依据根AGENTS/索引/任务与实际产物；不重跑Habitat、GPU推理或正式RL训练。

## 实现和必要依赖

Task22的6个Python模块、配置、README、REPORT和TRACE_SCHEMA原样归档；新增19项Task22单测、原任务书与AGM方法计划。方法计划描述后续训练设计，不代表本轮已经执行。

Task22实际依赖Task21新增`incremental.py`及修改后的`runtime.py`，本轮同步这两个文件。Task21旧runtime原样保存在`docs/release/task22_before_sources/Task21/runtime.py`；旧Task21专属SOURCE_MAP作为历史记录保留，累计SOURCE_MAP更新为当前版本。Task21冻结Adapter及历史结果不改，Stage1/Task4不改。

## 关键数据和边界

- 最终Actor初始化权重9.2MB，参数2,293,073，optimizer更新0；不是已训练策略。
- `final_runtime/traces/*.pt.gz`保留全部30条真实感知/导航transition，解压后逐字节与原PT一致；4份episode边界保留cursor、RNG、轨迹顺序及bootstrap契约。
- sampling为14步、deterministic为16步，各2集；三份8步截断显式保存bootstrap。保留结果、eval_info/trajectory、scope流和trace_summary。
- 最终`verified_validation/VALIDATION.json`保留6次no-Focus、10次受控N=0..4干预、4次复现、概率及梯度重放与冻结审计。
- 关键失败记录包含首次mode串线、fused路径概率误差、CPU/CUDA坐标检查失败及既有全套导入缺失；不把旧attempt计入30条最终分母。
- 不上传官方完整代码、导航权重、场景、采集图像；重复试跑Actor/trace与逐步scope/data副本排除，路径和hash见TASK22_EXCLUDED_FILES。

`TASK22_SOURCE_MAP.json`列出本轮原样/压缩源文件，累计清单为`FILE_MANIFEST.json`及`SOURCE_MAP.json`。历史报告里的FILES_MANIFEST指原机全量产物，不承诺排除文件已上传。

## 核验与重放

```bash
python3.11 tools/verify_task22_release.py

python3.11 tools/verify_task22_release.py \
  --extract-traces

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p test_scope_task22.py
```

解压默认拒绝覆盖已有文件；解压后在仓库根目录调用原episode loader，边界记录中的相对路径才能正确解析。PT仅对可信来源使用`torch.load(weights_only=True, map_location="cpu")`。只支持完成episode的特征/概率重放，不是Habitat任意中途状态快照，也不自动启动PPO。

原GPU报告的1e-5概率门禁与CPU复核误差分别记录，不把跨设备浮点差异冒充新GPU实验。完整模型运行仍需既有第三方依赖及机器路径，不能宣称跨机器开箱即用。

## 本次发布复核

67项SCOPE相关测试及4项协议回归全部通过，输出见`TASK22_SCOPE_REGRESSION.log`和`TASK22_PROTOCOL_REGRESSION.log`。用CPU安全加载初始化Actor和30条压缩trace，逐条检查来源、episode顺序、真实反馈及bootstrap，no-grad与训练模式梯度重放最大误差均为1.9073486328125e-6，未执行optimizer更新，结果见`TASK22_CPU_REPLAY.json`。4份边界分别8/8/6/8条，三份截断都有bootstrap。

新增2项归档测试验证最终证据与错误transition分母拒绝；同时复核此前Stage1/Task4/Task21的归档测试。原根全套的无关导入失败保持历史记录，不声明已修复。

源码版本以最终`verified_validation/model_contract.json`为准：真实闭环结束后`trajectory.py`和`validate.py`修复了CPU/CUDA坐标检查，故与较早`final_runtime/model_contract.json`有两个hash差异。两份历史contract均原样保留，不改旧hash冒充同一版本；模型/Adapter provenance一致。
