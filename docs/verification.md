# 工程验证记录

验证日期：2026-09-21。范围为 Python 工程与数据契约，不包含真实训练或模型效果评测。

## 已通过

- `python -m pytest -q`：32 项测试通过。
- `python -m compileall -q src scripts examples`：Python 文件编译检查通过。
- `python -m pip check`：基础运行/测试环境无依赖冲突。
- 66 条合成种子数据通过 Schema、证据引用和重复检测。
- 数据分组切分、消息格式导出、JSON Schema 导出和 AutoDL CPU 预检可运行。
- 真实 Qwen3-32B tokenizer 的非思考模板验证通过；66 条种子的最大完整训练序列为 2109 tokens，最大输入为 1672 tokens。
- Git 提交排除虚拟环境、权重、生成输出、缓存和真实环境文件。

分词器检查仅下载 tokenizer 文件，使用本机 Transformers 5.12.1；代码显式设置 return_dict=False，避免库版本默认值差异。该检查不代表 train extra 中固定的 Transformers 4.57.6/PEFT/bitsandbytes 组合完成 GPU 验证。

CPU 测试环境：Python 3.12.7、Pydantic 2.13.5、FastAPI 0.141.1、HTTPX 0.28.1、pytest 8.4.2。测试时出现两条 Starlette 测试客户端的上游弃用警告，不影响本次断言结果。

## 测试覆盖

- 多语言结构化字段、精确证据子串、人工请求证据、非法流程执行字段。
- 同源场景分组、重复文本、数据集合泄漏、未审核数据训练门槛。
- prompt/padding 损失屏蔽、真实 EOS 保留、超长样本拒绝。
- 缺失/非法预测进入分母、Schema 与 JSON 指标分离、无正例指标为 null。
- 模型不可用、非法输出、并发忙、长度限制、客户端错误与请求 id 校验。
- OpsPilot 上下文适配和 GrowthTriage 有损单标签优先规则。
- 合成生成器不能修改数据来源、审核状态和场景分组。

## 尚需目标环境验证

- CUDA 依赖组合、32B 权重加载、峰值显存和训练恢复。
- 正式语料审核、训练收敛、独立测试集各项效果指标。
- 真实 Adapter 推理、延迟及 OpsPilot/GrowthTriage 的实际联调。

不存在本次训练生成的 Adapter、训练曲线或已达到的业务准确率。
