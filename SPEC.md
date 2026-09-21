# RouteMind v0.1 实施规格

状态：已获用户授权实施 Python 工程并上传私有仓库；实际 GPU 训练不属于本次交付。

## 目标与范围
构建独立的 Qwen3-32B QLoRA 多语言反馈理解工程，服务 OpsPilot 与 GrowthTriage。交付数据流程、训练脚本、评估、推理 API、适配示例、测试和 AutoDL 文档，并上传 GitHub。
不修改现有消费项目；不替代 TurnPlanner、Flow YAML、Workflow；不执行退款或广告预算动作。

## 架构
数据导入/合成 → 校验与人工审核标记 → 场景分组切分 → Qwen3 模板格式化 → QLoRA Adapter 训练 → 独立测试集生成与评估 → FastAPI 推理 → 消费方适配示例。
核心依赖：Python、Transformers、PEFT、bitsandbytes、PyTorch、FastAPI、Pydantic、JSON Schema。运行时依赖与 GPU 训练依赖分离。

## 数据模型
- FeedbackRequest：request_id、messages（消息 id、role、content）；用户消息必需，业务元数据作为外部事实独立传递。
- FeedbackUnderstanding：schema_version、language、primary_intent、issues、request_conditions、corrections、explicit_human_request、urgency_signals、missing_information。
- Issue：intent、aspect、sentiment、status、slots、evidence_spans；slots 使用限定字段和 nullable 值，证据引用 message_id 与原文片段。
- 广告字段：ad_format、ad_placement、advertised_claim、reported_experience；未知值不补猜测。
- Condition：关联意图、条件原文及证据；Correction：被纠正字段、旧值、新值及证据。
- DatasetRecord：id、scenario_group_id、语言、输入、目标输出、来源、synthetic 标记、review_status；正式训练须显式指定允许使用的数据状态。
- PredictionRecord：样本 id、模型标识、adapter 标识、原始输出、解析/Schema/证据校验状态、延迟；保留失败输出用于评估。
业务标签覆盖订阅扣费、退款、Premium 权益、崩溃/性能、广告体验、隐私权限、登录、其他/未知；最终枚举集中维护，并导出 JSON Schema。
模型不输出 next_node、execute_action、Campaign 归属推断或确定性因果结论。

## API 契约
- GET /health：服务状态、后端类型与模型是否就绪，不把存活等同于模型可用。
- GET /schema：返回版本化输出 JSON Schema。
- POST /v1/understand：接收 FeedbackRequest，返回 request_id、model、adapter、结构化结果。
- 无效请求返回 422；模型未就绪返回 503；推理或输出校验失败返回明确错误，禁止静默返回成功或规则结果。
- 服务仅提供真实模型后端，加载基座及可选 Adapter；不提供规则回退或样本回放后端。
- 模型调用设置并发限制，客户端设置超时；服务配置请求长度及生成 token 上限。

## 数据与训练
六语言：英语、西语、葡语、印尼语、印地语、阿语。种子样本包含多意图、否定、条件退款、槽位纠正、混写及广告承诺差异；标注为合成、待审核。
提供兼容聊天 API 的可配置合成生成脚本，不默认发起收费调用，不将凭据写入仓库。
按 scenario_group_id 分组划分 train/validation/test，同源翻译与改写不得跨集合；检测重复 id、重复文本及不合法证据。
采用 Qwen3 非思考模板；仅 assistant 目标输出参与损失，padding 与 prompt 屏蔽；超长样本报错/统计，不静默截掉目标 JSON。
默认正式配置 Qwen/Qwen3-32B，4-bit NF4 QLoRA，梯度检查点、梯度累积；模型名称可配置。
保存随机种子、配置、数据摘要、依赖版本和 Adapter；只用 validation 做训练期间评估，test 独立保留。

## 评估
分别计算原始 JSON 可解析率、Schema 合规率、主要意图准确率、多意图 F1、槽位 F1、情绪准确率、人工诉求识别 precision/recall、证据有效率及延迟。
提供分语言统计、错误样本导出、同数据集基线与 Adapter 对比；无效输出计入失败，禁止从分母剔除。
区分 explicit_human_request 与最终人工升级策略，不将前者指标命名为整个系统升级召回率。
种子数据只能用于流程测试，不生成或预填达标指标。

## 文件结构
- pyproject.toml、README.md、.gitignore、.env.example：安装、配置和说明。
- src/routemind/：Schema、数据校验/切分/格式化、训练、生成与评估、推理后端、API、客户端。
- configs/：32B 正式训练配置。
- data/seeds/：六语言合成种子记录。
- scripts/：数据流程、AutoDL 环境检查、训练、评估与服务启动。
- examples/：OpsPilot 与 GrowthTriage 适配示例，保留消费方职责边界。
- tests/：Schema/证据、分组切分、防泄漏、目标 token masking、指标分母、API 错误、适配契约测试。
- docs/：数据标注规范、AutoDL 指南、架构边界、实验记录模板和设计决策。
- THIRD_PARTY_NOTICES.md：记录参考项目；复用其代码时附 MIT 版权声明。

## 验收标准
1. CPU 环境可安装基础依赖，种子校验、切分、格式化和离线评估可运行。
2. 测试覆盖错误路径和关键数据泄漏边界；测试目录中的替身验证 HTTP API 与两个适配契约，业务代码不提供模拟后端。
3. 训练与推理使用一致模板；GPU 训练提供环境预检和清晰启动步骤。
4. GitHub 提交不包含密钥、缓存、模型权重、训练输出或无关项目；推送后核对远端提交。

## 遗留事项
- 用户后续在租卡环境进行真实训练；显存需求通过实际配置预检及短跑确认。
- 多语言种子需人工审核，不能替代独立真实测试集。
- GitHub 登录已在允许联网的环境验证；发布到 Merlinlin03/RouteMind 私有仓库。
