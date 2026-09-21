# 数据与标注规范

## 单条记录

`DatasetRecord` 包含 id、scenario_group_id、source、synthetic、review_status、input、target。以 `src/routemind/schema.py` 为唯一契约来源，通过 `routemind schema` 导出 JSON Schema。

`source` 记录真实来源或合成模型；`synthetic` 不得因人工审核而改为 false。`review_status` 仅在真实审核后更新为 approved；不合格样本为 rejected。训练命令默认拒绝未审核训练/验证数据，拒绝 rejected 数据。

## 理解边界

1. 一个反馈可有多个 issue，意图与 aspect 分开。主要意图体现当前诉求。
2. status：unresolved 当前问题，resolved 已解决，conditional 条件成立才提出，negated 明确否定，unknown 信息不足。
3. 每个 issue 保留自己的情绪。广告负面不等于产品所有方面均负面。
4. slots 限定为已在文本出现的业务信息；未知为 null。用户声称支付不等于已核实支付。
5. 缺失信息指有助于理解/处理当前问题的槽位候选，不是命令模型替流程追问。
6. evidence_spans 的 quote 必须是对应 message_id 的连续原文片段。程序验证存在性，语义支持程度仍需审核。
7. request_conditions 保留条件原文；corrections 记录旧值、新值和证据，不直接修改业务状态。
8. explicit_human_request 仅代表明确人工诉求；human_request_evidence 必须来自用户。urgency_signals 是紧急语义信号，不是最终优先级或审批等级。
9. ad_format/ad_placement 可以标准化用户明确提到的广告形式/位置；advertised_claim 和 reported_experience 分别记录广告承诺与用户实际体验。
10. 不从语言推断国家，不编造素材/Campaign ID，不把时间先后标为已证实因果。

## 语言与样本覆盖

主语言 en/es/pt/id/hi/ar；实质混写标 mixed，无法识别标 unknown。品牌或操作系统名称借词本身不构成混写。

66 条种子是初始语料，不覆盖所有标签和自然语言分布。扩充时，应为每个业务标签建立多个独立场景，覆盖正/负/中性、隐含诉求、否定、条件、多轮纠正、未知意图和对抗指令。各语言需由有能力的审核者检查翻译及语义一致性。

## 切分

同一原始场景的翻译、改写、合成扩展必须共享 scenario_group_id。合并全部来源后再做分组切分，严禁逐条随机切分翻译样本。代码检查精确/规范化重复及场景泄漏；语义近重复仍需额外审核。

默认比例以场景组为单位约 70/15/15，并非按行数或语言严格分层。必须检查 manifest 和各集合语言/标签覆盖；初始小语料不能用来报告独立业务效果。正式语料应加入独立来源的人工标注测试集。

## 数据生命周期

采集或生成 → 数据来源记录 → Schema/证据检查 → 人工修订与审核 → 合并/分组切分 → 固定测试集 → 训练/验证。合成 API 的输出会强制标 synthetic=true、review_status=pending，并继承父场景组，生成模型无权自行批准数据。
