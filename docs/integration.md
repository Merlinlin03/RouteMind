# OpsPilot 与 GrowthTriage 接入契约

## OpsPilot

现有 TurnPlanner.predict(state, flows) 返回 TurnPlan，包含 task.commands、knowledge.intents 或 chitchat。RouteMind 输出不是 TurnPlan，不能直接调用 TurnPlan.from_dict(result)。

`to_opspilot_context` 生成两部分：

- routemind_semantics：完整结构化理解结果。
- slot_candidates：只映射现有 payment_order_id/app_version/device_model 白名单字段，排除已解决、否定和条件问题中的槽位。槽位保留证据，由消费方再校验。

接入时将语义结果加入 TurnPlanner 提示词输入，继续由 Planner 结合当前状态和 Flow 定义生成 commands，再由原 validator 与执行器处理。纠正结果不直接覆盖数据库，人工请求也不直接触发 start_flow。不要重复运行两个相互独立的意图分类器并各自修改状态。

本仓库提供客户端与上下文适配，不修改消费项目。消费方需要显式捕获 HTTPStatusError/TimeoutException，决定澄清、重试或人工处理；不得把服务失败变成成功的空标签。

## GrowthTriage

现有 FeedbackOutput 包含 summary、evidence_refs、uncertainties、labels，禁止额外字段。labels 只接受 repeat_exposure/promise_mismatch/other，并要求 id 集合与输入完全一致。

`to_growthtriage_labels` 使用 unresolved issues：

1. 有 ad_promise_mismatch → promise_mismatch。
2. 否则有 ad_frequency → repeat_exposure。
3. 否则 → other。

这是有损单标签兼容映射；混合问题优先标 promise_mismatch，并把完整 Understanding 单独保存为证据，不能丢弃其他 issue。脚本输出的 semantic_evidence 不是现有 FeedbackOutput 字段，不可直接塞入该对象。

summary/evidence_refs 等仍由原 Agent 与证据读取流程生成，RouteMind 不冒充已读取的工具证据。market/platform/campaign_id/creative_id/window_type 等关联键来自业务数据库，不由模型猜测。广告投诉只提供线索，不是 CPI 变化的已证实原因。
