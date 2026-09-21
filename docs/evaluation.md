# 评估口径

模型生成单独落盘，评估脚本只读取原始输出及 gold 标注，不调用在线模型。每行预测保留 id、model、adapter、raw_output、latency_ms、error。同一个文件不能混合多个模型或 Adapter；重复/未知 id 报错，缺失 id 按失败处理。

| 指标 | 定义 |
|---|---|
| JSON parse rate | 原始输出能被 JSON 解析的样本数 / 全部 gold 样本数 |
| Schema valid rate | JSON Schema 与跨字段约束均通过 / 全部样本 |
| Evidence valid rate | 上述校验及证据子串检查通过 / 全部样本 |
| Primary intent accuracy | 完整有效预测中主意图正确 / 全部样本 |
| Language accuracy | 完整有效预测中语言正确 / 全部样本 |
| Intent micro/macro F1 | 对所有提到的意图做集合评分（含条件、否定意图），按标签汇总 macro |
| Status F1 | 对 intent、aspect、status 三元组评分，区分当前/已解决/条件/否定 |
| Slot F1 | 非空槽位的 intent、aspect、字段、字符串值四元组精确匹配 |
| Sentiment set accuracy | 每个样本的 intent、aspect、sentiment 集合完全相同 / 全部样本 |
| Explicit human request | 仅明确人工诉求的 precision/recall/F1，另报全样本 accuracy |
| Evidence F1 | message_id 与 quote 精确匹配；不能替代证据语义支持审查 |
| Conditions/corrections/ad fields | 对相应结构化字段的精确集合匹配，供错误分析 |
| Latency mean / p95 | 所有已返回预测行的生成耗时，包含失败；不含模型首次加载 |

字段 F1 按 TP/FP/FN 计算；不存在正例/预测时相关分数为 null。空字段本身不会贡献 TP；因此始终同时查看全样本有效率和字段 F1，不能单独用槽位 F1 掩盖输出失败。

证据非法时整条结果不计为有效业务预测。JSON 可解析但 Schema 不合法，只提高 JSON parse rate，不提高业务正确率。保留 `.errors.jsonl` 进行人工复核。

`compare` 要求两个报告的 gold SHA256 一致，输出绝对差值。它比较传入报告，不证明实际做过训练；模型来源与实验过程应按实验记录模板留存。初始合成种子不构成真实效果证据。

## 实验记录模板

- 日期、执行人、代码 commit：
- 基座版本与 tokenizer 版本：
- Adapter 与 run_manifest 路径：
- GPU、显存峰值、依赖版本：
- 数据来源、样本数、各语言/标签覆盖、审核方法：
- train/validation/test 摘要及近重复处理：
- 训练配置、loss、是否成功收敛：
- 基线/Adapter 预测与报告路径：
- 困难样本差异与已知失败：
- 推理延迟与后续优化：

无实际记录时保持空白，不填写目标值充当结果。
