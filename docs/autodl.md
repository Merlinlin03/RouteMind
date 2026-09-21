# AutoDL 环境与执行

## 准备

使用 Linux GPU 实例，先确认驱动与 PyTorch CUDA wheel 兼容。模型权重、依赖缓存、Adapter 和 checkpoint 放到持久化数据盘，配置 `HF_HOME` 指向该盘。不要把权重上传本 Git 仓库。

Qwen3-32B 的 4-bit 权重并不等于训练峰值显存：还包括量化元数据、未量化张量、激活、优化器和临时缓冲。当前 max_length=8192 包含 Schema 提示词和输出，不能承诺某一显卡必然可运行。先运行 tokenize-check，再在正式租卡训练阶段做短跑测量；本次工程交付不执行这些训练。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 根据所租 GPU/镜像选择兼容的 PyTorch CUDA 安装命令，然后：
python -m pip install -e ".[train,test]"
python -m pip check
python -m pytest -q
```

训练库版本集中在 pyproject.toml 的 train extra。该组合提供可复现起点，尚未在本次交付中进行 CUDA 训练验证；不要把依赖安装成功等同于训练验证。

## 数据预检

```bash
routemind validate data/seeds/feedback.jsonl
routemind split data/seeds/feedback.jsonl
python scripts/autodl.py check --allow-unreviewed
```

这三步可在 CPU 环境执行。正式语料请更换输入路径并完成审核，然后去掉 `--allow-unreviewed`。`check` 显示软件、数据和 GPU 状态，不加载模型、不保证显存足够。

安装 tokenizer 依赖后，可以只下载分词器验证模板和长度：

```bash
routemind tokenize-check --config configs/qwen3_32b.json --allow-unreviewed
```

超长时应根据真实长度调整 max_length，或缩短输入/拆分合理上下文；不得静默截断 JSON 答案。

## 后续正式训练

```bash
python scripts/autodl.py train --config configs/qwen3_32b.json
# 中断恢复（使用真实 checkpoint 路径）：
python scripts/autodl.py train --config configs/qwen3_32b.json --resume outputs/qwen3-32b/checkpoint-100
```

当前脚本限定单进程、单 GPU。多 GPU 需要另外配置 FSDP/DeepSpeed，不要直接把 `device_map=auto` 当作分布式训练配置。run_manifest 保存数据摘要、配置、软件版本、GPU 和完成状态；训练开始时为 false，训练结束并保存 Adapter 后才设为 true。

## 推理与验证

模型和 Adapter 必须匹配。配置中的模型名使用相同的 Hugging Face ID 或完全一致的本地基座路径。启动单 worker 推理服务，在 `/health` 确认 ready=true 后再发送请求。

评估基线和微调模型必须使用同一数据文件、提示词和生成设置。记录 max-input-tokens/max-new-tokens 的调整；JSON 校验不通过的生成不能从统计中删除。

参考：[Qwen3 模型说明](https://huggingface.co/Qwen/Qwen3-32B)、[PEFT 量化微调](https://huggingface.co/docs/peft/developer_guides/quantization)、[Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)。
