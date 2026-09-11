# 古诗小模型教程 5–11 / Tiny Poetry LLM Tutorials 5–11

本系列在 Apple Silicon MacBook 上从零训练约 10M 参数的字符级 decoder-only Transformer。
This series trains an approximately 10M-parameter character-level decoder-only Transformer from scratch on an Apple Silicon MacBook.

> 前置步骤 / Prerequisite: 教程 4 的字符 tokenizer 必须先建立。词表只从训练集构建，避免验证集和测试集信息泄漏。Tutorial 4's character tokenizer must be built first. Its vocabulary is derived only from the training split to prevent validation/test leakage.

## 教程 5：把文本编码为训练 token / Encode text into training tokens

**中文：** 每首诗被格式化为“朝代、作者、题目、正文”四个字段，并在首尾加入 `<bos>` 和 `<eos>`。字符 ID 以 `uint16` 连续写入 `.bin` 文件。训练时使用内存映射读取，不需要一次把全部数据加载到内存。验证集或测试集里训练词表未见的字符会变成 `<unk>`。

**English:** Each poem is formatted into dynasty, author, title, and body fields, then wrapped with `<bos>` and `<eos>`. Character IDs are packed into `.bin` files as `uint16`. Memory mapping lets training sample the corpus without loading it all into RAM. Characters unseen in the training vocabulary become `<unk>` in validation or test data.

运行 / Run:

```bash
.venv/bin/python scripts/prepare_tokens.py \
  --data-dir data/processed --tokenizer artifacts/tokenizer.json \
  --output-dir data/tokens --report reports/token_data_report.txt
```

产物 / Outputs: `data/tokens/train.bin`, `validation.bin`, `test.bin`, `meta.json`.

## 教程 6：构建约 10M 参数 Transformer / Build the ~10M Transformer

**中文：** 模型是 GPT 风格的因果语言模型：字符嵌入、位置嵌入、10 个 pre-norm Transformer block、4 头自注意力和前馈网络。输入当前位置只能关注当前位置及之前的 token。输入嵌入与输出分类层共享权重，减少参数和内存。

**English:** The model is a GPT-style causal language model with character and positional embeddings, ten pre-norm Transformer blocks, four-head self-attention, and feed-forward networks. Each position can attend only to itself and earlier tokens. Input embeddings and the output classifier share weights to reduce parameters and memory.

运行 / Run:

```bash
.venv/bin/python scripts/model_summary.py \
  --config configs/model_10m.json --tokenizer artifacts/tokenizer.json \
  --report reports/model_summary.txt
```

模型结构 / Architecture: context 256, width 256, 10 layers, 4 heads, approximately 10M parameters.

## 教程 7：配置并验证 M5 训练 / Configure and validate M5 training

**中文：** `train_m5.json` 保存 batch size、学习率、warm-up、梯度裁剪、验证频率等超参数。设备设置为 `auto`：优先使用 Apple MPS，其次 CUDA，最后 CPU。正式训练前运行少量 step，确认前向、反向、优化器和 checkpoint 全部可用。

**English:** `train_m5.json` stores hyperparameters such as batch size, learning rate, warm-up, gradient clipping, and evaluation frequency. Device selection is `auto`: Apple MPS first, then CUDA, then CPU. A short smoke run verifies forward/backward passes, optimizer updates, and checkpoint writing before the full run.

冒烟测试 / Smoke test:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model_10m.json --train-config configs/train_m5.json \
  --tokenizer artifacts/tokenizer.json --tokens-dir data/tokens \
  --output-dir checkpoints/smoke --max-steps 2
```

## 教程 8：从零预训练 / Pretrain from scratch

**中文：** 模型使用 next-token prediction：给定前面的字符，预测下一个字符。默认执行 500 个优化 step，batch size 为 16、序列长度为 256，相当于观察约 205 万个 token。每 100 step 在验证集评估并保存 checkpoint。`best.pt` 是验证损失最低的版本，`last.pt` 是最后一步。

**English:** Training uses next-token prediction: given preceding characters, predict the next one. The default run performs 500 optimizer steps with batch size 16 and sequence length 256, exposing the model to about 2.05 million tokens. Validation and checkpoints run every 100 steps. `best.pt` is the checkpoint with the lowest validation loss; `last.pt` is the final step.

运行 / Run:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model_10m.json --train-config configs/train_m5.json \
  --tokenizer artifacts/tokenizer.json --tokens-dir data/tokens \
  --output-dir checkpoints/poetry-10m
```

这只是教学训练，不代表模型已经充分收敛。可以用 `--resume checkpoints/poetry-10m/last.pt` 延长训练，但应同时提高配置中的 `max_steps`。
This is an educational run, not a claim of full convergence. Training can be extended with `--resume checkpoints/poetry-10m/last.pt` after increasing `max_steps` in the configuration.

## 教程 9：独立测试集评估 / Evaluate on the held-out test set

**中文：** 训练完成后只用测试集做一次最终评估。脚本报告交叉熵 loss 和 perplexity。较低通常更好，但 perplexity 不能判断格律、意境或事实正确性，因此仍需要阅读生成样本。

**English:** After training, the held-out test set is used for one final evaluation. The script reports cross-entropy loss and perplexity. Lower is generally better, but perplexity cannot measure poetic meter, imagery, or factual correctness, so generated samples still require human inspection.

运行 / Run:

```bash
.venv/bin/python scripts/evaluate.py \
  --checkpoint checkpoints/poetry-10m/best.pt \
  --tokens data/tokens/test.bin --output reports/test_metrics.json
```

## 教程 10：按提示生成古诗 / Generate poetry from a prompt

**中文：** 输入与训练格式保持一致。temperature 控制随机程度，`top-k` 把候选限制在概率最高的若干字符。小模型可能重复、混合作者风格或虚构内容，这是预期限制，不应把输出当作真实古诗来源。

**English:** Prompts use the same structure as training records. Temperature controls randomness, while `top-k` restricts sampling to the most likely characters. A small model may repeat itself, blend author styles, or fabricate content; its output must not be treated as an authentic historical source.

运行 / Run:

```bash
.venv/bin/python scripts/generate.py \
  --checkpoint checkpoints/poetry-10m/best.pt --tokenizer artifacts/tokenizer.json \
  --prompt $'朝代：唐朝\n作者：李白\n题目：秋夜\n正文：' \
  --samples 3 --output reports/generated_samples.txt
```

## 教程 11：导出并保存模型 / Export and preserve the model

**中文：** 导出包只保留推理需要的权重、模型配置和 tokenizer，不包含 AdamW 优化器状态，因此体积小于训练 checkpoint。`manifest.json` 保存格式版本、训练 step 和 SHA-256，可用于验证文件没有损坏。

**English:** The export bundle keeps only inference weights, model configuration, and tokenizer. It omits AdamW optimizer state, so it is smaller than a training checkpoint. `manifest.json` records the format version, training step, and SHA-256 hashes for integrity verification.

运行 / Run:

```bash
.venv/bin/python scripts/export_model.py \
  --checkpoint checkpoints/poetry-10m/best.pt --tokenizer artifacts/tokenizer.json \
  --output-dir exports/poetry-10m
```

产物 / Outputs: `model.pt`, `tokenizer.json`, `manifest.json`, `README.md`.

## 学习结论 / Learning outcome

**中文：** 完成后，你已经走过一个最小 LLM 的完整生命周期：数据编码 → 模型定义 → 设备验证 → 预训练 → 测试评估 → 随机采样 → 推理导出。它的规模和数据量无法与通用大模型相比，但核心训练机制相同。

**English:** At completion, you have covered the full lifecycle of a minimal LLM: data encoding → model definition → device validation → pretraining → held-out evaluation → stochastic sampling → inference export. Its scale and dataset are not comparable to general-purpose LLMs, but the core training mechanism is the same.
