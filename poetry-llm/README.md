# Poetry LLM

一个从零训练的 10.03M 参数中文古诗字符级 Transformer，以及完整的本地推理、HTTP API、容器和 Rancher Fleet/Talos Kubernetes IaC。

A 10.03M-parameter character-level Chinese poetry Transformer trained from scratch, with native inference, an HTTP API, container packaging, and Rancher Fleet/Talos Kubernetes IaC.

## 能力 / Capabilities

- 五言绝句：4 句 × 5 字 / five-character quatrain: 4 × 5
- 七言绝句：4 句 × 7 字 / seven-character quatrain: 4 × 7
- 五言律诗：8 句 × 5 字 / five-character regulated verse: 8 × 5
- 七言律诗：8 句 × 7 字 / seven-character regulated verse: 8 × 7
- Form-aware checkpoint: step 5,000
- Strict-form test perplexity: 30.88

结构约束保证句数和字数，但不保证平仄、押韵、对仗或文学质量。
Structural decoding guarantees line count and width, but not tonal patterns, rhyme, parallelism, or literary quality.

## 最终模型 / Final model

```text
exports/poetry-10m/model.pt
exports/poetry-10m/tokenizer.json
exports/poetry-10m/manifest.json
```

## 命令行生成 / CLI generation

```bash
.venv/bin/python scripts/generate_form.py \
  --checkpoint exports/poetry-10m/model.pt \
  --tokenizer exports/poetry-10m/tokenizer.json \
  --form 五言绝句 --dynasty 唐朝 --author 李白 --title 月夜 --samples 3
```

## HTTP API

```bash
.venv/bin/pip install -r requirements-serving.txt

MODEL_DEVICE=mps \
MODEL_PATH=exports/poetry-10m/model.pt \
TOKENIZER_PATH=exports/poetry-10m/tokenizer.json \
POETRY_API_KEY="$(<.secrets/poetry-api-key)" \
.venv/bin/python -m uvicorn serving.app:app --host 127.0.0.1 --port 8080
```

```bash
curl http://127.0.0.1:8080/v1/generate \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $(<.secrets/poetry-api-key)" \
  -d '{"form":"七言绝句","dynasty":"唐朝","author":"李商隐","title":"月夜","samples":1}'
```

## Repository layout

```text
poetry_llm/                  model and tokenizer implementation
scripts/                     data, training, evaluation, generation, IaC helpers
serving/                     FastAPI application
exports/poetry-10m/          final inference model included in Git
k8s/                         Kubernetes desired state reconciled by Fleet
configs/                     model and training configurations
docs/                        bilingual tutorials and deployment documentation
```

## IaC / GitOps

GitHub Actions builds `linux/amd64`, publishes `ghcr.io/gcccheng/poetry-llm`, and pins `k8s/deployment.yaml` to the immutable image digest. Rancher Fleet reconciles that directory to the Talos cluster.

See `docs/KUBERNETES_DEPLOYMENT.md` for the complete bilingual procedure.

## Secret policy

No plaintext API key, GitHub token, kubeconfig, SSH key, or password belongs in Git. The API key stays in `.secrets/poetry-api-key`; Kubernetes stores only its SHA-256 verifier. Training data, `.venv`, and resumable optimizer checkpoints also remain local and ignored.
