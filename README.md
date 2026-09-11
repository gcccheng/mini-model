# Mini Model Lab

这个仓库保存从零训练、评估和部署小型语言模型的完整可复现实验。
This repository contains reproducible experiments for training, evaluating, and deploying small language models from scratch.

## Poetry LLM

`poetry-llm/` 是一个 10.03M 参数的中文古诗字符级 Transformer，支持五言绝句、七言绝句、五言律诗和七言律诗的结构约束生成。

`poetry-llm/` is a 10.03M-parameter character-level Chinese poetry Transformer with structurally constrained generation for four classical forms.

- Final inference model: `poetry-llm/exports/poetry-10m/model.pt`
- API: `poetry-llm/serving/app.py`
- Kubernetes IaC: `poetry-llm/k8s/`
- Rancher Fleet bootstrap: `fleet/poetry-llm-gitrepo.yaml`
- Deployment guide: `poetry-llm/docs/KUBERNETES_DEPLOYMENT.md`

Training datasets, local virtual environments, plaintext secrets, and resumable optimizer checkpoints are intentionally excluded from Git. The final inference model and tokenizer are included.
