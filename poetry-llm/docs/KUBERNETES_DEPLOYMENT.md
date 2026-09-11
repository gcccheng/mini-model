# 在 Talos Kubernetes 上运行古诗模型 / Serving Poetry LLM on Talos Kubernetes

## 1. 架构 / Architecture

中文：FastAPI 在启动时把 `model.pt` 和 `tokenizer.json` 加载到 CPU。Kubernetes 运行一个副本，通过 NodePort 30080 暴露。模型已内置进容器，因此不依赖当前集群尚未安装的 StorageClass。

English: FastAPI loads `model.pt` and `tokenizer.json` into CPU memory at startup. Kubernetes runs one replica and exposes it through NodePort 30080. The model is embedded in the image, avoiding a dependency on a StorageClass, which this cluster does not currently have.

实际集群 / Observed cluster:

- Kubernetes v1.35.8, Talos v1.12.12
- Three `amd64` VMs: `talos-cp1`, `talos-w1`, `talos-w2`
- Each node: 2 vCPU, approximately 2 GB physical RAM
- No IngressClass, StorageClass, or LoadBalancer implementation

## 2. 本机运行 API / Run the API natively on the Mac

中文：本机测试可以使用 MPS 或 CPU。服务依赖需要先安装。

English: Native testing can use MPS or CPU. Install the service dependencies first.

```bash
cd /Users/lillian/git/mini-model/poetry-llm
.venv/bin/pip install -r requirements-serving.txt

MODEL_DEVICE=mps \
MODEL_PATH=exports/poetry-10m/model.pt \
TOKENIZER_PATH=exports/poetry-10m/tokenizer.json \
POETRY_API_KEY="$(<.secrets/poetry-api-key)" \
.venv/bin/python -m uvicorn serving.app:app --host 127.0.0.1 --port 8080
```

另一个终端 / In another terminal:

```bash
curl -s http://127.0.0.1:8080/readyz

curl -s http://127.0.0.1:8080/v1/generate \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $(<.secrets/poetry-api-key)" \
  -d '{
    "form": "五言绝句",
    "dynasty": "唐朝",
    "author": "李白",
    "title": "月夜",
    "samples": 1
  }'
```

## 3. 构建 amd64 镜像 / Build the amd64 image

中文：MacBook 是 arm64，而 Talos 节点是 amd64，因此必须指定 `--platform linux/amd64`。GitHub Actions 会自动构建并推送 GHCR 镜像，再把 Deployment 固定到不可变 digest。

English: The MacBook is arm64 while the Talos nodes are amd64, so the build must specify `--platform linux/amd64`. GitHub Actions builds and publishes the GHCR image, then pins the Deployment to its immutable digest.

```bash
cd /Users/lillian/git/mini-model/poetry-llm
git push origin main
gh run watch
```

Workflow publishes this image and updates the manifest automatically:

```yaml
image: ghcr.io/gcccheng/poetry-llm@sha256:...
```

## 4. API Key / API key

中文：真实 key 只保存在 `.secrets/poetry-api-key`，该目录被 Git 忽略。Deployment 只保存 SHA-256 哈希，因此 GitHub 中没有可用于调用 API 的秘密。

English: The raw key exists only in `.secrets/poetry-api-key`, which Git ignores. The Deployment stores only its SHA-256 verifier, so GitHub contains no credential that can call the API.

```bash
POETRY_API_KEY="$(<.secrets/poetry-api-key)"
```

把本地 key 复制到密码管理器；不要提交 `.secrets/`。
Copy the local key into a password manager; never commit `.secrets/`.

## 5. Rancher Fleet GitOps 部署 / Rancher Fleet GitOps deployment

中文：应用程序资源不使用手工 `kubectl apply -k`。唯一一次 bootstrap 是把仓库中的 `fleet/poetry-llm-gitrepo.yaml` 应用到 Rancher 上游；之后 Fleet 持续将 `poetry-llm/k8s` 同步到 Talos 集群。

English: Application resources are not managed with manual `kubectl apply -k`. The only bootstrap operation applies the repository's `fleet/poetry-llm-gitrepo.yaml` to the upstream Rancher cluster. Fleet then continuously reconciles `poetry-llm/k8s` into the Talos cluster.

```bash
cd /Users/lillian/git/mini-model
kubectl --context rancher apply -f fleet/poetry-llm-gitrepo.yaml

kubectl --context rancher -n fleet-default get gitrepo poetry-llm
kubectl --context talos -n poetry-llm get pods,svc -o wide
kubectl --context talos -n poetry-llm rollout status deployment/poetry-api --timeout=180s
kubectl --context talos -n poetry-llm logs deployment/poetry-api
```

GHCR package must be public for credential-free Fleet deployment.

## 6. 访问 API / Call the API

NodePort 可以从任意节点 IP 访问：The NodePort is reachable through any node IP:

```bash
curl -s http://192.168.0.192:30080/readyz

curl -s http://192.168.0.192:30080/v1/generate \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $(<.secrets/poetry-api-key)" \
  -d '{
    "form": "七言绝句",
    "dynasty": "唐朝",
    "author": "李商隐",
    "title": "月夜",
    "samples": 2,
    "temperature": 0.7
  }'
```

API 文档 / Interactive API documentation:

```text
http://192.168.0.192:30080/docs
```

## 7. 更新与回滚 / Updates and rollback

中文：重新训练后，替换 `exports/poetry-10m/` 中的推理文件并提交。GitHub Actions 构建镜像、推送 GHCR、把 Deployment 更新到不可变 digest 并提交；Fleet 检测提交后自动部署。

English: After retraining, replace the inference files under `exports/poetry-10m/` and commit. GitHub Actions builds the image, pushes it to GHCR, pins the Deployment to the immutable digest, and commits that change. Fleet then deploys the new Git state.

```bash
git add poetry-llm/exports/poetry-10m
git commit -m "model: publish a new poetry checkpoint"
git push
gh run watch
```

回滚 / Roll back:

```bash
git revert <commit-that-updated-the-image-digest>
git push
```

## 8. 安全与公网访问 / Security and public access

中文：NodePort 当前只适合可信局域网测试。API key 在普通 HTTP 上传输时不是加密的。若要让互联网用户访问，需要先部署 Ingress/Gateway 和 TLS，或通过受保护的反向代理/隧道暴露；不要直接在路由器上把 30080 映射到公网。

English: NodePort is suitable only for trusted LAN testing here. An API key sent over plain HTTP is not encrypted. Before Internet exposure, install an Ingress/Gateway with TLS or use a protected reverse proxy/tunnel. Do not directly port-forward 30080 from the router to the Internet.

## 9. 资源说明 / Resource notes

中文：Deployment 请求 250m CPU/512Mi 内存，限制为 1.5 CPU/1200Mi 内存，并调度到 amd64 节点。当前工作节点约有 1.46 GiB allocatable RAM，因此先保持单副本。扩容前建议把工作节点内存增加到至少 4 GB。

English: The Deployment requests 250m CPU/512Mi and limits usage to 1.5 CPU/1200Mi on amd64 nodes. Each worker currently has only about 1.46 GiB allocatable RAM, so keep one replica initially. Increase worker memory to at least 4 GB before scaling replicas or adding ingress components.
