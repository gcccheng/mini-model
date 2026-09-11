FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DEVICE=cpu \
    MODEL_PATH=/app/model/model.pt \
    TOKENIZER_PATH=/app/model/tokenizer.json \
    TORCH_NUM_THREADS=2 \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY requirements-serving.txt /app/requirements-serving.txt
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements-serving.txt \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY poetry_llm /app/poetry_llm
COPY serving /app/serving
COPY exports/poetry-10m/model.pt /app/model/model.pt
COPY exports/poetry-10m/tokenizer.json /app/model/tokenizer.json
COPY exports/poetry-10m/manifest.json /app/model/manifest.json

RUN chown -R 10001:10001 /app
USER 10001:10001

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
