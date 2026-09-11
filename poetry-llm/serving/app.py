from __future__ import annotations

import hmac
import hashlib
import os
import secrets
import threading
import time
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Optional

import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from poetry_llm.model import GPT, GPTConfig
from poetry_llm.tokenizer import CharTokenizer


MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model/model.pt"))
TOKENIZER_PATH = Path(os.getenv("TOKENIZER_PATH", "/app/model/tokenizer.json"))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
FORM_SPECS = {
    "五言绝句": (4, 5),
    "七言绝句": (4, 7),
    "五言律诗": (8, 5),
    "七言律诗": (8, 7),
}
PoeticForm = Literal["五言绝句", "七言绝句", "五言律诗", "七言律诗"]


class GenerateRequest(BaseModel):
    form: PoeticForm = "五言绝句"
    dynasty: str = Field(default="唐朝", min_length=1, max_length=16)
    author: str = Field(default="佚名", min_length=1, max_length=32)
    title: str = Field(default="月夜", min_length=1, max_length=64)
    samples: int = Field(default=1, ge=1, le=5)
    temperature: float = Field(default=0.70, gt=0.0, le=2.0)
    top_k: int = Field(default=20, ge=1, le=200)
    repetition_penalty: float = Field(default=1.20, ge=1.0, le=2.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)


class PoemSample(BaseModel):
    seed: int
    text: str
    lines: list[str]


class GenerateResponse(BaseModel):
    model_step: int
    form: PoeticForm
    dynasty: str
    author: str
    title: str
    samples: list[PoemSample]
    elapsed_ms: float
    disclaimer: str


def is_han_character(token: str) -> bool:
    if len(token) != 1:
        return False
    name = unicodedata.name(token, "")
    return "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name


class ModelService:
    def __init__(self, model_path: Path, tokenizer_path: Path, device_name: str):
        torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "2"))))
        self.device = torch.device(device_name)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = GPT(GPTConfig(**checkpoint["model_config"]))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.tokenizer = CharTokenizer.load(tokenizer_path)
        self.model_step = int(checkpoint.get("step", checkpoint.get("training_step", 0)))
        self.allowed_ids = [
            index
            for index, token in enumerate(self.tokenizer.tokens)
            if is_han_character(token)
        ]
        self.comma_id = self.tokenizer.token_to_id["，"]
        self.stop_id = self.tokenizer.token_to_id["。"]
        self.lock = threading.Lock()

    def _unsupported_prompt_chars(self, prompt: str) -> list[str]:
        return sorted(
            {char for char in prompt if self.tokenizer.token_to_id.get(char) is None}
        )

    @torch.inference_mode()
    def _sample_character(
        self,
        ids: torch.Tensor,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
    ) -> torch.Tensor:
        context = ids[:, -self.model.config.block_size :]
        logits, _ = self.model(context)
        logits = logits[:, -1, :] / temperature
        if repetition_penalty > 1:
            for token_id in set(ids[0].tolist()):
                value = logits[0, token_id]
                logits[0, token_id] = (
                    value / repetition_penalty if value > 0 else value * repetition_penalty
                )
        constrained = torch.full_like(logits, -float("inf"))
        constrained[:, self.allowed_ids] = logits[:, self.allowed_ids]
        values, _ = torch.topk(constrained, min(top_k, len(self.allowed_ids)))
        constrained[constrained < values[:, [-1]]] = -float("inf")
        probabilities = F.softmax(constrained, dim=-1)
        return torch.multinomial(probabilities, 1)

    def generate(self, request: GenerateRequest) -> tuple[list[PoemSample], int]:
        line_count, width = FORM_SPECS[request.form]
        prompt = (
            f"体裁：{request.form}\n"
            f"朝代：{request.dynasty}\n"
            f"作者：{request.author}\n"
            f"题目：{request.title}\n"
            "正文："
        )
        unsupported = self._unsupported_prompt_chars(prompt)
        if unsupported:
            joined = "".join(unsupported)
            raise ValueError(f"prompt contains characters absent from tokenizer: {joined}")
        base_seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
        results: list[PoemSample] = []
        with self.lock:
            for sample_index in range(request.samples):
                sample_seed = base_seed + sample_index
                torch.manual_seed(sample_seed)
                prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
                ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                lines = []
                for line_index in range(line_count):
                    characters = []
                    for _ in range(width):
                        next_id = self._sample_character(
                            ids,
                            request.temperature,
                            request.top_k,
                            request.repetition_penalty,
                        )
                        ids = torch.cat((ids, next_id), dim=1)
                        characters.append(self.tokenizer.decode(next_id[0].tolist()))
                    punctuation_id = self.comma_id if line_index % 2 == 0 else self.stop_id
                    punctuation = "，" if line_index % 2 == 0 else "。"
                    ids = torch.cat(
                        (
                            ids,
                            torch.tensor(
                                [[punctuation_id]], dtype=torch.long, device=self.device
                            ),
                        ),
                        dim=1,
                    )
                    lines.append("".join(characters) + punctuation)
                results.append(
                    PoemSample(seed=sample_seed, text="".join(lines), lines=lines)
                )
        return results, base_seed


service: Optional[ModelService] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = ModelService(MODEL_PATH, TOKENIZER_PATH, MODEL_DEVICE)
    yield
    service = None


app = FastAPI(
    title="Poetry LLM API",
    version="1.0.0",
    description="A small educational model for structurally constrained Chinese poetry generation.",
    lifespan=lifespan,
)


def require_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None,
) -> None:
    expected = os.getenv("POETRY_API_KEY", "")
    expected_hash = os.getenv("POETRY_API_KEY_SHA256", "")
    valid = True
    if expected_hash:
        supplied_hash = hashlib.sha256((x_api_key or "").encode("utf-8")).hexdigest()
        valid = hmac.compare_digest(supplied_hash, expected_hash)
    elif expected:
        valid = x_api_key is not None and hmac.compare_digest(x_api_key, expected)
    if not valid:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readiness() -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {"status": "ready", "model_step": service.model_step}


@app.get("/v1/model", dependencies=[Depends(require_api_key)])
def model_info() -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {
        "name": "poetry-10m-form",
        "parameters": service.model.parameter_count(),
        "training_step": service.model_step,
        "device": str(service.device),
        "supported_forms": list(FORM_SPECS),
    }


@app.post(
    "/v1/generate",
    response_model=GenerateResponse,
    dependencies=[Depends(require_api_key)],
)
def generate(request: GenerateRequest) -> GenerateResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    started = time.perf_counter()
    try:
        samples, _ = service.generate(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return GenerateResponse(
        model_step=service.model_step,
        form=request.form,
        dynasty=request.dynasty,
        author=request.author,
        title=request.title,
        samples=samples,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        disclaimer="AI-generated draft; not an authentic historical work.",
    )
