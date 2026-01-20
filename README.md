# vLLM GB10 / DGX Spark

A custom vLLM Docker build for **NVIDIA GB10 (DGX Spark)** that keeps the CUDA‑enabled PyTorch from NVIDIA’s 25.12 image and builds vLLM from source so optimized kernels can work on **SM121** GPUs.

## Why this exists
- The stock vLLM images don’t fully support GB10 kernel compilation.
- This image uses **CUDA 13.1 toolchain** (from NVIDIA PyTorch 25.12) and builds vLLM from source without overwriting torch.

## Quick start (Docker)
Pull (once published):
```bash
docker pull th0rgal/vllm-gb10-dgx-spark:latest
```

Run:
```bash
docker run -d --name step3-vl \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -v /root/.cache/huggingface:/root/.cache/huggingface \
  th0rgal/vllm-gb10-dgx-spark:latest \
  vllm serve stepfun-ai/Step3-VL-10B \
    -tp 1 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code
```

## Build locally
```bash
docker build -t vllm-gb10:dev .
```

Pin a vLLM commit:
```bash
docker build --build-arg VLLM_COMMIT=<commit-or-tag> -t vllm-gb10:dev .
```

## Notes
- Base image: `nvcr.io/nvidia/pytorch:25.12-py3` (CUDA 13.1).
- The Dockerfile installs vLLM deps **without** replacing torch/torchaudio/torchvision.
