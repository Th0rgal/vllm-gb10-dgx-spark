# vLLM GB10 (DGX Spark) Custom Build

Custom vLLM build for **NVIDIA GB10 / DGX Spark** using CUDA 13.1+ toolchain to enable optimized kernels on SM121 GPUs.

## What this repo provides
- Dockerfile that builds vLLM from source on top of `nvcr.io/nvidia/pytorch:25.12-py3`.
- Keeps the CUDA-enabled PyTorch from the base image (avoids CPU-only wheels).
- Installs vLLM deps without overriding torch/torchaudio/torchvision.

## Build
```bash
docker build -t vllm-gb10:dev .
```

Pin a specific vLLM commit:
```bash
docker build --build-arg VLLM_COMMIT=<commit-or-tag> -t vllm-gb10:dev .
```

## Run (OpenAI-compatible API)
```bash
docker run -d --name step3-vl \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -v /root/.cache/huggingface:/root/.cache/huggingface \
  vllm-gb10:dev \
  vllm serve stepfun-ai/Step3-VL-10B \
    -tp 1 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code
```

## Docker Hub (publish)
```bash
# login
docker login

# tag
docker tag vllm-gb10:dev th0rgal/vllm-gb10-dgx-spark:latest

# push
docker push th0rgal/vllm-gb10-dgx-spark:latest
```

## Notes
- If Triton/Inductor kernels fail on GB10, confirm the CUDA toolchain in the image supports `sm_121`:
  ```bash
  /usr/local/cuda/bin/ptxas --version
  ```
- The base image uses CUDA 13.1 with forward-compatibility.
