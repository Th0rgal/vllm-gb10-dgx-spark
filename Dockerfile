# Custom vLLM build for GB10 (compute 12.1) on DGX Spark
# Base: NVIDIA vLLM container (includes Spark-specific CUDA/Triton/LLVM patches)
FROM nvcr.io/nvidia/vllm:25.12.post1-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build tools for compiling vLLM from source
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake ninja-build python3-dev pkg-config \
      libssl-dev libffi-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /opt/vllm
ARG VLLM_COMMIT=main
RUN git fetch --all && git checkout ${VLLM_COMMIT}

# Install build + runtime deps without overwriting CUDA-enabled torch in base image.
RUN pip install -U pip setuptools wheel setuptools_scm \
    && grep -vE '^(torch|torchaudio|torchvision)' requirements/build.txt > /tmp/req_build.txt \
    && grep -vE '^(torch|torchaudio|torchvision)' requirements/common.txt > /tmp/req_common.txt \
    && grep -vE '^(torch|torchaudio|torchvision|-r )' requirements/cuda.txt > /tmp/req_cuda.txt \
    && pip install -r /tmp/req_build.txt -r /tmp/req_common.txt -r /tmp/req_cuda.txt \
    && VLLM_USE_PRECOMPILED=0 MAX_JOBS=4 \
       pip install --no-build-isolation --no-deps .

EXPOSE 8000
CMD ["bash", "-lc", "vllm --version"]
