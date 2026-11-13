FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /artifact
COPY requirements.txt requirements.lock pyproject.toml README.md ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.lock

COPY . .
RUN python3 -m pip install --no-deps -e .

ENTRYPOINT ["python3", "train_pair.py"]
CMD ["--help"]
