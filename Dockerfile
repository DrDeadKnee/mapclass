FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps from requirements.txt, skipping the torch line
# (torch is already provided by the base image with CUDA support)
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY scripts/ scripts/

ENTRYPOINT ["python", "scripts/finetune_paligemma.py"]
