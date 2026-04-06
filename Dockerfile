FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps from requirements.txt, skipping the torch line
# (torch is already provided by the base image with CUDA support)
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY scripts/ scripts/

ENTRYPOINT ["python", "scripts/finetune_paligemma.py"]
