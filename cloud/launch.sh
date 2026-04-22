#!/usr/bin/env bash
# Launch a GPU VM on GCE to run PaliGemma fine-tuning.
# Training data is pulled from GCS; checkpoints are uploaded back when done.
# The VM shuts itself down on completion.
#
# Usage:
#   HF_TOKEN=<token> ./cloud/launch.sh [VM_NAME]
#
# Environment variables (all optional except HF_TOKEN):
#   HF_TOKEN       HuggingFace token for downloading PaliGemma (required)
#   GPU_TYPE       GPU accelerator type (default: nvidia-l4)
#   ZONE           GCE zone (default: northamerica-northeast1-b)
#   TAG            Docker image tag (default: latest)
#   EPOCHS         Training epochs (default: 2)
#   BATCH_SIZE     Per-device batch size (default: 4)
#   GRAD_ACCUM     Gradient accumulation steps (default: 4)
#   LR             Learning rate (default: 1e-4)
#   LORA_RANK      LoRA rank (default: 16)
#   USE_4BIT       Load model in 4-bit quantisation: true/false (default: false)
#   PREEMPTIBLE    Use a preemptible VM for lower cost: true/false (default: true)
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set}"

PROJECT=narrative-strategy-campaign
REGION=northamerica-northeast1
ZONE=${ZONE:-northamerica-northeast1-c}
BUCKET=mapclass-training-northeast1
REPO=mapclass
IMAGE=paligemma-trainer
TAG=${TAG:-latest}

GPU_TYPE=${GPU_TYPE:-nvidia-tesla-t4}
EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-1e-4}
LORA_RANK=${LORA_RANK:-16}
USE_4BIT=${USE_4BIT:-false}
PREEMPTIBLE=${PREEMPTIBLE:-true}

VM_NAME=${1:-mapclass-train-$(date +%Y%m%d-%H%M)}
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE}:${TAG}"

FOUR_BIT_FLAG=""
[[ "${USE_4BIT}" == "true" ]] && FOUR_BIT_FLAG="--use-4bit"

PREEMPTIBLE_FLAG=""
[[ "${PREEMPTIBLE}" == "true" ]] && PREEMPTIBLE_FLAG="--preemptible"

# Write startup script to a temp file
STARTUP=$(mktemp)
trap "rm -f ${STARTUP}" EXIT

cat > "${STARTUP}" <<STARTUP_EOF
#!/usr/bin/env bash
set -euo pipefail
exec > /var/log/training.log 2>&1

echo "=== VM startup: \$(date) ==="

# Wait for NVIDIA driver installation (DL VM images install on first boot)
for i in \$(seq 1 30); do
  nvidia-smi &>/dev/null && break
  echo "Waiting for GPU driver... (\${i}/30)"
  sleep 10
done
nvidia-smi

# Install Docker (not included in the pytorch-2-9 DL VM image).
# Wait for any apt/dpkg locks held by cloud-init or unattended-upgrades on first boot.
if ! command -v docker &>/dev/null; then
  echo "Installing docker.io ..."
  while fuser /var/lib/dpkg/lock-frontend &>/dev/null || fuser /var/lib/apt/lists/lock &>/dev/null; do
    echo "Waiting for apt lock ..."
    sleep 5
  done
  for i in \$(seq 1 5); do
    apt-get update && apt-get install -y docker.io python3-venv && break
    echo "apt install failed, retry \${i}/5 ..."
    sleep 10
  done
  command -v docker &>/dev/null || { echo "Failed to install docker"; exit 1; }
fi

# Ensure python3-venv is present even if docker was already installed
dpkg -s python3-venv &>/dev/null || apt-get install -y python3-venv

# Authenticate Docker against Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Pull training image
docker pull ${REGISTRY}

# Sync training data from GCS
mkdir -p /data/toons
gcloud storage rsync gs://${BUCKET}/data/toons /data/toons --recursive

CHECKPOINT_DIR=/checkpoints
mkdir -p \${CHECKPOINT_DIR}/tb

# Start TensorBoard on the host, reading the checkpoint dir mounted into the
# training container. Runs in the background; the docker run below blocks.
echo "=== Starting TensorBoard on port 6006 ==="
# Use the DL VM's /opt/conda Python: it ships with setuptools and is not
# PEP 668 managed, so tensorboard installs cleanly. Avoids Ubuntu 24.04
# system Python 3.12 venv pitfalls (missing setuptools / pkg_resources,
# typing_extensions apt conflicts).
if [[ -x /opt/conda/bin/python3 ]]; then
  TB_PY=/opt/conda/bin/python3
  TB_BIN=/opt/conda/bin/tensorboard
  \${TB_PY} -m pip install --quiet --upgrade tensorboard
else
  echo "WARN: /opt/conda not found on VM; falling back to system venv."
  TB_VENV=/opt/tb-venv
  python3 -m venv \${TB_VENV}
  \${TB_VENV}/bin/pip install --upgrade pip
  # setuptools 81+ split pkg_resources into a separate package; tensorboard
  # 2.20 still imports it from setuptools core. Pin below that boundary.
  \${TB_VENV}/bin/pip install 'setuptools<81' wheel tensorboard
  if ! \${TB_VENV}/bin/python -c "import pkg_resources" 2>/dev/null; then
    echo "ERROR: pkg_resources still not importable. Dumping venv state:"
    \${TB_VENV}/bin/pip list
    ls -la \${TB_VENV}/lib/python3.12/site-packages/ | head -40
  fi
  TB_PY=\${TB_VENV}/bin/python
  TB_BIN=\${TB_VENV}/bin/tensorboard
fi

nohup \${TB_BIN} \\
  --logdir=\${CHECKPOINT_DIR}/tb \\
  --host=0.0.0.0 \\
  --port=6006 \\
  --reload_interval=10 \\
  &>/var/log/tensorboard.log &
TB_PID=\$!
sleep 3
if ! kill -0 \${TB_PID} 2>/dev/null; then
  echo "ERROR: TensorBoard failed to start. Last log lines:"
  tail -n 30 /var/log/tensorboard.log || true
  echo "Continuing without TensorBoard ..."
else
  echo "TensorBoard pid: \${TB_PID}"
fi

echo "=== Starting training: \$(date) ==="

docker run --rm --gpus all \\
  -v /data:/data \\
  -v \${CHECKPOINT_DIR}:/checkpoints \\
  -e HF_TOKEN=${HF_TOKEN} \\
  -e HF_HOME=/data/hf_cache \\
  ${REGISTRY} \\
  --toon-dir /data/toons \\
  --output-dir /checkpoints \\
  --epochs ${EPOCHS} \\
  --batch-size ${BATCH_SIZE} \\
  --grad-accum ${GRAD_ACCUM} \\
  --lr ${LR} \\
  --lora-rank ${LORA_RANK} \\
  ${FOUR_BIT_FLAG}

echo "=== Training complete: \$(date) ==="

# Upload checkpoints to GCS
gcloud storage rsync \${CHECKPOINT_DIR} gs://${BUCKET}/checkpoints/${VM_NAME} --recursive

echo "=== Checkpoints uploaded to gs://${BUCKET}/checkpoints/${VM_NAME}/ ==="
echo "=== Shutting down ==="
shutdown -h now
STARTUP_EOF

# Determine machine type for the GPU
case "${GPU_TYPE}" in
  nvidia-l4)            MACHINE_TYPE=g2-standard-4 ;;
  nvidia-tesla-t4)      MACHINE_TYPE=n1-standard-4 ;;
  nvidia-tesla-p4)      MACHINE_TYPE=n1-standard-4 ;;
  *)                    MACHINE_TYPE=n1-standard-4 ;;
esac

# Detect caller's public IP so the firewall rule can be scoped to a /32.
USER_IP=$(curl -fsS --max-time 10 https://ifconfig.me 2>/dev/null \
       || curl -fsS --max-time 10 https://ipv4.icanhazip.com 2>/dev/null \
       || true)
USER_IP=${USER_IP//[[:space:]]/}
if [[ -z "${USER_IP}" ]]; then
  echo "Could not auto-detect your public IP; set USER_IP=x.x.x.x to override." >&2
  exit 1
fi

FW_RULE="allow-tensorboard-6006"
NET_TAG="paligemma-trainer"
if gcloud compute firewall-rules describe "${FW_RULE}" --project="${PROJECT}" &>/dev/null; then
  echo "Updating firewall rule ${FW_RULE} → source ${USER_IP}/32"
  gcloud compute firewall-rules update "${FW_RULE}" \
    --project="${PROJECT}" \
    --source-ranges="${USER_IP}/32" \
    --quiet
else
  echo "Creating firewall rule ${FW_RULE} (tcp:6006 from ${USER_IP}/32)"
  gcloud compute firewall-rules create "${FW_RULE}" \
    --project="${PROJECT}" \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:6006 \
    --source-ranges="${USER_IP}/32" \
    --target-tags="${NET_TAG}" \
    --quiet
fi

echo "Creating VM: ${VM_NAME}"
echo "  Zone:    ${ZONE}"
echo "  GPU:     ${GPU_TYPE} on ${MACHINE_TYPE}"
echo "  Image:   ${REGISTRY}"
echo "  Epochs:  ${EPOCHS}  LR: ${LR}  LoRA rank: ${LORA_RANK}"
echo ""

gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=${GPU_TYPE},count=1" \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --tags="${NET_TAG}" \
  --metadata="install-nvidia-driver=True" \
  --metadata-from-file="startup-script=${STARTUP}" \
  --scopes=cloud-platform \
  ${PREEMPTIBLE_FLAG}

EXT_IP=$(gcloud compute instances describe "${VM_NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "VM launched. Stream logs with:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} -- tail -f /var/log/training.log"
echo ""
echo "TensorBoard (after ~1–2 min of startup + docker install):"
echo "  http://${EXT_IP}:6006"
echo "  (firewall allows tcp:6006 only from ${USER_IP}/32)"
echo ""
echo "Checkpoints will be saved to:"
echo "  gs://${BUCKET}/checkpoints/${VM_NAME}/"
