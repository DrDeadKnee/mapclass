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

# Authenticate Docker against Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Pull training image
docker pull ${REGISTRY}

# Sync training data from GCS
mkdir -p /data/toons
gcloud storage rsync gs://${BUCKET}/data/toons /data/toons --recursive

echo "=== Starting training: \$(date) ==="

CHECKPOINT_DIR=/checkpoints
mkdir -p \${CHECKPOINT_DIR}

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
  --metadata="install-nvidia-driver=True" \
  --metadata-from-file="startup-script=${STARTUP}" \
  --scopes=cloud-platform \
  ${PREEMPTIBLE_FLAG}

echo ""
echo "VM launched. Stream logs with:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} -- tail -f /var/log/training.log"
echo ""
echo "Checkpoints will be saved to:"
echo "  gs://${BUCKET}/checkpoints/${VM_NAME}/"
