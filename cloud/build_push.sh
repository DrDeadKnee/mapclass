#!/usr/bin/env bash
# Build the training container and push to Artifact Registry.
# Run from the repo root: ./cloud/build_push.sh [TAG]
set -euo pipefail

PROJECT=narrative-strategy-campaign
REGION=northamerica-northeast1
REPO=mapclass
IMAGE=paligemma-trainer
TAG=${1:-latest}

REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE}"

# Create Artifact Registry repo if it doesn't exist yet
gcloud artifacts repositories describe "${REPO}" \
    --location="${REGION}" --project="${PROJECT}" &>/dev/null \
  || gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT}"

# Authenticate Docker
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build and push using docker compose
docker compose build trainer
docker compose push trainer

echo ""
echo "Pushed: ${REGISTRY}:${TAG}"
