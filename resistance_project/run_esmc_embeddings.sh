#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)"

PROJECT_ROOT="${SCRIPT_DIR}"

LAMGEN_ROOT="$(
    cd "${PROJECT_ROOT}/.." && pwd
)"

ESMC_PYTHON="${ESMC_PYTHON:-python}"

ESMC_SCRIPT="${PROJECT_ROOT}/scripts/embedding/esmc_embeddings.py"
SEQUENCE_DIR="${PROJECT_ROOT}/data/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/embeddings"

if [ "$#" -lt 2 ]; then
    echo "Usage:"
    echo "  $0 WT_FASTA_NAME MT_FASTA_NAME [--force]"
    echo
    echo "Example:"
    echo "  $0 P00533_WT.fasta P00533_T790M.fasta --force"
    exit 1
fi

WT_NAME="$1"
MT_NAME="$2"
shift 2

WT_PATH="${SEQUENCE_DIR}/${WT_NAME}"
MT_PATH="${SEQUENCE_DIR}/${MT_NAME}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"

echo
echo "ESM-C WT/MT embedding"
echo "  LAMGEN_ROOT : ${LAMGEN_ROOT}"
echo "  PROJECT_ROOT: ${PROJECT_ROOT}"
echo "  PYTHON      : ${ESMC_PYTHON}"
echo "  WT          : ${WT_PATH}"
echo "  MT          : ${MT_PATH}"
echo "  OUTPUT      : ${OUTPUT_DIR}"
echo

cd "${LAMGEN_ROOT}"

exec "${ESMC_PYTHON}" \
    "${ESMC_SCRIPT}" \
    --wt-fasta "${WT_PATH}" \
    --mt-fasta "${MT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
