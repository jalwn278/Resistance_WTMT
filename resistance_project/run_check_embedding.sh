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

CHECK_SCRIPT="${PROJECT_ROOT}/scripts/check_embedding.py"
EMBEDDING_DIR="${PROJECT_ROOT}/embeddings"

if [ "$#" -lt 1 ]; then
    echo "Usage:"
    echo "  $0 EMBEDDING_NAME [EMBEDDING_NAME ...]"
    echo
    echo "Examples:"
    echo "  $0 P00533_WT.npy"
    echo "  $0 P00533_WT.npy P00533_T790M.npy"
    echo "  $0 P00533_WT P00533_T790M"
    exit 1
fi

echo
echo "Embedding check"
echo "  LAMGEN_ROOT : ${LAMGEN_ROOT}"
echo "  PROJECT_ROOT: ${PROJECT_ROOT}"
echo "  PYTHON      : ${ESMC_PYTHON}"
echo

for embedding_name in "$@"; do

    if [[ "${embedding_name}" != *.npy ]]; then
        embedding_name="${embedding_name}.npy"
    fi

    embedding_path="${EMBEDDING_DIR}/${embedding_name}"

    echo "Checking embedding:"
    echo "  ${embedding_path}"
    echo

    "${ESMC_PYTHON}" \
        "${CHECK_SCRIPT}" \
        --embedding "${embedding_path}"

    echo
done
