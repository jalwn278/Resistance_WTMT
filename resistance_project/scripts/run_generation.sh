#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)"

PROJECT_ROOT="$(
    cd "${SCRIPT_DIR}/.." && pwd
)"

LAMGEN_ROOT="$(
    cd "${PROJECT_ROOT}/.." && pwd
)"

LAMGEN_PYTHON="${LAMGEN_PYTHON:-python}"

GEN_SCRIPT="${PROJECT_ROOT}/scripts/molecule_generation.py"
EMBEDDING_DIR="${PROJECT_ROOT}/embeddings"

if [ "$#" -lt 2 ]; then
    echo "Usage:"
    echo "  $0 WT_EMBEDDING MT_EMBEDDING [generation options]"
    echo
    echo "Examples:"
    echo "  $0 P00533_WT P00533_T790M"
    echo "  $0 P00533_WT P00533_T790M --batch-size 2 --epochs 3"
    echo "  $0 P00533_WT P00533_T790M --batch-size 2 --epochs 3 --force-overwrite"
    exit 1
fi

wt_name="$1"
mt_name="$2"
shift 2

if [[ "${wt_name}" != *.npy ]]; then
    wt_name="${wt_name}.npy"
fi

if [[ "${mt_name}" != *.npy ]]; then
    mt_name="${mt_name}.npy"
fi

wt_path="${EMBEDDING_DIR}/${wt_name}"
mt_path="${EMBEDDING_DIR}/${mt_name}"

echo
echo "LaMGen WT/MT generation"
echo "  LAMGEN_ROOT : ${LAMGEN_ROOT}"
echo "  PYTHON      : ${LAMGEN_PYTHON}"
echo "  WT          : ${wt_path}"
echo "  MT          : ${mt_path}"
echo

cd "${LAMGEN_ROOT}"

exec "${LAMGEN_PYTHON}" \
    "${GEN_SCRIPT}" \
    --wt-embedding "${wt_path}" \
    --mt-embedding "${mt_path}" \
    "$@"
