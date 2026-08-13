#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Project paths
# ============================================================

LAMGEN_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)"

ENV_DIR="${LAMGEN_ROOT}/resistance_project/environment"

LAMGEN_YML="${ENV_DIR}/lamgen.yml"
ESMC_YML="${ENV_DIR}/esmc.yml"

VENDOR_DIR="${ENV_DIR}/vendor"
ESMC_WHEEL="${VENDOR_DIR}/transformers-4.57.6-py3-none-any.whl"


# ============================================================
# Basic file checks
# ============================================================

echo
echo "============================================================"
echo "LaMGen WT/MT server setup"
echo "============================================================"
echo "Project root : ${LAMGEN_ROOT}"
echo "Environment  : ${ENV_DIR}"
echo

required_files=(
    "${LAMGEN_YML}"
    "${ESMC_YML}"
    "${ENV_DIR}/environment_yml.sha256"
    "${ESMC_WHEEL}"
    "${VENDOR_DIR}/transformers-4.57.6-py3-none-any.whl.sha256"
)

for file in "${required_files[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "[ERROR] Required file not found:"
        echo "        ${file}"
        exit 1
    fi
done

echo "[PASS] Required environment files found."


# ============================================================
# Verify checksums
# ============================================================

echo
echo "Checking environment YAML files..."

(
    cd "${ENV_DIR}"
    sha256sum -c environment_yml.sha256
)

echo
echo "Checking custom ESM-C Transformers wheel..."

(
    cd "${VENDOR_DIR}"
    sha256sum -c transformers-4.57.6-py3-none-any.whl.sha256
)

echo "[PASS] File integrity checks passed."


# ============================================================
# Find Conda
# ============================================================

if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then

    CONDA_BIN="${CONDA_EXE}"

elif command -v conda >/dev/null 2>&1; then

    CONDA_BIN="$(command -v conda)"

elif [ -x "${HOME}/miniconda3/bin/conda" ]; then

    CONDA_BIN="${HOME}/miniconda3/bin/conda"

elif [ -x "${HOME}/anaconda3/bin/conda" ]; then

    CONDA_BIN="${HOME}/anaconda3/bin/conda"

else

    echo
    echo "[ERROR] Conda was not found."
    echo "Install Miniconda/Anaconda first and run this script again."
    exit 1

fi

echo
echo "[PASS] Conda found:"
echo "       ${CONDA_BIN}"


# ============================================================
# Read environment names from YAML
# ============================================================

LAMGEN_ENV="${LAMGEN_ENV_NAME:-$(
    awk '/^name:/ {print $2; exit}' "${LAMGEN_YML}"
)}"

ESMC_ENV="${ESMC_ENV_NAME:-$(
    awk '/^name:/ {print $2; exit}' "${ESMC_YML}"
)}"

if [ -z "${LAMGEN_ENV}" ] || [ -z "${ESMC_ENV}" ]; then
    echo "[ERROR] Failed to read environment names from YAML."
    exit 1
fi

echo
echo "Environment names:"
echo "  LaMGen : ${LAMGEN_ENV}"
echo "  ESM-C  : ${ESMC_ENV}"


# ============================================================
# Environment creation helper
# ============================================================

create_environment() {

    local env_name="$1"
    local yml_file="$2"

    echo
    echo "------------------------------------------------------------"
    echo "Checking environment: ${env_name}"
    echo "------------------------------------------------------------"

    if "${CONDA_BIN}" run -n "${env_name}" python -V \
        >/dev/null 2>&1; then

        echo "[SKIP] Conda environment already exists: ${env_name}"

    else

        echo "[INFO] Creating Conda environment: ${env_name}"

        "${CONDA_BIN}" env create \
            -n "${env_name}" \
            -f "${yml_file}"

        echo "[PASS] Environment created: ${env_name}"

    fi
}


# ============================================================
# Create environments
# ============================================================

create_environment \
    "${LAMGEN_ENV}" \
    "${LAMGEN_YML}"

create_environment \
    "${ESMC_ENV}" \
    "${ESMC_YML}"


# ============================================================
# Install the custom Transformers build for ESM-C
# ============================================================

echo
echo "------------------------------------------------------------"
echo "Installing custom ESM-C Transformers wheel"
echo "------------------------------------------------------------"

"${CONDA_BIN}" run \
    -n "${ESMC_ENV}" \
    python -m pip install \
    --no-deps \
    --force-reinstall \
    "${ESMC_WHEEL}"

echo "[PASS] Custom Transformers wheel installed."

# ============================================================
# Restore LaMGen generation assets
# ============================================================

ASSET_DIR="${ENV_DIR}/assets"
LAMGEN_ASSET_ARCHIVE="${ASSET_DIR}/lamgen_generation_assets.tar.gz"
LAMGEN_ASSET_SHA256="${ASSET_DIR}/lamgen_generation_assets.tar.gz.sha256"
LAMGEN_FILES_SHA256="${ASSET_DIR}/lamgen_assets.sha256"
echo
echo "------------------------------------------------------------"
echo "Checking LaMGen generation assets"
echo "------------------------------------------------------------"

required_lamgen_assets=(
    "${LAMGEN_ROOT}/Pretrained_model/config.json"
    "${LAMGEN_ROOT}/Pretrained_model/pytorch_model.bin"
    "${LAMGEN_ROOT}/checkpoint/dual/dual_target_ckpt"
    "${LAMGEN_ROOT}/data/torsion_voc.csv"
)

assets_complete=true

for file in "${required_lamgen_assets[@]}"; do
    if [ ! -f "${file}" ]; then
        assets_complete=false
        break
    fi
done

if [ "${assets_complete}" = true ]; then

    echo "[SKIP] LaMGen generation assets already exist."

else

    if [ ! -f "${LAMGEN_ASSET_ARCHIVE}" ]; then
        echo "[ERROR] LaMGen asset archive not found:"
        echo "        ${LAMGEN_ASSET_ARCHIVE}"
        exit 1
    fi

    if [ ! -f "${LAMGEN_ASSET_SHA256}" ]; then
        echo "[ERROR] LaMGen asset checksum not found:"
        echo "        ${LAMGEN_ASSET_SHA256}"
        exit 1
    fi

    echo "[INFO] Verifying LaMGen asset archive..."

    (
        cd "${ASSET_DIR}"
        sha256sum -c lamgen_generation_assets.tar.gz.sha256
    )

    echo "[INFO] Restoring LaMGen generation assets..."

    tar -xzf "${LAMGEN_ASSET_ARCHIVE}" \
        -C "${LAMGEN_ROOT}"

    echo "[PASS] LaMGen generation assets restored."

fi


# ============================================================
# Verify restored LaMGen assets
# ============================================================

for file in "${required_lamgen_assets[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "[ERROR] Required LaMGen asset missing after restore:"
        echo "        ${file}"
        exit 1
    fi
done

echo "[PASS] LaMGen generation assets available."
if [ ! -f "${LAMGEN_FILES_SHA256}" ]; then
    echo "[ERROR] LaMGen asset file checksum not found:"
    echo "        ${LAMGEN_FILES_SHA256}"
    exit 1
fi

echo "[INFO] Verifying LaMGen generation asset files..."

(
    cd "${LAMGEN_ROOT}"
    sha256sum -c resistance_project/environment/assets/lamgen_assets.sha256
)

echo "[PASS] LaMGen generation asset files verified."
# ============================================================
# Prepare fixed ESM-C model snapshot
# ============================================================

ESMC_MODEL_INFO="${ENV_DIR}/models/esmc_6b_model.txt"

if [ ! -f "${ESMC_MODEL_INFO}" ]; then
    echo "[ERROR] ESM-C model information file not found:"
    echo "        ${ESMC_MODEL_INFO}"
    exit 1
fi

ESMC_MODEL_ID="$(
    awk -F= '$1=="model_id" {print $2}' "${ESMC_MODEL_INFO}"
)"

ESMC_MODEL_REVISION="$(
    awk -F= '$1=="revision" {print $2}' "${ESMC_MODEL_INFO}"
)"

if [ -z "${ESMC_MODEL_ID}" ] || [ -z "${ESMC_MODEL_REVISION}" ]; then
    echo "[ERROR] Failed to read ESM-C model ID or revision."
    exit 1
fi

echo
echo "------------------------------------------------------------"
echo "Preparing ESM-C model"
echo "------------------------------------------------------------"
echo "Model    : ${ESMC_MODEL_ID}"
echo "Revision : ${ESMC_MODEL_REVISION}"
echo

"${CONDA_BIN}" run \
    -n "${ESMC_ENV}" \
    python -c \
"from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${ESMC_MODEL_ID}',
    revision='${ESMC_MODEL_REVISION}',
)
print('[PASS] ESM-C model snapshot available')"

# ============================================================
# LaMGen environment test
# ============================================================

echo
echo "------------------------------------------------------------"
echo "Testing LaMGen environment"
echo "------------------------------------------------------------"

cd "${LAMGEN_ROOT}"

"${CONDA_BIN}" run \
    -n "${LAMGEN_ENV}" \
    python -c \
'import torch, numpy, transformers;
from model.lamgen_model import LaMGen_dual;
from utils.bert_tokenizer import ExpressionBertTokenizer;
print("torch:", torch.__version__);
print("torch cuda:", torch.version.cuda);
print("cuda available:", torch.cuda.is_available());
print("numpy:", numpy.__version__);
print("transformers:", transformers.__version__);
print("[PASS] LaMGen core imports successful")'


# ============================================================
# ESM-C environment test
# ============================================================

echo
echo "------------------------------------------------------------"
echo "Testing ESM-C environment"
echo "------------------------------------------------------------"

"${CONDA_BIN}" run \
    -n "${ESMC_ENV}" \
    python -c \
'import torch, numpy, transformers, tokenizers, accelerate;
from transformers import AutoTokenizer, AutoModel;
import transformers.models.esmc;
print("torch:", torch.__version__);
print("torch cuda:", torch.version.cuda);
print("cuda available:", torch.cuda.is_available());
print("numpy:", numpy.__version__);
print("transformers:", transformers.__version__);
print("tokenizers:", tokenizers.__version__);
print("accelerate:", accelerate.__version__);
print("[PASS] ESM-C core imports successful")'


# ============================================================
# Done
# ============================================================

echo
echo "============================================================"
echo "[PASS] Environment setup completed"
echo "============================================================"
echo
echo "LaMGen environment : ${LAMGEN_ENV}"
echo "ESM-C environment  : ${ESMC_ENV}"
echo
