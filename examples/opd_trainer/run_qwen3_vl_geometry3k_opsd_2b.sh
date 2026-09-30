#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# On-policy self-distillation on Geometry3K with privileged teacher hints.
#
#   MODE=baseline : vanilla OPSD
#   MODE=sopd     : vanilla OPSD + TPC + PA
#
# The frozen teacher and the trainable student start from exactly the same
# checkpoint. The student sees the ordinary prompt; the teacher additionally
# sees the `teacher_prompt` hint stored in geo3k_train_R.parquet.
MODE="${MODE:-baseline}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"

MODEL_NAME="${MODEL_NAME:-Qwen3-VL-2B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/path/to/2b/model}"

case "${MODE}" in
    baseline)
        export TPC_ENABLED=False
        export TPC_TEACHER_GATING_ENABLED=False
        export PA_ENABLED=False
        DEFAULT_EXP_NAME="OPSD_geometry3k_S2B_T2B_privilegedHint_baseline_${TOTAL_EPOCHS}epochs"
        ;;
    sopd)
        export TPC_ENABLED=True
        export TPC_TEACHER_GATING_ENABLED=True
        export PA_ENABLED=True
        DEFAULT_EXP_NAME="SOPD_geometry3k_S2B_T2B_privilegedHint_tpc0005_m60_pa002_s0p2_${TOTAL_EPOCHS}epochs"
        ;;
    *)
        echo "Unknown MODE=${MODE}; expected baseline or sopd" >&2
        exit 2
        ;;
esac

export CUR_EXP_NAME="${CUR_EXP_NAME:-${DEFAULT_EXP_NAME}}"
export PROJECT_NAME="${PROJECT_NAME:-verl_opsd_geometry3k}"
export EXP_NAME="${EXP_NAME:-fsdp/student-${MODEL_NAME}-teacher-${MODEL_NAME}-${CUR_EXP_NAME}/loss-k1-pg-True}"
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output/${CUR_EXP_NAME}}"
export RAY_TMPDIR="${RAY_TMPDIR:-/path/to/ray_${USER}}"

export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/path/to/Geometry3K/train.parquet}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-/path/to/Geometry3K/test.parquet}"
export MAX_PROMPT="${MAX_PROMPT:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-192}"
export TEST_FREQ="${TEST_FREQ:-20}"
export TOTAL_EPOCHS

# Vanilla sampled-token policy-gradient OPSD.
export DISTILLATION_LOSS_MODE=k1
export USE_POLICY_GRADIENT=True

# Teacher-Calibrated Policy Contrast (TPC), inactive in baseline mode.
export TPC_GATE_MODE=teacher_positive
export TPC_MASK_STRATEGY=random_mask
export TPC_MASK_RATIO=0.6
export TPC_PATCH_SIZE="${TPC_PATCH_SIZE:-14}"
export TPC_COEF=0.005
export TPC_LOG_RATIO_CLIP="${TPC_LOG_RATIO_CLIP:-20.0}"
export TPC_KL_MAX="${TPC_KL_MAX:-10.0}"

# Policy Agreement (PA), inactive in baseline mode.
export PA_NOISE_STD=0.2
export PA_COEF=0.02
export PA_LOG_RATIO_CLIP="${PA_LOG_RATIO_CLIP:-20.0}"
export PA_KL_MAX="${PA_KL_MAX:-10.0}"

exec bash "${SCRIPT_DIR}/run_qwen3_vl_2b.sh" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    distillation.teacher_model.model_path="${MODEL_PATH}" \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.save_freq=20 \
    trainer.rollout_data_freq=10 \
    data.val_max_samples=-1 \
    "$@"
