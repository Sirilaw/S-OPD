#!/usr/bin/env bash
set -euo pipefail

###############################################################################
#                         User configuration
# Edit this section, then run: bash evaluation/vlmevalkit/run_eval.sh
###############################################################################

# One independent TP=1 model replica is launched per listed GPU. The process
# count is derived automatically, so changing GPU is sufficient.
GPU="${GPU:-0,1,2,3}"

# Primary model. Set EVAL_PRIMARY_MODEL=false to evaluate only EXTRA_MODELS.
EVAL_PRIMARY_MODEL=false

# 2B model
MODEL_NAME="qwen3vl-2b-instruct"
MODEL_PATH="/path/to/qwen3vl-2b-instruct"

# 4B MODEL
MODEL_NAME="qwen3vl-4b-instruct"
MODEL_PATH="/path/to/qwen3vl-4b-instruct"

MODEL_ALIAS="base"

# Optional additional models. Leave empty for the normal single-model run.
# Format: "alias|model_path". The text before "|" becomes that model's alias.
EXTRA_MODELS=(
    # 2B models
    "alias|model_path"
)

# Local datasets. At least one local or built-in dataset must be configured.
# Format: "benchmark_name|manifest_path"
LOCAL_BENCHMARKS=(
    "benchmark_name|manifest_path"
)
EVAL_LOCAL_BENCHMARKS=false  # Set false when evaluating only built-in datasets.

# VLMEvalKit built-in datasets, for example: MMMU_DEV_VAL or MathVista_MINI.
BUILTIN_BENCHMARKS=(
    # Add any other VLMEvalKit built-in dataset name here.
)
# Common built-in benchmark switches. Enable any combination.
# MathVerse and MathVista use the official VLMEvalKit MINI variants.
EVAL_ZEROBENCH_SUB=false # ZEROBench_sub
EVAL_MMMU_VAL=false      # MMMU_DEV_VAL
EVAL_WEMATH=false        # WeMath
EVAL_LOGICVISTA=false    # LogicVista
EVAL_MATHVISTA=false     # MathVista_MINI
EVAL_MATHVISION=false    # MathVision
EVAL_MATHVERSE=false     # MathVerse_MINI

EVAL_VISUALPUZZLES=false  # VisualPuzzles (exact-match multiple-choice scoring)
EVAL_VSTAR_BENCH=true   # V* Bench (multiple-choice; no LLM judge required)
# Enables both full-image and crop-image views for the official zooming gap.
EVAL_ZOOMBENCH=true     # ZoomBench + ZoomBench_Crop (open questions need LLM judge)
EVAL_POPE=false            # POPE object-hallucination benchmark (F1/accuracy/precision/recall)
EVAL_HALLUSIONBENCH=false  # HallusionBench visual-illusion/language-hallucination diagnostics


# ZoomBench is downloaded from Hugging Face on first use. Its two views share
# one cache, while their predictions and scores remain separate datasets.
ZOOMBENCH_REPO="inclusionAI/ZoomBench"
ZOOMBENCH_SPLIT="test"

# Optional image roots for local datasets.
# Format: "benchmark_name|image_directory"
IMAGE_ROOTS=(
    # "VisionR1-Test|/path/to/images"
)

# Override dataset column detection. Leave empty for automatic detection.
QUESTION_FIELD=""
ANSWER_FIELD=""
IMAGE_FIELD=""

# ------------ Generation Configuration ------------
# Prompt handling.
SYSTEM_PROMPT="You are a helpful assistant"  # Set to "" to disable.
STRIP_USER_SUFFIX=true  # true: remove suffix from prompts in Vision-R1. false: keep the original prompt.

# Generation and vLLM settings.
BACKEND="vllm"              # vllm or transformers
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
# Decoding mode:
#   greedy (default): deterministic, one inference per example.
#   sample_once: one seeded sampling inference per example using the paper
#                settings below; this is not an @K metric.
#   acc@8: eight independently seeded sampling evaluations.
SAMPLING_MODE="${SAMPLING_MODE:-greedy}"
SAMPLING_SEED="${SAMPLING_SEED:-0}"
SAMPLE_ONCE_TEMPERATURE="${SAMPLE_ONCE_TEMPERATURE:-0.7}"
SAMPLE_ONCE_TOP_P="${SAMPLE_ONCE_TOP_P:-0.8}"
SAMPLE_ONCE_TOP_K="${SAMPLE_ONCE_TOP_K:-20}"
SAMPLE_ONCE_REPETITION_PENALTY="${SAMPLE_ONCE_REPETITION_PENALTY:-1.0}"
SAMPLE_ONCE_PRESENCE_PENALTY="${SAMPLE_ONCE_PRESENCE_PENALTY:-1.5}"
GPU_MEMORY_UTILIZATION=0.8
MAX_MODEL_LEN=32768
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-0}"
# With multiple TP=1 replicas, let every completed request claim the next
# global sample instead of assigning a fixed dataset slice to each GPU.
GLOBAL_DYNAMIC_QUEUE="${GLOBAL_DYNAMIC_QUEUE:-true}"
DYNAMIC_INFLIGHT_PER_GPU="${DYNAMIC_INFLIGHT_PER_GPU:-64}"
BALANCED_SHARDING="${BALANCED_SHARDING:-true}"


# Optional judge settings. MathVista/MathVerse/MathVision official scoring
# generally needs a working judge API. Keep JUDGE_MODEL empty when not needed.
# Supply credentials through the environment; never store API keys in this file.
JUDGE_API_KEY="${JUDGE_API_KEY:-}"

# Use local vllm server
JUDGE_MODEL="qwen3-30b-a3b"               # Example: "gpt-4o-mini"
JUDGE_BASE_URL="http://127.0.0.1:8001/v1"
JUDGE_ARGS=""                
JUDGE_API_NPROC=32

# Runtime.
PYTHON_BIN="${PYTHON_BIN:-python}"  # Or an absolute conda-environment Python path.
MODE="${MODE:-all}"         # all, infer, or eval
REUSE=false                # Reuse existing inference output when supported.
VERBOSE=false
OFFLINE=false                # Set false the first time a built-in dataset must download.
DISABLE_INTERNAL_LOG=true   # Keep only the single LOG_FILE console log.
USE_TMUX="${USE_TMUX:-true}"
TMUX_SESSION_PREFIX="vlmeval"  # A unique suffix is added automatically.
LOCK_OUTPUT=true               # Prevent concurrent writes to the same alias/benchmark.
DRY_RUN="${DRY_RUN:-false}"

# Lossless artifacts. JSON avoids Excel's 32,767-character cell limit and is
# used for predictions plus MathVerse extract/score intermediate tables.
PRED_FORMAT="${PRED_FORMAT:-json}"
EVAL_FORMAT="${EVAL_FORMAT:-json}"

# Paths. A single console log is written to LOG_FILE.
EVAL_ROOT="${EVAL_ROOT:-/path/to/eval_results}"
OUTPUT_ROOT="${EVAL_ROOT}/${MODEL_NAME}"
DATASET_CACHE_ROOT="${EVAL_ROOT}/dataset_cache"
BUILTIN_DATA_CACHE_ROOT="${DATASET_CACHE_ROOT}/vlmevalkit_builtin"
WORKLOAD_PROFILE_DIR="${WORKLOAD_PROFILE_DIR:-${DATASET_CACHE_ROOT}/workload_profiles}"
# Each model log is derived as:
# ${OUTPUT_ROOT}/${alias}/${BENCH_NAME}/evaluation.log
HF_HOME="/path/to/hf_home"
VLMEVALKIT_ROOT=""          # Empty means <repo>/third_party/VLMEvalKit.

# Extra arguments passed directly to VLMEvalKit.
EXTRA_ARGS=(
    # "--reuse-aux"
    # "none"
)

###############################################################################
#                         End user configuration
###############################################################################

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLMEVALKIT_ROOT="${VLMEVALKIT_ROOT:-${REPO_ROOT}/third_party/VLMEvalKit}"
RUNNER="${REPO_ROOT}/evaluation/vlmevalkit/run_qwen3vl.py"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        0|false|no|n|off) return 1 ;;
        *) die "Invalid boolean value: $1" ;;
    esac
}

case "${SAMPLING_MODE,,}" in
    greedy)
        SAMPLING_MODE="greedy"
        TEMPERATURE=0
        TOP_P=1
        TOP_K=20
        REPETITION_PENALTY=1.0
        PRESENCE_PENALTY=1.5
        SAMPLING_RUNS=1
        ;;
    sample|sample_once|single_sample)
        SAMPLING_MODE="sample_once"
        TEMPERATURE="${SAMPLE_ONCE_TEMPERATURE}"
        TOP_P="${SAMPLE_ONCE_TOP_P}"
        TOP_K="${SAMPLE_ONCE_TOP_K}"
        REPETITION_PENALTY="${SAMPLE_ONCE_REPETITION_PENALTY}"
        PRESENCE_PENALTY="${SAMPLE_ONCE_PRESENCE_PENALTY}"
        SAMPLING_RUNS=1
        ;;
    acc@8|acc_at_8|acc8)
        SAMPLING_MODE="acc@8"
        TEMPERATURE=1
        TOP_P=1
        TOP_K=20
        REPETITION_PENALTY=1.0
        PRESENCE_PENALTY=1.5
        SAMPLING_RUNS=8
        ;;
    *)
        die "SAMPLING_MODE must be greedy, sample_once, or acc@8"
        ;;
esac
[[ "${SAMPLING_SEED}" =~ ^[0-9]+$ ]] || die "SAMPLING_SEED must be a non-negative integer"
[[ "${TEMPERATURE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || die "Temperature must be a non-negative number, got ${TEMPERATURE}"
[[ "${TOP_P}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || die "Top-p must be numeric, got ${TOP_P}"
[[ "${TOP_K}" =~ ^-?[0-9]+$ ]] || die "Top-k must be an integer, got ${TOP_K}"
[[ "${REPETITION_PENALTY}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || die "Repetition penalty must be a positive number, got ${REPETITION_PENALTY}"
[[ "${PRESENCE_PENALTY}" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || die "Presence penalty must be numeric, got ${PRESENCE_PENALTY}"

[[ "${GPU}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "GPU must be a comma-separated list of numeric IDs"
IFS=',' read -r -a GPU_IDS <<< "${GPU}"
GPU_COUNT="${#GPU_IDS[@]}"
# Keep the data-parallel topology synchronized with GPU automatically. An
# explicit environment override is retained for advanced/debug use only.
DATA_PARALLEL_NPROC="${DATA_PARALLEL_NPROC:-${GPU_COUNT}}"
[[ "${DATA_PARALLEL_NPROC}" =~ ^[1-9][0-9]*$ ]] || die "DATA_PARALLEL_NPROC must be a positive integer"
if (( DATA_PARALLEL_NPROC != GPU_COUNT )); then
    die "For one model replica per GPU, DATA_PARALLEL_NPROC (${DATA_PARALLEL_NPROC}) must equal GPU count (${GPU_COUNT})"
fi

if ! is_true "${EVAL_LOCAL_BENCHMARKS}"; then
    LOCAL_BENCHMARKS=()
fi
is_true "${EVAL_WEMATH}" && BUILTIN_BENCHMARKS+=("WeMath")
is_true "${EVAL_LOGICVISTA}" && BUILTIN_BENCHMARKS+=("LogicVista")
is_true "${EVAL_MATHVERSE}" && BUILTIN_BENCHMARKS+=("MathVerse_MINI")
is_true "${EVAL_MATHVISTA}" && BUILTIN_BENCHMARKS+=("MathVista_MINI")
is_true "${EVAL_MATHVISION}" && BUILTIN_BENCHMARKS+=("MathVision")
is_true "${EVAL_ZEROBENCH_SUB}" && BUILTIN_BENCHMARKS+=("ZEROBench_sub")
is_true "${EVAL_MMMU_VAL}" && BUILTIN_BENCHMARKS+=("MMMU_DEV_VAL")
is_true "${EVAL_VISUALPUZZLES}" && BUILTIN_BENCHMARKS+=("VisualPuzzles")
is_true "${EVAL_VSTAR_BENCH}" && BUILTIN_BENCHMARKS+=("VStarBench")
if is_true "${EVAL_ZOOMBENCH}"; then
    BUILTIN_BENCHMARKS+=("ZoomBench" "ZoomBench_Crop")
fi
is_true "${EVAL_POPE}" && BUILTIN_BENCHMARKS+=("POPE")
is_true "${EVAL_HALLUSIONBENCH}" && BUILTIN_BENCHMARKS+=("HallusionBench")

BENCHMARK_NAMES=()
for spec in "${LOCAL_BENCHMARKS[@]}"; do
    BENCHMARK_NAMES+=("${spec%%|*}")
done
for name in "${BUILTIN_BENCHMARKS[@]}"; do
    [[ -n "${name}" ]] && BENCHMARK_NAMES+=("${name}")
done
((${#BENCHMARK_NAMES[@]} > 0)) || die "Configure at least one benchmark"

BENCH_NAME="$(IFS=+; echo "${BENCHMARK_NAMES[*]}")"
BENCH_NAME="${BENCH_NAME//\//_}"
BENCH_NAME="${BENCH_NAME// /_}"

MODELS=()
if is_true "${EVAL_PRIMARY_MODEL}"; then
    MODELS+=("${MODEL_ALIAS}|${MODEL_PATH}")
fi
MODELS+=("${EXTRA_MODELS[@]}")
((${#MODELS[@]} > 0)) || die "No models selected: enable EVAL_PRIMARY_MODEL or configure EXTRA_MODELS"

# Use the first selected model in the launcher message/session name. Each model
# still writes to its own ${OUTPUT_ROOT}/${alias}/${BENCH_NAME}/evaluation.log.
FIRST_MODEL_ALIAS="${MODELS[0]%%|*}"
if [[ "${SAMPLING_MODE}" == "acc@8" ]]; then
    LOG_FILE="${OUTPUT_ROOT}/${FIRST_MODEL_ALIAS}/${BENCH_NAME}/acc_at_8/sample_01/evaluation.log"
elif [[ "${SAMPLING_MODE}" == "sample_once" ]]; then
    LOG_FILE="${OUTPUT_ROOT}/${FIRST_MODEL_ALIAS}/${BENCH_NAME}/sample_once/evaluation.log"
else
    LOG_FILE="${OUTPUT_ROOT}/${FIRST_MODEL_ALIAS}/${BENCH_NAME}/evaluation.log"
fi

if is_true "${USE_TMUX}" && [[ "${VLMEVAL_INSIDE_TMUX:-0}" != "1" ]]; then
    command -v tmux >/dev/null 2>&1 || die "tmux is not installed"
    session_tag="${FIRST_MODEL_ALIAS}_${BENCH_NAME}_gpu${GPU}"
    session_tag="$(printf '%s' "${session_tag}" | tr -c '[:alnum:]_-' '_')"
    TMUX_SESSION="${TMUX_SESSION_PREFIX}_${session_tag}_$(date +'%Y%m%d_%H%M%S')_$$"
    printf -v tmux_command \
        'env VLMEVAL_INSIDE_TMUX=1 GPU=%q DATA_PARALLEL_NPROC=%q MAX_NEW_TOKENS=%q MODE=%q SAMPLING_MODE=%q SAMPLING_SEED=%q SAMPLE_ONCE_TEMPERATURE=%q SAMPLE_ONCE_TOP_P=%q SAMPLE_ONCE_TOP_K=%q SAMPLE_ONCE_REPETITION_PENALTY=%q SAMPLE_ONCE_PRESENCE_PENALTY=%q VLLM_BATCH_SIZE=%q GLOBAL_DYNAMIC_QUEUE=%q DYNAMIC_INFLIGHT_PER_GPU=%q BALANCED_SHARDING=%q WORKLOAD_PROFILE_DIR=%q PRED_FORMAT=%q EVAL_FORMAT=%q DRY_RUN=%q EVAL_ROOT=%q PYTHON_BIN=%q bash %q' \
        "${GPU}" "${DATA_PARALLEL_NPROC}" "${MAX_NEW_TOKENS}" "${MODE}" "${SAMPLING_MODE}" "${SAMPLING_SEED}" \
        "${SAMPLE_ONCE_TEMPERATURE}" "${SAMPLE_ONCE_TOP_P}" "${SAMPLE_ONCE_TOP_K}" \
        "${SAMPLE_ONCE_REPETITION_PENALTY}" "${SAMPLE_ONCE_PRESENCE_PENALTY}" "${VLLM_BATCH_SIZE}" \
        "${GLOBAL_DYNAMIC_QUEUE}" "${DYNAMIC_INFLIGHT_PER_GPU}" "${BALANCED_SHARDING}" \
        "${WORKLOAD_PROFILE_DIR}" "${PRED_FORMAT}" "${EVAL_FORMAT}" "${DRY_RUN}" \
        "${EVAL_ROOT}" "${PYTHON_BIN}" "${SCRIPT_PATH}"
    mkdir -p "$(dirname "${LOG_FILE}")"
    tmux new-session -d \
        -s "${TMUX_SESSION}" \
        -c "${REPO_ROOT}" \
        "${tmux_command}"
    echo "Evaluation started in tmux session: ${TMUX_SESSION}"
    echo "Attach: tmux attach -t ${TMUX_SESSION}"
    echo "Log: ${LOG_FILE}"
    exit 0
fi

[[ -f "${RUNNER}" ]] || die "Runner not found: ${RUNNER}"
[[ -f "${VLMEVALKIT_ROOT}/run.py" ]] || die "VLMEvalKit run.py not found: ${VLMEVALKIT_ROOT}/run.py"
[[ -n "${MODEL_NAME}" ]] || die "MODEL_NAME cannot be empty"
if is_true "${EVAL_PRIMARY_MODEL}"; then
    [[ -n "${MODEL_ALIAS}" ]] || die "MODEL_ALIAS cannot be empty when EVAL_PRIMARY_MODEL=true"
    [[ -n "${MODEL_PATH}" ]] || die "MODEL_PATH cannot be empty when EVAL_PRIMARY_MODEL=true"
fi
[[ "${BACKEND}" == "vllm" || "${BACKEND}" == "transformers" ]] || die "BACKEND must be vllm or transformers"
[[ "${MODE}" == "all" || "${MODE}" == "infer" || "${MODE}" == "eval" ]] || die "MODE must be all, infer, or eval"
[[ "${PRED_FORMAT}" == "json" || "${PRED_FORMAT}" == "tsv" ]] \
    || die "PRED_FORMAT must be json or tsv for lossless long-text storage"
[[ "${EVAL_FORMAT}" == "json" || "${EVAL_FORMAT}" == "csv" ]] \
    || die "EVAL_FORMAT must be json or csv"
[[ "${VLLM_BATCH_SIZE}" =~ ^[0-9]+$ ]] || die "VLLM_BATCH_SIZE must be a non-negative integer"
[[ "${DYNAMIC_INFLIGHT_PER_GPU}" =~ ^[1-9][0-9]*$ ]] \
    || die "DYNAMIC_INFLIGHT_PER_GPU must be a positive integer"
is_true "${GLOBAL_DYNAMIC_QUEUE}" || [[ "${GLOBAL_DYNAMIC_QUEUE,,}" =~ ^(0|false|no|n|off)$ ]] \
    || die "Invalid GLOBAL_DYNAMIC_QUEUE value: ${GLOBAL_DYNAMIC_QUEUE}"
is_true "${BALANCED_SHARDING}" || [[ "${BALANCED_SHARDING,,}" =~ ^(0|false|no|n|off)$ ]] \
    || die "Invalid BALANCED_SHARDING value: ${BALANCED_SHARDING}"
mkdir -p "$(dirname "${LOG_FILE}")" "${DATASET_CACHE_ROOT}" "${BUILTIN_DATA_CACHE_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME
export LMUData="${BUILTIN_DATA_CACHE_ROOT}"
export VLMEVAL_VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE}"
export VLMEVAL_WORKLOAD_PROFILE_DIR="${WORKLOAD_PROFILE_DIR}"
export VLMEVAL_DYNAMIC_INFLIGHT_PER_GPU="${DYNAMIC_INFLIGHT_PER_GPU}"
if is_true "${BALANCED_SHARDING}"; then
    export VLMEVAL_BALANCED_SHARDING=1
else
    export VLMEVAL_BALANCED_SHARDING=0
fi
export PRED_FORMAT EVAL_FORMAT
export PYTHONUNBUFFERED=1
if is_true "${OFFLINE}"; then
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

COMMON_ARGS=(
    --vlmevalkit-root "${VLMEVALKIT_ROOT}"
    --backend "${BACKEND}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --repetition-penalty "${REPETITION_PENALTY}"
    --presence-penalty "${PRESENCE_PENALTY}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --dataset-cache-root "${DATASET_CACHE_ROOT}"
    --zoombench-repo "${ZOOMBENCH_REPO}"
    --zoombench-split "${ZOOMBENCH_SPLIT}"
    --mode "${MODE}"
)

for spec in "${LOCAL_BENCHMARKS[@]}"; do
    [[ "${spec}" == *"|"* ]] || die "Invalid LOCAL_BENCHMARKS entry: ${spec}"
    name="${spec%%|*}"
    manifest="${spec#*|}"
    [[ -n "${name}" && -f "${manifest}" ]] || die "Local benchmark is invalid or missing: ${spec}"
    COMMON_ARGS+=(--benchmark "${name}=${manifest}")
done

for name in "${BUILTIN_BENCHMARKS[@]}"; do
    [[ -n "${name}" ]] && COMMON_ARGS+=(--builtin-benchmark "${name}")
done

for spec in "${IMAGE_ROOTS[@]}"; do
    [[ "${spec}" == *"|"* ]] || die "Invalid IMAGE_ROOTS entry: ${spec}"
    name="${spec%%|*}"
    image_root="${spec#*|}"
    [[ -d "${image_root}" ]] || die "Image root does not exist: ${image_root}"
    COMMON_ARGS+=(--image-root "${name}=${image_root}")
done

[[ -n "${QUESTION_FIELD}" ]] && COMMON_ARGS+=(--question-field "${QUESTION_FIELD}")
[[ -n "${ANSWER_FIELD}" ]] && COMMON_ARGS+=(--answer-field "${ANSWER_FIELD}")
[[ -n "${IMAGE_FIELD}" ]] && COMMON_ARGS+=(--image-field "${IMAGE_FIELD}")
[[ -n "${SYSTEM_PROMPT}" ]] && COMMON_ARGS+=(--system-prompt "${SYSTEM_PROMPT}")
is_true "${STRIP_USER_SUFFIX}" && COMMON_ARGS+=(--strip-user-suffix)
is_true "${REUSE}" && COMMON_ARGS+=(--reuse)
is_true "${VERBOSE}" && COMMON_ARGS+=(--verbose)
is_true "${DISABLE_INTERNAL_LOG}" && COMMON_ARGS+=(--disable-internal-log)
is_true "${GLOBAL_DYNAMIC_QUEUE}" && COMMON_ARGS+=(
    --global-dynamic-queue
    --dynamic-inflight-per-gpu "${DYNAMIC_INFLIGHT_PER_GPU}"
)
if [[ -n "${JUDGE_MODEL}" ]]; then
    COMMON_ARGS+=(--judge "${JUDGE_MODEL}" --judge-api-nproc "${JUDGE_API_NPROC}")
    [[ -n "${JUDGE_API_KEY}" ]] && COMMON_ARGS+=(--judge-key "${JUDGE_API_KEY}")
    [[ -n "${JUDGE_BASE_URL}" ]] && COMMON_ARGS+=(--judge-base-url "${JUDGE_BASE_URL}")
    [[ -n "${JUDGE_ARGS}" ]] && COMMON_ARGS+=(--judge-args "${JUDGE_ARGS}")
fi

print_command() {
    local redact_next=false
    local arg
    for arg in "$@"; do
        if is_true "${redact_next}"; then
            printf ' %q' "***REDACTED***"
            redact_next=false
        else
            printf ' %q' "${arg}"
            [[ "${arg}" == "--judge-key" ]] && redact_next=true
        fi
    done
    printf '\n'
}

for spec in "${MODELS[@]}"; do
    [[ "${spec}" == *"|"* ]] || die "Invalid model entry: ${spec}"
    alias="${spec%%|*}"
    model_path="${spec#*|}"
    [[ -n "${alias}" && -n "${model_path}" ]] || die "Invalid model entry: ${spec}"
    [[ "${alias}" != *"/"* ]] || die "Model alias cannot contain '/': ${alias}"

    for ((sample_run = 1; sample_run <= SAMPLING_RUNS; sample_run++)); do
        sample_seed=$((SAMPLING_SEED + sample_run - 1))
        if [[ "${SAMPLING_MODE}" == "acc@8" ]]; then
            printf -v sample_name 'sample_%02d' "${sample_run}"
            work_dir="${OUTPUT_ROOT}/${alias}/${BENCH_NAME}/acc_at_8/${sample_name}"
        elif [[ "${SAMPLING_MODE}" == "sample_once" ]]; then
            work_dir="${OUTPUT_ROOT}/${alias}/${BENCH_NAME}/sample_once"
        else
            work_dir="${OUTPUT_ROOT}/${alias}/${BENCH_NAME}"
        fi
        model_log="${work_dir}/evaluation.log"
        mkdir -p "${work_dir}"

        if is_true "${LOCK_OUTPUT}"; then
            command -v flock >/dev/null 2>&1 || die "flock is required when LOCK_OUTPUT=true"
            exec {lock_fd}>"${work_dir}/.evaluation.lock"
            if ! flock -n "${lock_fd}"; then
                die "Another evaluation is already writing to: ${work_dir}"
            fi
        fi

        runner_args=(
            "${RUNNER}"
            --model-path "${model_path}"
            --work-dir "${work_dir}"
            --seed "${sample_seed}"
            "${COMMON_ARGS[@]}"
            "${EXTRA_ARGS[@]}"
        )
        if (( DATA_PARALLEL_NPROC > 1 )); then
            # torchrun gives each rank one GPU. VLMEvalKit partitions samples by
            # rank and merges the rank-local prediction dictionaries on rank 0.
            # Since each rank sees one GPU, Qwen3VLChat selects TP=1 rather than
            # splitting this small model across every visible device.
            command=(
                "${PYTHON_BIN}" -m torch.distributed.run
                --standalone
                "--nproc-per-node=${DATA_PARALLEL_NPROC}"
                "${runner_args[@]}"
            )
            parallel_mode="data-parallel (${DATA_PARALLEL_NPROC} replicas, TP=1)"
        else
            command=("${PYTHON_BIN}" "${runner_args[@]}")
            parallel_mode="single process"
        fi

        {
            echo "===== VLMEvalKit evaluation $(date --iso-8601=seconds) ====="
            echo "Model name: ${MODEL_NAME}"
            echo "Model alias: ${alias}"
            echo "Model path: ${model_path}"
            echo "GPU: ${GPU}"
            echo "Parallel mode: ${parallel_mode}"
            echo "Work directory: ${work_dir}"
            echo "Dataset cache: ${DATASET_CACHE_ROOT}"
            echo "Built-in dataset cache: ${BUILTIN_DATA_CACHE_ROOT}"
            echo "System prompt: ${SYSTEM_PROMPT}"
            echo "Max new tokens: ${MAX_NEW_TOKENS}"
            echo "Sampling mode: ${SAMPLING_MODE}"
            echo "Sampling run: ${sample_run}/${SAMPLING_RUNS}"
            echo "Sampling seed: ${sample_seed}"
            echo "Temperature: ${TEMPERATURE}"
            echo "Top-p: ${TOP_P}"
            echo "Top-k: ${TOP_K}"
            echo "Repetition penalty: ${REPETITION_PENALTY}"
            echo "Presence penalty: ${PRESENCE_PENALTY}"
            echo "Max model length: ${MAX_MODEL_LEN}"
            echo "Global dynamic queue: ${GLOBAL_DYNAMIC_QUEUE}"
            if is_true "${GLOBAL_DYNAMIC_QUEUE}"; then
                echo "Dynamic in-flight requests per GPU: ${DYNAMIC_INFLIGHT_PER_GPU}"
                echo "vLLM submission mode: asynchronous global refill"
            elif (( VLLM_BATCH_SIZE == 0 )); then
                echo "vLLM submission window: all rank-local requests (continuous refill)"
            else
                echo "vLLM submission window: ${VLLM_BATCH_SIZE}"
            fi
            echo "Balanced longest-first fallback: ${BALANCED_SHARDING}"
            echo "Workload profile directory: ${WORKLOAD_PROFILE_DIR}"
            echo "Prediction format: ${PRED_FORMAT}"
            echo "Evaluation format: ${EVAL_FORMAT}"
            printf "Command:"
            print_command "${command[@]}"
            if ! is_true "${DRY_RUN}"; then
                "${command[@]}"
            fi
            echo "===== ${alias} ${SAMPLING_MODE} ${sample_run}/${SAMPLING_RUNS} END $(date --iso-8601=seconds) ====="
        } 2>&1 | tee "${model_log}"

        if is_true "${LOCK_OUTPUT}"; then
            flock -u "${lock_fd}"
            exec {lock_fd}>&-
        fi
    done
done

echo "===== All evaluations finished $(date --iso-8601=seconds) ====="
