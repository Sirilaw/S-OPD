#!/usr/bin/env bash
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Enable WandB offline mode
export WANDB_MODE=offline
# Always start an isolated local Ray cluster for this training job. This avoids
# auto-connecting to another user's cluster through the shared /tmp/ray state.
unset RAY_ADDRESS
export RAY_TMPDIR="${RAY_TMPDIR:-/path/to/ray_${USER}}"
############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

STUDENT_MODEL="Qwen3-VL-4B-Instruct"
TEACHER_MODEL="Qwen3-VL-8B-Instruct"
STUDENT_MODEL_PATH=/path/to/4b/model
# TEACHER_MODEL_PATH=/path/to/8b/model
TEACHER_MODEL_PATH=/path/to/8b/grpo/model

DISTILLATION_LOSS_MODE="${DISTILLATION_LOSS_MODE:-k1}"
USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT:-True}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-True}"

CUR_EXP_NAME="${CUR_EXP_NAME:-your_exp_name}"


PROJECT_NAME="${PROJECT_NAME:-verl_sopd_ViRL39K_${CUR_EXP_NAME}}"
EXP_NAME="${EXP_NAME:-fsdp/student-${STUDENT_MODEL}-teacher-${TEACHER_MODEL}-${CUR_EXP_NAME}/loss-${DISTILLATION_LOSS_MODE}-pg-${USE_POLICY_GRADIENT}}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output/${STUDENT_MODEL}/${CUR_EXP_NAME}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollouts}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
BEST_CHECKPOINT_METRIC="${BEST_CHECKPOINT_METRIC:-null}"

MAX_PROMPT="${MAX_PROMPT:-4096}"

MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}" # set to 2048 for 4B model

MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-128}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TEST_FREQ="${TEST_FREQ:--1}" # -1 for no validation
FULL_CHECKPOINT_SAVE_FREQ="${FULL_CHECKPOINT_SAVE_FREQ:-10}"
HUGGINGFACE_SAVE_FREQ="${HUGGINGFACE_SAVE_FREQ:-50}"


virl39k_train_path="${TRAIN_DATA_PATH:-/path/to/ViRL39K/train.parquet}"
# The trainer requires a non-empty validation loader even when validation is
# disabled. Reuse the training file and limit it to one unused placeholder.
virl39k_val_path="${VAL_DATA_PATH:-$virl39k_train_path}"
TRAIN_FILES="['$virl39k_train_path']"
TEST_FILES="['$virl39k_val_path']"


# Teacher-Calibrated Policy Contrast (TPC): maximize the student KL between
# original and masked images at tokens selected by the teacher's probability
# drop.
TPC_ENABLED="${TPC_ENABLED:-True}"
TPC_TEACHER_GATING_ENABLED="${TPC_TEACHER_GATING_ENABLED:-True}"
TPC_GATE_MODE="${TPC_GATE_MODE:-teacher_positive}"
TPC_MASK_STRATEGY="${TPC_MASK_STRATEGY:-random_mask}"
TPC_MASK_RATIO="${TPC_MASK_RATIO:-0.6}"
TPC_PATCH_SIZE="${TPC_PATCH_SIZE:-14}"
TPC_COEF="${TPC_COEF:-0.005}"
TPC_LOG_RATIO_CLIP="${TPC_LOG_RATIO_CLIP:-20.0}"
TPC_KL_MAX="${TPC_KL_MAX:-10.0}"

# Policy Agreement (PA): minimize the student KL between original and mildly
# Gaussian-perturbed images.
PA_ENABLED="${PA_ENABLED:-True}"
PA_NOISE_STD="${PA_NOISE_STD:-0.1}"
PA_COEF="${PA_COEF:-0.02}"
PA_LOG_RATIO_CLIP="${PA_LOG_RATIO_CLIP:-20.0}"
PA_KL_MAX="${PA_KL_MAX:-10.0}"

# Log paths. Override LOG_DIR when launching if a different location is needed.
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
RUN_TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
export VERL_ZMQ_NAMESPACE="${VERL_ZMQ_NAMESPACE:-${USER}_${RUN_TIMESTAMP}_$$}"
CONSOLE_LOG_PATH="${LOG_DIR}/train_${RUN_TIMESTAMP}.log"
export VERL_FILE_LOGGER_PATH="${LOG_DIR}/metrics_${RUN_TIMESTAMP}.jsonl"

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-False}"

ACTOR_MICRO_BATCH_SIZE_PER_GPU=1
ACTOR_MAX_TOKEN_LEN_PER_GPU=$(( ACTOR_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$(( ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=False

# STUDENT_WORLD_SIZE=2
STUDENT_WORLD_SIZE=2
TEACHER_RESOURCE_POOL=True
# TEACHER_WORLD_SIZE=2
TEACHER_WORLD_SIZE=2
ENFORCE_EAGER=True # true for faster debugging

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.teacher_prompt_key=teacher_prompt
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=$FILTER_OVERLONG_PROMPTS
    data.filter_overlong_prompts_workers=32
    data.truncation='error'
    data.shuffle=True
    data.image_key=images
    data.val_max_samples=1
)

MODEL=(
    actor_rollout_ref.model.path=$STUDENT_MODEL_PATH
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

DISTILLATION=(
    distillation.enabled=True
    distillation.num_workers=8
    distillation.teacher_model.enable_resource_pool=$TEACHER_RESOURCE_POOL
    distillation.teacher_model.n_gpus_per_node=$TEACHER_WORLD_SIZE
    distillation.teacher_model.nnodes=1
    distillation.teacher_model.model_path=$TEACHER_MODEL_PATH
    distillation.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_model.inference.name=$ROLLOUT_NAME
    distillation.teacher_model.inference.gpu_memory_utilization=0.65
    distillation.teacher_model.inference.free_cache_engine=False
    distillation.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_model.inference.max_num_seqs=128
    +distillation.teacher_model.inference.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
    distillation.teacher_prompt_key=teacher_prompt
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT

    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
    # Backend mapping: TPC is implemented by the existing PAPO loss fields.
    distillation.distillation_loss.papo_enabled=$TPC_ENABLED
    distillation.distillation_loss.papo_teacher_gating_enabled=$TPC_TEACHER_GATING_ENABLED
    distillation.distillation_loss.papo_teacher_gate_mode=$TPC_GATE_MODE
    distillation.distillation_loss.papo_negative_strategy=$TPC_MASK_STRATEGY
    distillation.distillation_loss.papo_mask_ratio=$TPC_MASK_RATIO
    distillation.distillation_loss.papo_patch_size=$TPC_PATCH_SIZE
    distillation.distillation_loss.papo_coef=$TPC_COEF
    distillation.distillation_loss.papo_log_ratio_clip=$TPC_LOG_RATIO_CLIP
    distillation.distillation_loss.papo_kl_max=$TPC_KL_MAX
    # Backend mapping: PA is implemented by the Gaussian near-pull loss fields.
    distillation.distillation_loss.gaussian_near_pull_enabled=$PA_ENABLED
    distillation.distillation_loss.gaussian_near_pull_std=$PA_NOISE_STD
    distillation.distillation_loss.gaussian_near_pull_coef=$PA_COEF
    distillation.distillation_loss.gaussian_near_pull_log_ratio_clip=$PA_LOG_RATIO_CLIP
    distillation.distillation_loss.gaussian_near_pull_kl_max=$PA_KL_MAX
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=128
    actor_rollout_ref.rollout.n=1
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb","file"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=$FULL_CHECKPOINT_SAVE_FREQ
    trainer.huggingface_save_freq=$HUGGINGFACE_SAVE_FREQ
    trainer.test_freq=$TEST_FREQ
    trainer.total_epochs=$TOTAL_EPOCHS
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=auto
    trainer.log_val_generations=5
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR
    trainer.default_local_dir=$CHECKPOINT_DIR
    # Rotation is handled after saving so 50-step HF milestones are preserved.
    trainer.max_actor_ckpt_to_keep=null
    trainer.max_full_actor_ckpt_to_keep=1 # only the newest step keeps resumable FSDP/optimizer/extra state
    +trainer.max_optimizer_ckpt_to_keep=1
    trainer.best_checkpoint_metric=$BEST_CHECKPOINT_METRIC
    trainer.best_checkpoint_mode=max
    trainer.rollout_data_freq=10
)

REWARD=(
    reward.custom_reward_function.path="${SCRIPT_DIR}/virl39k_reward.py"
)

RAY=(
    +ray_kwargs.ray_init.address=local
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR"
)



############################ Launch ############################

mkdir -p "$LOG_DIR"
echo "Console log: $CONSOLE_LOG_PATH"
echo "Metrics log: $VERL_FILE_LOGGER_PATH"

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='example_ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${RAY[@]}" \
    "$@" 2>&1 | tee "$CONSOLE_LOG_PATH"
