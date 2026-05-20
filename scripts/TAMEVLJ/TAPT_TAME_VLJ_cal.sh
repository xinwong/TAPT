# custom config
DATA="/mnt/shared-storage-user/wangxin2/CLIP/"

TRAINER=TAMEVLJ
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASETS=("imagenet")
SEED=1
EPOCHS=(80)
MODEL_VARIANTS_STR="${MODEL_VARIANTS:-vit_b16 vit_b32}"
read -r -a MODEL_VARIANTS <<< "${MODEL_VARIANTS_STR}"

STATS_ROOT=./stats/TAME
LEGACY_STATS_ROOT=./stats/TAPT
STAT_PREFIX=TAME_VLJ
LEGACY_PREFIXES=("MoE_VLJ" "VLJ")

resolve_existing_file() {
    local primary="$1"
    shift

    if [ -f "${primary}" ]; then
        echo "${primary}"
        return 0
    fi

    for candidate in "$@"; do
        if [ -f "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "${primary}"
}

resolve_model_settings() {
    local model_variant="$1"

    case "${model_variant}" in
        vit_b16)
            MODEL_SLUG=vitb16
            CFG=TAPT_vit_b16_c2_ep100_batch32_2ctx_9depth_cal
            WEIGHTSPATH_ADV='/mnt/shared-storage-user/wangxin2/PromptLearning/Multimodal-Adversarial-Prompt-Tuning/output2025/train/imagenet/MoEAdvMaPLe/vit_b16_c2_ep100_batch32_2ctx_9depth_cross_datasets_16shots'
            WEIGHTSPATH_CLEAN='/mnt/shared-storage-user/wangxin2/PromptLearning/Multimodal-Adversarial-Prompt-Tuning/output2025/train/imagenet/MoEAdvMaPLe/vit_b16_c2_ep100_batch32_2ctx_9depth_cross_datasets_clean_16shots'
            ;;
        vit_b32)
            MODEL_SLUG=vitb32
            CFG=TAPT_vit_b32_c2_ep100_batch32_2ctx_9depth_cal
            WEIGHTSPATH_ADV='/mnt/shared-storage-user/wangxin2/PromptLearning/Multimodal-Adversarial-Prompt-Tuning/output2025/train/imagenet/MoEAdvMaPLe/vit_b32_c2_ep100_batch32_2ctx_9depth_cross_datasets_16shots'
            WEIGHTSPATH_CLEAN='/mnt/shared-storage-user/wangxin2/PromptLearning/Multimodal-Adversarial-Prompt-Tuning/output2025/train/imagenet/MoEAdvMaPLe/vit_b32_c2_ep100_batch32_2ctx_9depth_cross_datasets_clean_16shots'
            ;;
        *)
            echo "Unsupported MODEL_VARIANT=${model_variant}" >&2
            return 1
            ;;
    esac

    STATS_DIR="${STATS_ROOT}/${MODEL_SLUG}"
    mkdir -p "${STATS_DIR}"
}

resolve_stat_path() {
    local kind="$1"
    local variant="$2"
    local candidates=("${STATS_DIR}/${STAT_PREFIX}_${kind}_${MODEL_SLUG}_train_${variant}.pt")
    local legacy_prefix

    for legacy_prefix in "${LEGACY_PREFIXES[@]}"; do
        candidates+=("${LEGACY_STATS_ROOT}/${MODEL_SLUG}/${legacy_prefix}_${kind}_${MODEL_SLUG}_train_${variant}.pt")
        candidates+=("./stats/${MODEL_SLUG}/${legacy_prefix}_${kind}_${MODEL_SLUG}_train_${variant}.pt")
    done

    resolve_existing_file "${candidates[@]}"
}

run_cal() {
    local dataset="$1"
    local load_ep="$2"
    local ckpt_tag="$3"
    local weightspath="$4"
    local dir="./output/${TRAINER}/cal/${MODEL_SLUG}/${ckpt_tag}"
    local model_dir="${weightspath}/seed${SEED}"
    local vis_means
    local vis_vars
    local vis_means_clean
    local vis_vars_clean

    vis_means="$(resolve_stat_path means adv)"
    vis_vars="$(resolve_stat_path vars adv)"
    vis_means_clean="$(resolve_stat_path means clean)"
    vis_vars_clean="$(resolve_stat_path vars clean)"

    echo "Evaluating ${ckpt_tag} checkpoint for ${MODEL_SLUG}"
    echo "Running CAL job and saving logs to ${dir}"

    "${PYTHON_BIN}" train.py \
    --root "${DATA}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${dataset}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${dir}" \
    --model-dir "${model_dir}" \
    --load-epoch "${load_ep}" \
    --eval-only \
    DATASET.NUM_SHOTS 0 \
    TAPT.VIS_MEANS "${vis_means}" \
    TAPT.VIS_VARS "${vis_vars}" \
    TAPT.VIS_MEANS_CLEAN "${vis_means_clean}" \
    TAPT.VIS_VARS_CLEAN "${vis_vars_clean}"
}

for MODEL_VARIANT in "${MODEL_VARIANTS[@]}"; do
    resolve_model_settings "${MODEL_VARIANT}" || exit 1

    for DATASET in "${DATASETS[@]}"; do
        for LOADEP in "${EPOCHS[@]}"; do
            run_cal "${DATASET}" "${LOADEP}" adv "${WEIGHTSPATH_ADV}"
            run_cal "${DATASET}" "${LOADEP}" clean "${WEIGHTSPATH_CLEAN}"
        done
    done
done
