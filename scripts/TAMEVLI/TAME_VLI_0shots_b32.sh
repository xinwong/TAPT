## TAPT-TAME Test-Time Evaluation (TAME-VLI, ViT-B/32, lane04 / WU5)

# custom config
DATA="/path/to/CLIP/"

TRAINER=TAMEVLI

DATASETS=("caltech101" "dtd" "eurosat" "oxford_pets" "fgvc_aircraft" "oxford_flowers" "food101" "stanford_cars" "sun397" "ucf101" "imagenet")
SEED=1
EPOCHS=(100)
ATTACKS=("pgd" "di")

WEIGHTSPATH='/path/to/PromptLearning/Multimodal-Adversarial-Prompt-Tuning/output2025/train/imagenet/MoEAdvIVLP/vit_b32_c2_ep100_batch32_2+2ctx_9depth_16shots'

COMMON_PART=$(basename ${WEIGHTSPATH})

CFG=TAPT_vit_b32_c2_ep100_batch32_2ctx_9depth_l1_cross_datasets_step1_adv
SHOTS=0

MODEL_DIR=${WEIGHTSPATH}/seed${SEED}

for ATTACK in "${ATTACKS[@]}"; do

    ADV_DIR=/path/to/TAPT/output2025/evaluation/${ATTACK}/MoEAdvIVLP

    for DATASET in "${DATASETS[@]}"; do
        for LOADEP in "${EPOCHS[@]}"; do
            DIR=output/${ATTACK}/${TRAINER}/${CFG}_${SHOTS}shots/TAPT_eps1_step1_${SHOTS}shots/adv/${DATASET}/seed${SEED}/${LOADEP}
            ADVDATA_DIR=${ADV_DIR}/${COMMON_PART}/${DATASET}/seed${SEED}/${LOADEP}/
            if [ -d "$DIR" ]; then
                echo "Results are already available in ${DIR}. Skipping..."
            else
                echo "Evaluating model"
                echo "Runing the first phase job and save the output to ${DIR}"
                CUDA_VISIBLE_DEVICES=0 python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --model-dir ${MODEL_DIR} \
                --advdata-dir ${ADVDATA_DIR} \
                --load-epoch ${LOADEP} \
                --tapt \
                DATASET.NUM_SHOTS ${SHOTS} \
                TAPT.VIS_MEANS ./stats/TAME/vitb32/TAME_VIL_means_vitb32_train_adv.pt \
                TAPT.VIS_VARS ./stats/TAME/vitb32/TAME_VIL_vars_vitb32_train_adv.pt \
                TAPT.VIS_MEANS_CLEAN ./stats/TAME/vitb32/TAME_VIL_means_vitb32_train_clean.pt \
                TAPT.VIS_VARS_CLEAN ./stats/TAME/vitb32/TAME_VIL_vars_vitb32_train_clean.pt
            fi

        done
    done
done
