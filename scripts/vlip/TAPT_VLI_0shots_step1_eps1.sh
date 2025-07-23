## TAPT Clean Dataset Evaluation

# custom config
DATA="/path/to/CLIP"

TRAINER=TAPTVLI

DATASETS=("caltech101" "dtd" "eurosat" "oxford_pets" "fgvc_aircraft" "oxford_flowers" "food101" "stanford_cars" "sun397" "ucf101" "imagenet")
SEED=1
EPOCHS=(100)

WEIGHTSPATH='/path/to/Multimodal-Adversarial-Prompt-Tuning/output/train/imagenet/AdvIVLP/vit_b16_c2_ep100_batch32_2+2ctx_9depth_16shots'

CFG=TAPT_vit_b16_c2_ep100_batch32_2ctx_9depth_l1_cross_datasets_step1_clean
SHOTS=0

MODEL_DIR=${WEIGHTSPATH}/seed${SEED}

for DATASET in "${DATASETS[@]}"; do
    for LOADEP in "${EPOCHS[@]}"; do
        DIR=output/${TRAINER}/${CFG}_${SHOTS}shots/TAPT_eps1_step1_${SHOTS}shots/clean/${DATASET}/seed${SEED}/${LOADEP}
        if [ -d "$DIR" ]; then
            echo "Results are already available in ${DIR}. Skipping..."
        else
            echo "Evaluating model"
            echo "Runing the first phase job and save the output to ${DIR}"
            # Evaluate on evaluation datasets
            CUDA_VISIBLE_DEVICES=0 python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --tapt \
            DATASET.NUM_SHOTS ${SHOTS} \

        fi

    done
done



## TAPT Adversarial Dataset Evaluation

# custom config
DATA="/path/to/CLIP"

TRAINER=TAPTVLI

DATASETS=("caltech101" "dtd" "eurosat" "oxford_pets" "fgvc_aircraft" "oxford_flowers" "food101" "stanford_cars" "sun397" "ucf101" "imagenet")
SEED=1
EPOCHS=(100)

WEIGHTSPATH='/path/to/Multimodal-Adversarial-Prompt-Tuning/output/train/imagenet/AdvIVLP/vit_b16_c2_ep100_batch32_2+2ctx_9depth_16shots'

# Extract the common part from WEIGHTSPATH
COMMON_PART=$(basename ${WEIGHTSPATH})

CFG=TAPT_vit_b16_c2_ep100_batch32_2ctx_9depth_l1_cross_datasets_step1_adv
SHOTS=0

MODEL_DIR=${WEIGHTSPATH}/seed${SEED}
ADV_DIR='/path/to/adv_dataset/'

for DATASET in "${DATASETS[@]}"; do
    for LOADEP in "${EPOCHS[@]}"; do
        DIR=output/${TRAINER}/${CFG}_${SHOTS}shots/TAPT_eps1_step1_${SHOTS}shots/adv/${DATASET}/seed${SEED}/${LOADEP}
        ADVDATA_DIR=${ADV_DIR}/${COMMON_PART}/${DATASET}/seed${SEED}/${LOADEP}/
        if [ -d "$DIR" ]; then
            echo "Results are already available in ${DIR}. Skipping..."
        else
            echo "Evaluating model"
            echo "Runing the first phase job and save the output to ${DIR}"
            # Evaluate on evaluation datasets
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
            --tpat \
            DATASET.NUM_SHOTS ${SHOTS} \

        fi

    done
done
