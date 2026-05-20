# custom config
DATA="/path/to/CLIP/"

TRAINER=TAPTVLI

DATASETS=("imagenet")
SEED=1
EPOCHS=(100)


# WEIGHTSPATH='/path/to/MMoP/output/train/imagenet/AdvIVLP/vit_b32_c2_ep100_batch32_2+2ctx_9depth_16shots'       # adv
WEIGHTSPATH='/path/to/MMoP/output/train/imagenet/AdvIVLP/vit_b32_c2_ep100_batch32_2+2ctx_9depth_clean_16shots'     # clean

CFG=TAPT_vit_b32_c2_ep100_batch32_2ctx_9depth_cal
SHOTS=0

MODEL_DIR=${WEIGHTSPATH}/seed${SEED}

for DATASET in "${DATASETS[@]}"; do
    for LOADEP in "${EPOCHS[@]}"; do
        DIR=./output/${TRAINER}/cal
        if [ -d "$DIR" ]; then
            echo "Results are already available in ${DIR}. Skipping..."
        else
            echo "Evaluating model"
            echo "Runing the first phase job and save the output to ${DIR}"
            # Evaluate on evaluation datasets
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            TAPT.VIS_MEANS ./stats/old/imagenet_VLI_means_adv.pt \
            TAPT.VIS_VARS ./stats/old/imagenet_VLI_vars_adv.pt \
            TAPT.VIS_MEANS_CLEAN ./stats/old/imagenet_VLI_means_clean.pt \
            TAPT.VIS_VARS_CLEAN ./stats/old/imagenet_VLI_vars_clean.pt \

        fi

    done
done
