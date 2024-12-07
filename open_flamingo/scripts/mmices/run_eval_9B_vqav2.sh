#!/bin/bash
export HF_HOME="/PATH/TO/.cache/huggingface"
BASE_COCO_DATA_PATH="/PATH/TO/COCO"
BASE_VQAv2_PATH="/PATH/TO/vqav2"
VQAV2_IMG_TRAIN_PATH="${BASE_COCO_DATA_PATH}/train2014/"
VQAV2_ANNO_TRAIN_PATH="${BASE_VQAv2_PATH}/v2_mscoco_train2014_annotations.json"
VQAV2_QUESTION_TRAIN_PATH="${BASE_VQAv2_PATH}/v2_OpenEnded_mscoco_train2014_questions.json"
VQAV2_IMG_TEST_PATH="${BASE_COCO_DATA_PATH}/val2014"

#VQAV2_ANNO_TEST_PATH="${BASE_COCO_DATA_PATH}/v2_mscoco_val2014_annotations.json"
#VQAV2_QUESTION_TEST_PATH="${BASE_COCO_DATA_PATH}/v2_OpenEnded_mscoco_val2014_questions.json"
VQAV2_ANNO_TEST_PATH="${BASE_VQAv2_PATH}/karpathy_test_ann_vqav2_format.json"
VQAV2_QUESTION_TEST_PATH="${BASE_VQAv2_PATH}/karpathy_test_ques_vqav2_format.json"
VQAv2_OUT_DIR="${BASE_VQAv2_PATH}/rice_features"


# 9B
CKPT_PATH="/PATH/TO/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/e6e175603712c7007fe3b9c0d50bdcfbd83adfc2/checkpoint.pt"
LM_MODEL="anas-awadalla/mpt-7b"
CROSS_ATTN_EVERY_N_LAYERS=4

SHOTS=$1
MASTER_PORT=$2
BS=$3
VISUAL_MODE="no_images"
#VISUAL_MODE="random"

export CUDA_VISIBLE_DEVICES=0
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`
COMMENT="9B-rice-vqav2-shots-${SHOTS}"
RESULTS_FILE="./results/results_${TIMESTAMP}_${COMMENT}.json"
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUs" --master_port=${MASTER_PORT} open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path ${LM_MODEL} \
    --lm_tokenizer_path ${LM_MODEL} \
    --cross_attn_every_n_layers ${CROSS_ATTN_EVERY_N_LAYERS} \
    --checkpoint_path ${CKPT_PATH} \
    --results_file ${RESULTS_FILE} \
    --precision amp_bf16 \
    --batch_size ${BS} \
    --num_trials 1 \
    --shots ${SHOTS} \
    --trial_seeds 42 \
    --demo_mode  "gold" \
    --visual_demo_mode $VISUAL_MODE \
    --rices \
    --rices_find_by_ranking_similar_text \
    --cached_demonstration_features ${VQAv2_OUT_DIR} \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai \
    --eval_vqav2 \
    --vqav2_train_image_dir_path ${VQAV2_IMG_TRAIN_PATH} \
    --vqav2_train_annotations_json_path ${VQAV2_ANNO_TRAIN_PATH} \
    --vqav2_train_questions_json_path ${VQAV2_QUESTION_TRAIN_PATH} \
    --vqav2_test_image_dir_path ${VQAV2_IMG_TEST_PATH} \
    --vqav2_test_annotations_json_path ${VQAV2_ANNO_TEST_PATH} \
    --vqav2_test_questions_json_path ${VQAV2_QUESTION_TEST_PATH}


