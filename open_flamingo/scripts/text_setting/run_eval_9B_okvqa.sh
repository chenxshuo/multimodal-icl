#!/bin/bash
export HF_HOME="/PATH/TO/.cache/huggingface"
BASE_COCO_DATA_PATH="/PATH/TO/COCO"
COCO_IMG_TRAIN_PATH="${BASE_COCO_DATA_PATH}/train2014"
COCO_IMG_VAL_PATH="${BASE_COCO_DATA_PATH}/val2014"
BASE_DATA_PATH="/PATH/TO/datasets"
OK_VQA_TRAIN_ANNO_PATH="${BASE_DATA_PATH}/okvqa/mscoco_train2014_annotations.json"
OK_VQA_TRAIN_QUES_PATH="${BASE_DATA_PATH}/okvqa/OpenEnded_mscoco_train2014_questions.json"
OK_VQA_VAL_ANNO_PATH="${BASE_DATA_PATH}/okvqa/mscoco_val2014_annotations.json"
OK_VQA_VAL_QUES_PATH="${BASE_DATA_PATH}/okvqa/OpenEnded_mscoco_val2014_questions.json"

# 9B
CKPT_PATH="/PATH/TO/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/e6e175603712c7007fe3b9c0d50bdcfbd83adfc2/checkpoint.pt"
LM_MODEL="anas-awadalla/mpt-7b"
CROSS_ATTN_EVERY_N_LAYERS=4

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`

MODE=$1
MASTER_PORT=$2
SHOT=$3
BS=$4

#MODE="only_labels"
#MODE="fixed_pseudo_question_length"
VISUAL_MODE="random"
COMMENT="9B-okvqa-$MODE-$VISUAL_MODE"

RESULTS_FILE="results_${TIMESTAMP}_${COMMENT}.json"
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
    --shots ${SHOT} \
    --trial_seeds 42 \
    --demo_mode  $MODE \
    --visual_demo_mode $VISUAL_MODE \
    --eval_ok_vqa \
    --ok_vqa_train_image_dir_path ${COCO_IMG_TRAIN_PATH} \
    --ok_vqa_train_annotations_json_path ${OK_VQA_TRAIN_ANNO_PATH} \
    --ok_vqa_train_questions_json_path ${OK_VQA_TRAIN_QUES_PATH} \
    --ok_vqa_test_image_dir_path ${COCO_IMG_VAL_PATH} \
    --ok_vqa_test_annotations_json_path ${OK_VQA_VAL_ANNO_PATH} \
    --ok_vqa_test_questions_json_path ${OK_VQA_VAL_QUES_PATH} \

