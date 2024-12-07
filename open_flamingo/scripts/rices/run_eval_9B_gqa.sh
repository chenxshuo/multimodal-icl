#!/bin/bash
export HF_HOME="/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/.cache/huggingface"
ANNO_BASE="/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/robustness/datasets/gqa"
GQA_IMG="${ANNO_BASE}/images"
GQA_TRAIN_QUES_PATH="${ANNO_BASE}/train_ques_vqav2_format.json"
GQA_TRAIN_ANNO_PATH="${ANNO_BASE}/train_anno_vqav2_format.json"
GQA_VAL_QUES_PATH="${ANNO_BASE}/test_ques_vqav2_format.json"
GQA_VAL_ANNO_PATH="${ANNO_BASE}/test_anno_vqav2_format.json"
OUT_DIR="${ANNO_BASE}/rice_features"

# 9B
CKPT_PATH="/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/e6e175603712c7007fe3b9c0d50bdcfbd83adfc2/checkpoint.pt"
LM_MODEL="anas-awadalla/mpt-7b"
CROSS_ATTN_EVERY_N_LAYERS=4

SHOTS=$1
MASTER_PORT=$2
BS=$3

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`
COMMENT="9B-rice-gqa-shots-${SHOTS}"
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
    --shots ${SHOTS} \
    --trial_seeds 42 \
    --demo_mode  "gold" \
    --visual_demo_mode "random" \
    --rices \
    --cached_demonstration_features ${OUT_DIR} \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai \
    --eval_gqa \
    --gqa_image_dir_path ${GQA_IMG} \
    --gqa_train_questions_json_path ${GQA_TRAIN_QUES_PATH} \
    --gqa_train_annotations_json_path ${GQA_TRAIN_ANNO_PATH} \
    --gqa_test_questions_json_path ${GQA_VAL_QUES_PATH} \
    --gqa_test_annotations_json_path ${GQA_VAL_ANNO_PATH} \
