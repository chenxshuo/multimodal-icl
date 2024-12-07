#!/bin/bash
export HF_HOME="/PATH/TO/.cache/huggingface"
BASE_COCO_DATA_PATH="/PATH/TO/COCO"

COCO_IMG_TRAIN_PATH="${BASE_COCO_DATA_PATH}/train2014"
COCO_IMG_VAL_PATH="${BASE_COCO_DATA_PATH}/val2014"
COCO_ANNO_PATH="${BASE_COCO_DATA_PATH}/annotations-2014/captions_val2014.json"
COCO_KARPATHY_PATH="${BASE_COCO_DATA_PATH}/dataset_coco.json"
COCO_RICE_FEAT="/PATH/TO/robustness/datasets/coco/rice_features"
CACHED_SHOT_RESULT="/PATH/TO/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/demo_mode_gold/visual_demo_mode_random/coco/shot_32/2023-09-20_14-54-39/coco_results_shots_32.json"
#CACHED_SHOT_RESULT="/PATH/TO/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results/OF9B/demo_mode_gold/visual_demo_mode_random/coco/shot_0/2023-09-20_08-28-26/coco_results_shots_0.json"
# 9B
CKPT_PATH="/PATH/TO/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/e6e175603712c7007fe3b9c0d50bdcfbd83adfc2/checkpoint.pt"
LM_MODEL="anas-awadalla/mpt-7b"
CROSS_ATTN_EVERY_N_LAYERS=4

#CKPT_PATH="/PATH/TO/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b-langinstruct/snapshots/656bbbcd4508db84ccc83c02361011c6fe92ae93/checkpoint.pt"
#LM_MODEL="anas-awadalla/mpt-1b-redpajama-200b-dolly"
#CROSS_ATTN_EVERY_N_LAYERS=1


SHOTS=$1
MASTER_PORT=$2
BS=$3
SIMILAR_IN_TOP_K=200
VISUAL_MODE="random"
#VISUAL_MODE="no_images"

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`
COMMENT="9B-rice-coco-shots-${SHOTS}"
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
    --visual_demo_mode $VISUAL_MODE \
    --rices \
    --rices_find_by_ranking_similar_text \
    --rices_find_by_ranking_similar_text_similar_in_top_k ${SIMILAR_IN_TOP_K} \
    --cached_demonstration_features ${COCO_RICE_FEAT} \
    --caption_shot_results  $CACHED_SHOT_RESULT \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai \
    --eval_coco \
    --coco_train_image_dir_path ${COCO_IMG_TRAIN_PATH} \
    --coco_val_image_dir_path ${COCO_IMG_VAL_PATH} \
    --coco_karpathy_json_path ${COCO_KARPATHY_PATH} \
    --coco_annotations_json_path ${COCO_ANNO_PATH}
