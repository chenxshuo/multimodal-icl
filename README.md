# Can Multimodal Large Language Models Truly Perform Multimodal In-Context Learning?

The offical code repository for paper **Can Multimodal Large Language Models Truly Perform Multimodal In-Context Learning?** (WACV2025)

# Installation 

```bash
git clone git@github.com:chenxshuo/multimodal-icl.git
cd multimodal-icl
pip install -r requirements.txt
pip install -e .
```

# Folder Structure 

```
./open_flamingo
├── eval
│   ├── __init__.py
│   ├── classification_utils.py
│   ├── coco_metric.py
│   ├── data
│   ├── eval_datasets.py
│   ├── eval_model.py
│   ├── evaluate.py # main entrance of evaluating tasks 
│   ├── models
│   ├── ok_vqa_utils.py
│   ├── rices.py # implementation of rices and mmices 
│   ├── rices_text.py
│   ├── utils.py
│   └── vqa_metric.py
├── scripts 
│   ├── mmices # to reproduce mmices 
│   ├── no_images # to reproduce experiments that remove context images 
│   ├── ori # to reproduce normal multimodal icl 
│   └── text_setting # to reproduce experiments with different context text settings 
└── src # model implementation 
    ├── __init__.py
    ├── factory.py
    ├── flamingo.py
    ├── flamingo_lm.py
    ├── helpers.py
    └── utils.py
```

# Usage 

## Random Selection and RICES 

```bash
# icl random selection, vqav2, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/reproduction/run_eval_9B_vqav2.sh 4 26000 64

# icl random selection, gqa, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/reproduction/run_eval_9B_gqa.sh 4 26000 64

# icl random selection, okvqa, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/reproduction/run_eval_9B_okvqa.sh 4 26000 64

# icl random selection, coco, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/reproduction/run_eval_9B_coco.sh 4 26000 64


# icl rices, vqav2, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/rices/run_eval_9B_vqav2.sh 4 26000 64

# icl rices, gqa, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/rices/run_eval_9B_gqa.sh 4 26000 64

# icl rices, okvqa, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/rices/run_eval_9B_okvqa.sh 4 26000 64

# icl rices, coco, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/rices/run_eval_9B_coco.sh 4 26000 64


```

## Removing context images 

```bash
# run eval on vqav2, shot 4, port 26001, batch size 16, using blank images as context demos 
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh 4 26001 16 blank_images

# run eval on gqa, shot 4, port 26001, batch size 16, using blank images as context demos 
bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh 4 26001 16 blank_images


# run eval on okvqa, shot 4, port 26001, batch size 16, using blank images as context demos 
bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh 4 26001 16 blank_images


# run eval on coco, shot 4, port 26001, batch size 16, using blank images as context demos 
bash open_flamingo/scripts/no_images/run_eval_9B_coco.sh 4 26001 16 blank_images

```

## MMICES

```bash


# icl mmices, vqav2, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/mmices/run_eval_9B_vqav2.sh 4 26000 64

# icl mmices, gqa, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/mmices/run_eval_9B_gqa.sh 4 26000 64

# icl mmices, okvqa, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/mmices/run_eval_9B_okvqa.sh 4 26000 64

# icl mmices, coco, shot 4, port 26000, batch size 64 
bash open_flamingo/scripts/mmices/run_eval_9B_coco.sh 4 26000 64

```

# Acknowledgment 
This repo is built on [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)

# Citation 
Please cite our paper if you find this repo useful in your research:

```bibtex
@article{chen2023understanding,
title={Can Multimodal Large Language Models Truly Perform Multimodal In-Context Learning?},
author={Chen, Shuo and Han, Zhen and He, Bailan and Buckley, Mark and Torr, Philip and Tresp, Volker and Gu, Jindong},
journal={arXiv preprint arXiv:2311.18021},
year={2023}
}
```