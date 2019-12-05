#!/usr/bin/env bash

mkdir checkpoints-release

# BaseModel -- Pretrained on Conceptual Captions and VQA  
wget https://s3.amazonaws.com/visdial-bert/checkpoints/bestmodel_no_dense_finetuning -O checkpoints-release/basemodel

# BaseModel + Dense Annotation finetuning
wget https://s3.amazonaws.com/visdial-bert/checkpoints/bestmodel_plus_dense -O checkpoints-release/basemodel_dense

# BaseModel + Dense Annotation finetuning + NSP loss
wget https://s3.amazonaws.com/visdial-bert/checkpoints/bestmodel_plus_dense_plus_nsp -O checkpoints-release/basemodel_dense_nsp

# VQA ViLBERT pretrained weights
wget https://s3.amazonaws.com/visdial-bert/checkpoints/vqa_weights -O checkpoints-release/vqa_pretrained_weights

# Conceptual Caption ViLBERT pretrained weights
wget https://s3.amazonaws.com/visdial-bert/checkpoints/concep_cap_weights -O checkpoints-release/concep_cap_pretrained_weights
