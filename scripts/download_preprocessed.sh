#!/usr/bin/env bash

# Processed image features for VisDial v1.0
# To generate these files, look in the preprocessing folder and the corresponding section in the README

wget -r https://s3.amazonaws.com/visdial-bert/data/visdial_image_feats.lmdb -O data/visdial/visdial_img_feat.lmdb

# Processed dialog data for VisDial v1.0 
wget https://s3.amazonaws.com/visdial-bert/data/visdial_1.0_train_processed.json -O data/visdial/visdial_1.0_train_processed.json
wget https://s3.amazonaws.com/visdial-bert/data/visdial_1.0_val_processed.json -O data/visdial/visdial_1.0_val_processed.json
wget https://s3.amazonaws.com/visdial-bert/data/visdial_1.0_test_processed.json -O data/visdial/visdial_1.0_test_processed.json

# Samples on the train split with the dense annotations
wget https://s3.amazonaws.com/visdial-bert/data/visdial_1.0_train_dense_processed.json -O data/visdial/visdial_1.0_train_dense_processed.json

# Processed Dense Annotations
wget https://s3.amazonaws.com/visdial-bert/data/visdial_1.0_train_dense_annotations_processed.json -O data/visdial/visdial_1.0_train_dense_annotations_processed.json
wget https://s3.amazonaws.com/visdial-bert/data/visdial_1.0_val_dense_annotations_processed.json -O data/visdial/visdial_1.0_val_dense_annotations_processed.json