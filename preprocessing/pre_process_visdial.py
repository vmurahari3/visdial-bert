import os
import concurrent.futures
import json
import argparse
import glob
import importlib  
import sys
from pytorch_transformers.tokenization_bert import BertTokenizer

import torch

def read_options(argv=None):
    parser = argparse.ArgumentParser(description='Options')    
    #-------------------------------------------------------------------------
    # Data input settings
    parser.add_argument('-visdial_train', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_train.json', help='json file containing train split of visdial data')

    parser.add_argument('-visdial_val', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_val.json',
                            help='json file containing val split of visdial data')
    parser.add_argument('-visdial_test', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_test.json',
                            help='json file containing test split of visdial data')
    parser.add_argument('-visdial_val_ndcg', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_val_dense_annotations.json',
                            help='JSON file with dense annotations')
    parser.add_argument('-visdial_train_ndcg', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_train_dense_annotations.json',
                            help='JSON file with dense annotations')

    parser.add_argument('-max_seq_len', default=256, type=int,
                            help='the max len of the input representation of the dialog encoder')
    #-------------------------------------------------------------------------
    # Logging settings

    parser.add_argument('-save_path_train', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_train_processed.json',
                            help='Path to save processed train json')
    parser.add_argument('-save_path_val', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_val_processed.json',
                            help='Path to save val json')
    parser.add_argument('-save_path_test', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_test_processed.json',
                            help='Path to save test json')

    parser.add_argument('-save_path_train_dense_samples', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_train_dense_processed.json',
                            help='Path to save processed train json')
    parser.add_argument('-save_path_val_ndcg', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_val_dense_annotations_processed.json',
                            help='Path to save processed ndcg data for the val split')
    parser.add_argument('-save_path_train_ndcg', default='/srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_train_dense_annotations_processed.json',
                            help='Path to save processed ndcg data for the train split')

    try:
        parsed = vars(parser.parse_args(args=argv))
    except IOError as msg:
        parser.error(str(msg)) 
    return parsed

if __name__ == "__main__":
    params = read_options() 
    # read all the three splits 

    f = open(params['visdial_train'])
    input_train = json.load(f)
    input_train_data = input_train['data']['dialogs']
    train_questions = input_train['data']['questions']
    train_answers = input_train['data']['answers']
    f.close()

    # read train dense annotations
    f = open(params['visdial_train_ndcg'])
    input_train_ndcg = json.load(f)
    f.close()

    f = open(params['visdial_val'])
    input_val = json.load(f)
    input_val_data = input_val['data']['dialogs']
    val_questions = input_val['data']['questions']
    val_answers = input_val['data']['answers'] 
    f.close()

    f = open(params['visdial_val_ndcg'])
    input_val_ncdg = json.load(f)
    f.close()
    
    f = open(params['visdial_test'])
    input_test = json.load(f)
    input_test_data = input_test['data']['dialogs']
    test_questions = input_test['data']['questions']
    test_answers = input_test['data']['answers'] 
    f.close()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    max_seq_len = params["max_seq_len"]
    num_illegal_train = 0
    num_illegal_val = 0
    num_illegal_test = 0
    # process train
    i = 0
    while i < len(input_train_data):
        cur_dialog = input_train_data[i]['dialog']
        caption = input_train_data[i]['caption']
        tot_len = 22 + len(tokenizer.encode(caption)) # account for 21 sep tokens, CLS token and caption
        for rnd in range(len(cur_dialog)):
            tot_len += len(tokenizer.encode(train_answers[cur_dialog[rnd]['answer']]))
            tot_len += len(tokenizer.encode(train_questions[cur_dialog[rnd]['question']]))
        if tot_len <= max_seq_len:
            i += 1
        else:
            input_train_data.pop(i)
            num_illegal_train += 1

    train_data_dense = []
    train_img_id_to_index = {input_train_data[i]['image_id']: i for i in range(len(input_train_data))}
    # pre process dense annotations on train
    i = 0
    while i < len(input_train_ndcg):
        remove = False
        index = None
        train_sample = None
        if input_train_ndcg[i]['image_id'] in train_img_id_to_index:
            img_id = input_train_ndcg[i]['image_id']
            index = train_img_id_to_index[img_id]
            cur_round = input_train_ndcg[i]['round_id'] - 1
            train_sample = input_train_data[index]
            # check if the sample is legal
            caption = train_sample['caption']
            tot_len = 1 # CLS token
            tot_len += len(tokenizer.encode(caption)) + 1
            for rnd in range(cur_round):
                tot_len += len(tokenizer.encode(train_questions[cur_dialog[rnd]['question']])) + 1
                if rnd != input_train_ndcg[i]['round_id']:
                    tot_len += len(tokenizer.encode(train_answers[cur_dialog[rnd]['answer']])) + 1
            for option in train_sample['dialog'][cur_round]['answer_options']:
                cur_len = len(tokenizer.encode(train_answers[option])) + 1 + tot_len
                if cur_len > max_seq_len:
                    print("image id", img_id)
                    remove = True
                    break
        else:
            remove = True

        if remove:
            input_train_ndcg.pop(i)
        else:
            train_data_dense.append(train_sample.copy())
            i += 1
    
    assert len(input_train_ndcg) == len(train_data_dense)
    print(len(input_train_ndcg))
    input_train_dense = input_train.copy()
    input_train_dense['data']['dialogs'] = train_data_dense

    # process val
    i = 0
    while i <  len(input_val_data):
        remove = False
        cur_dialog = input_val_data[i]['dialog']
        caption = input_val_data[i]['caption']
        tot_len = 1 # CLS token
        tot_len += len(tokenizer.encode(caption)) + 1
        for rnd in range(len(cur_dialog)):
            tot_len += len(tokenizer.encode(val_questions[cur_dialog[rnd]['question']])) + 1
            for option in cur_dialog[rnd]['answer_options']:
                cur_len = len(tokenizer.encode(val_answers[option])) + 1 + tot_len
                if cur_len > max_seq_len:
                    input_val_data.pop(i)
                    input_val_ncdg.pop(i)
                    num_illegal_val += 1
                    remove = True
                    break
            if not remove:
                tot_len += len(tokenizer.encode(val_answers[cur_dialog[rnd]['answer']])) + 1
            else:
                break
        if not remove:
            i += 1
    
    i = 0
    # process test
    while i <  len(input_test_data):
        remove = False
        cur_dialog = input_test_data[i]['dialog']
        input_test_data[i]['round_id'] = len(cur_dialog)
        caption = input_test_data[i]['caption']
        tot_len = 1 # CLS token
        tot_len += len(tokenizer.encode(caption)) + 1
        for rnd in range(len(cur_dialog)):
            tot_len += len(tokenizer.encode(test_questions[cur_dialog[rnd]['question']])) + 1
            if rnd != len(cur_dialog)-1:
                tot_len += len(tokenizer.encode(test_answers[cur_dialog[rnd]['answer']])) + 1
        
        max_len_cur_sample = tot_len

        for option in cur_dialog[-1]['answer_options']:
            cur_len = len(tokenizer.encode(test_answers[option])) + 1 + tot_len
            if cur_len > max_seq_len:
                print("image id", input_test_data[i]['image_id'])
                print(cur_len)
                print(len(cur_dialog))
                # print(cur_dialog)
                remove = True
                if max_len_cur_sample < cur_len:
                    max_len_cur_sample = cur_len
        if remove:
            # need to process this sample by removing a few rounds
            num_illegal_test += 1
            while max_len_cur_sample > max_seq_len:
                cur_round_len = len(tokenizer.encode(test_questions[cur_dialog[0]['question']])) + 1 +  \
                    len(tokenizer.encode(test_answers[cur_dialog[0]['answer']])) + 1
                cur_dialog.pop(0)
                max_len_cur_sample -= cur_round_len
            # print("truncated dialog", cur_dialog)
        
        i += 1
    '''
    # store processed files
    '''
    with open(params['save_path_train'],'w') as train_out_file:
        json.dump(input_train, train_out_file)

    with open(params['save_path_val'],'w') as val_out_file:
        json.dump(input_val, val_out_file)
    with open(params['save_path_val_ndcg'],'w') as val_ndcg_out_file:
        json.dump(input_val_ncdg, val_ndcg_out_file)    
    with open(params['save_path_test'],'w') as test_out_file:
        json.dump(input_test, test_out_file)

    with open(params['save_path_train_dense_samples'],'w') as train_dense_out_file:
        json.dump(input_train_dense, train_dense_out_file)

    with open(params['save_path_train_ndcg'],'w') as train_ndcg_out_file:
        json.dump(input_train_ndcg, train_ndcg_out_file)

    # spit stats

    print("number of illegal train samples", num_illegal_train)
    print("number of illegal val samples", num_illegal_val)
    print("number of illegal test samples", num_illegal_test)