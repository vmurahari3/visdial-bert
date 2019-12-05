import torch
from torch.utils import data
import json
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
import random
from utils.data_utils import list2tensorpad, encode_input, encode_image_input
from utils.image_features_reader import ImageFeaturesH5Reader

class VisdialDatasetDense(data.Dataset):

    def __init__(self, params):
        'Initialization'
        self.numDataPoints = {}
        num_samples_train = params['num_train_samples']
        num_samples_val = params['num_val_samples']
        self._image_features_reader = ImageFeaturesH5Reader(params['visdial_image_feats'])
        with open(params['visdial_processed_train_dense']) as f:
            self.visdial_data_train = json.load(f)
            if params['overfit']:
                if num_samples_train:
                    self.numDataPoints['train'] = num_samples_train
                else:                
                    self.numDataPoints['train'] = 5
            else:
                if num_samples_train:
                    self.numDataPoints['train'] = num_samples_train
                else:
                    self.numDataPoints['train'] = len(self.visdial_data_train['data']['dialogs'])

        with open(params['visdial_processed_val']) as f:
            self.visdial_data_val = json.load(f)
            if params['overfit']:
                if num_samples_val:
                    self.numDataPoints['val'] = num_samples_val
                else:
                    self.numDataPoints['val'] = 5
            else:
                if num_samples_val:
                    self.numDataPoints['val'] = num_samples_val
                else:
                    self.numDataPoints['val'] = len(self.visdial_data_val['data']['dialogs'])

        self.overfit = params['overfit']

        with open(params['visdial_processed_train_dense_annotations']) as f:
            self.visdial_data_train_ndcg = json.load(f)
        with open(params['visdial_processed_val_dense_annotations']) as f:
            self.visdial_data_val_ndcg = json.load(f)

        #train val setup
        self.numDataPoints['trainval'] = self.numDataPoints['train'] + self.numDataPoints['val']

        self.num_options = params["num_options"]
        self._split = 'train'
        self.subsets = ['train', 'val', 'trainval']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
        # fetching token indicecs of [CLS] and [SEP]
        tokens = ['[CLS]','[MASK]','[SEP]']
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        self.CLS = indexed_tokens[0]
        self.MASK = indexed_tokens[1]
        self.SEP  = indexed_tokens[2]
        self.params = params
        self._max_region_num = 37

    def __len__(self):
        return self.numDataPoints[self._split]
    
    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets
        self._split = split

    def __getitem__(self, index):

        def pruneRounds(context, num_rounds):
            start_segment = 1
            len_context = len(context)
            cur_rounds = (len(context) // 2) + 1
            l_index = 0
            if cur_rounds > num_rounds:
                # caption is not part of the final input
                l_index = len_context - (2 * num_rounds)
                start_segment = 0   
            return context[l_index:], start_segment
     
        # Combining all the dialog rounds with the [SEP] and [CLS] token
        MAX_SEQ_LEN = self.params['max_seq_len'] 
        cur_data = None
        cur_dense_annotations = None
        if self._split == 'train':
            cur_data = self.visdial_data_train['data']
            cur_dense_annotations = self.visdial_data_train_ndcg
        elif self._split == 'val':
            if self.overfit:
                cur_data = self.visdial_data_train['data']
                cur_dense_annotations = self.visdial_data_train_ndcg
            else:
                cur_data = self.visdial_data_val['data']
                cur_dense_annotations = self.visdial_data_val_ndcg
        else:
            if index >= self.numDataPoints['train']:
                cur_data = self.visdial_data_val
                cur_dense_annotations = self.visdial_data_val_ndcg
                index -= self.numDataPoints['train']
            else:
                cur_data = self.visdial_data_train
                cur_dense_annotations = self.visdial_data_train_ndcg
        # number of options to score on
        num_options = self.num_options
        assert num_options == 100

        dialog = cur_data['dialogs'][index]
        cur_questions = cur_data['questions']
        cur_answers = cur_data['answers']
        img_id = dialog['image_id']
        assert img_id == cur_dense_annotations[index]['image_id']

        cur_rnd_utterance = [self.tokenizer.encode(dialog['caption'])]
        options_all = []
        cur_rounds = cur_dense_annotations[index]['round_id'] 
        for rnd,utterance in enumerate(dialog['dialog'][:cur_rounds]):
            cur_rnd_utterance.append(self.tokenizer.encode(cur_questions[utterance['question']]))
            if rnd != cur_rounds - 1:
                cur_rnd_utterance.append(self.tokenizer.encode(cur_answers[utterance['answer']]))
        for answer_option in dialog['dialog'][cur_rounds - 1]['answer_options']:
            cur_option = cur_rnd_utterance.copy()
            cur_option.append(self.tokenizer.encode(cur_answers[answer_option]))
            options_all.append(cur_option)
            assert len(cur_option) == 2 * cur_rounds + 1

        gt_option = dialog['dialog'][cur_rounds - 1]['gt_index']

        tokens_all = []
        mask_all = []
        segments_all = []
        sep_indices_all = []
        hist_len_all = []
        
        for _, option in enumerate(options_all):
            option, start_segment = pruneRounds(option, self.params['visdial_tot_rounds'])
            tokens, segments, sep_indices, mask = encode_input(option, start_segment ,self.CLS, 
            self.SEP, self.MASK ,max_seq_len=MAX_SEQ_LEN, mask_prob=0)

            tokens_all.append(tokens)
            mask_all.append(mask)
            segments_all.append(segments)
            sep_indices_all.append(sep_indices)
            hist_len_all.append(torch.LongTensor([len(option)-1]))
            
        tokens_all = torch.cat(tokens_all,0)
        mask_all = torch.cat(mask_all,0)
        segments_all = torch.cat(segments_all, 0)
        sep_indices_all = torch.cat(sep_indices_all, 0)
        hist_len_all = torch.cat(hist_len_all,0)

        item = {}
        item['tokens'] = tokens_all.unsqueeze(0)
        item['segments'] = segments_all.unsqueeze(0)
        item['sep_indices'] = sep_indices_all.unsqueeze(0)
        item['mask'] = mask_all.unsqueeze(0)
        item['hist_len'] = hist_len_all.unsqueeze(0)
        item['image_id'] = torch.LongTensor([img_id])

        # add image features. Expand them to create batch * num_rounds * num options * num bbox * img feats
        features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
        features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, \
                                    image_target, max_regions=self._max_region_num, mask_prob=0)

        item['image_feat'] = features
        item['image_loc'] = spatials
        item['image_mask'] = image_mask
        item['image_target'] = image_target
        item['image_label'] = image_label

        # add dense annotation fields
        item['gt_relevance_round_id'] = torch.LongTensor([cur_rounds])
        item['gt_relevance'] = torch.Tensor(cur_dense_annotations[index]['relevance'])
        item['gt_option'] = torch.LongTensor([gt_option])

        # add next sentence labels for training with the nsp loss as well
        nsp_labels = torch.ones(*tokens_all.unsqueeze(0).shape[:-1])
        nsp_labels[:,gt_option] = 0
        item['next_sentence_labels'] = nsp_labels.long()

        return item