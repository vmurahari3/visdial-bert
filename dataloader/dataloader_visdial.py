import torch
from torch.utils import data
import json
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
import random
from utils.data_utils import list2tensorpad, encode_input, encode_image_input
from utils.image_features_reader import ImageFeaturesH5Reader
class VisdialDataset(data.Dataset):

    def __init__(self, params):

        self.numDataPoints = {}
        num_samples_train = params['num_train_samples']
        num_samples_val = params['num_val_samples']
        self._image_features_reader = ImageFeaturesH5Reader(params['visdial_image_feats'])
        with open(params['visdial_processed_train']) as f:
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
        with open(params['visdial_processed_test']) as f:
            self.visdial_data_test = json.load(f)
            self.numDataPoints['test'] = len(self.visdial_data_test['data']['dialogs'])

        self.overfit = params['overfit']
        with open(params['visdial_processed_val_dense_annotations']) as f:
            self.visdial_data_val_dense = json.load(f)
        self.num_options = params["num_options"]
        self._split = 'train'
        self.subsets = ['train','val','test']
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

        def tokens2str(seq):
            dialog_sequence = ''
            for sentence in seq:
                for word in sentence:
                    dialog_sequence += self.tokenizer._convert_id_to_token(word) + " "
                dialog_sequence += ' </end> '
            dialog_sequence = dialog_sequence.encode('utf8')
            return dialog_sequence

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
        if self._split == 'train':
            cur_data = self.visdial_data_train['data']
        elif self._split == 'val':
            if self.overfit:
                cur_data = self.visdial_data_train['data']
            else:
                cur_data = self.visdial_data_val['data']
        else:
            cur_data = self.visdial_data_test['data']
        
        # number of options to score on
        num_options = self.num_options
        assert num_options > 1 and num_options <= 100
        
        dialog = cur_data['dialogs'][index]
        cur_questions = cur_data['questions']
        cur_answers = cur_data['answers']
        img_id = dialog['image_id']

        if self._split == 'train':
            utterances = []
            utterances_random = []
            tokenized_caption = self.tokenizer.encode(dialog['caption']) 
            utterances.append([tokenized_caption])
            utterances_random.append([tokenized_caption])
            tot_len = len(tokenized_caption) + 2 # add a 1 for the CLS token as well as the sep tokens which follows the caption
            for rnd,utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance = utterances[-1].copy()
                cur_rnd_utterance_random = utterances[-1].copy()
                
                tokenized_question = self.tokenizer.encode(cur_questions[utterance['question']])
                tokenized_answer = self.tokenizer.encode(cur_answers[utterance['answer']])
                cur_rnd_utterance.append(tokenized_question)
                cur_rnd_utterance.append(tokenized_answer)

                question_len = len(tokenized_question)
                answer_len = len(tokenized_answer)
                tot_len += question_len + 1 # the additional 1 is for the sep token
                tot_len += answer_len + 1 # the additional 1 is for the sep token

                cur_rnd_utterance_random.append(self.tokenizer.encode(cur_questions[utterance['question']]))
                # randomly select one random utterance in that round
                utterances.append(cur_rnd_utterance)

                num_inds = len(utterance['answer_options'])
                gt_option_ind = utterance['gt_index']

                negative_samples = []

                for _ in range(self.params["num_negative_samples"]):

                    all_inds = list(range(100))
                    all_inds.remove(gt_option_ind)
                    all_inds = all_inds[:(num_options-1)]
                    tokenized_random_utterance = None
                    option_ind = None

                    while len(all_inds):
                        option_ind = random.choice(all_inds)
                        tokenized_random_utterance = self.tokenizer.encode(cur_answers[utterance['answer_options'][option_ind]])
                        # the 1 here is for the sep token at the end of each utterance
                        if(MAX_SEQ_LEN >= (tot_len + len(tokenized_random_utterance) + 1)):
                            break
                        else:
                            all_inds.remove(option_ind)
                    if len(all_inds) == 0:
                        # all the options exceed the max len. Truncate the last utterance in this case.
                        tokenized_random_utterance = tokenized_random_utterance[:answer_len]
                    t = cur_rnd_utterance_random.copy()
                    t.append(tokenized_random_utterance)
                    negative_samples.append(t)

                utterances_random.append(negative_samples)
            # removing the caption in the beginning
            utterances = utterances[1:]
            utterances_random = utterances_random[1:]
            assert len(utterances) == len(utterances_random) == 10

            tokens_all_rnd = []
            mask_all_rnd = []
            segments_all_rnd = []
            sep_indices_all_rnd = []
            next_labels_all_rnd = []
            hist_len_all_rnd = []

            for j,context in enumerate(utterances):
                tokens_all = []
                mask_all = []
                segments_all = []
                sep_indices_all = []
                next_labels_all = []
                hist_len_all = []

                context, start_segment = pruneRounds(context, self.params['visdial_tot_rounds'])
                # print("{}: {}".format(j, tokens2str(context)))
                tokens, segments, sep_indices, mask = encode_input(context, start_segment, self.CLS,
                 self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=self.params["mask_prob"])
                tokens_all.append(tokens)
                mask_all.append(mask)
                sep_indices_all.append(sep_indices)
                next_labels_all.append(torch.LongTensor([0]))
                segments_all.append(segments)
                hist_len_all.append(torch.LongTensor([len(context)-1]))
                negative_samples = utterances_random[j]

                for context_random in negative_samples:
                    context_random, start_segment = pruneRounds(context_random, self.params['visdial_tot_rounds'])
                    # print("{}: {}".format(j, tokens2str(context_random)))
                    tokens_random, segments_random, sep_indices_random, mask_random = encode_input(context_random, start_segment, self.CLS, 
                    self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN, mask_prob=self.params["mask_prob"])
                    tokens_all.append(tokens_random)
                    mask_all.append(mask_random)
                    sep_indices_all.append(sep_indices_random)
                    next_labels_all.append(torch.LongTensor([1]))
                    segments_all.append(segments_random)
                    hist_len_all.append(torch.LongTensor([len(context_random)-1]))

                tokens_all_rnd.append(torch.cat(tokens_all,0).unsqueeze(0))
                mask_all_rnd.append(torch.cat(mask_all,0).unsqueeze(0))
                segments_all_rnd.append(torch.cat(segments_all, 0).unsqueeze(0))
                sep_indices_all_rnd.append(torch.cat(sep_indices_all, 0).unsqueeze(0))
                next_labels_all_rnd.append(torch.cat(next_labels_all, 0).unsqueeze(0))
                hist_len_all_rnd.append(torch.cat(hist_len_all,0).unsqueeze(0))

            tokens_all_rnd = torch.cat(tokens_all_rnd,0)
            mask_all_rnd = torch.cat(mask_all_rnd,0)
            segments_all_rnd = torch.cat(segments_all_rnd, 0)
            sep_indices_all_rnd = torch.cat(sep_indices_all_rnd, 0)
            next_labels_all_rnd = torch.cat(next_labels_all_rnd, 0)
            hist_len_all_rnd = torch.cat(hist_len_all_rnd,0)
                   
            item = {}

            item['tokens'] = tokens_all_rnd
            item['segments'] = segments_all_rnd
            item['sep_indices'] = sep_indices_all_rnd
            item['mask'] = mask_all_rnd
            item['next_sentence_labels'] = next_labels_all_rnd
            item['hist_len'] = hist_len_all_rnd
            
            # get image features
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
            features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, image_target, max_regions=self._max_region_num)
            item['image_feat'] = features
            item['image_loc'] = spatials
            item['image_mask'] = image_mask
            item['image_target'] = image_target
            item['image_label'] = image_label
            return item
        
        elif self.split == 'val':
            # append all the 100 options and return all the 100 options concatenated with history
            # that will lead to 1000 forward passes for a single image
            gt_relevance = None
            utterances = []
            gt_option_inds = []
            utterances.append([self.tokenizer.encode(dialog['caption'])])            
            options_all = []
            for rnd,utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance = utterances[-1].copy()
                cur_rnd_utterance.append(self.tokenizer.encode(cur_questions[utterance['question']]))
                # current round
                gt_option_ind = utterance['gt_index']
                option_inds = []
                option_inds.append(gt_option_ind) 
                all_inds = list(range(100))
                all_inds.remove(gt_option_ind)
                all_inds = all_inds[:(num_options-1)]
                option_inds.extend(all_inds)
                gt_option_inds.append(0)
                cur_rnd_options = []
                answer_options = [utterance['answer_options'][k] for k in option_inds]
                assert len(answer_options) == len(option_inds) == num_options
                assert answer_options[0] == utterance['answer']

                if rnd == self.visdial_data_val_dense[index]['round_id'] - 1:
                    gt_relevance = torch.Tensor(self.visdial_data_val_dense[index]['gt_relevance'])
                    # shuffle based on new indices
                    gt_relevance = gt_relevance[torch.LongTensor(option_inds)]
                for answer_option in answer_options:
                    cur_rnd_cur_option = cur_rnd_utterance.copy()
                    cur_rnd_cur_option.append(self.tokenizer.encode(cur_answers[answer_option]))
                    cur_rnd_options.append(cur_rnd_cur_option)
                cur_rnd_utterance.append(self.tokenizer.encode(cur_answers[utterance['answer']]))
                utterances.append(cur_rnd_utterance)
                options_all.append(cur_rnd_options)
            # encode the input and create batch x 10 x 100 * max_len arrays (batch x num_rounds x num_options)            
            tokens_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            hist_len_all = []
            
            for rnd,cur_rnd_options in enumerate(options_all):

                tokens_all_rnd = []
                mask_all_rnd = []
                segments_all_rnd = []
                sep_indices_all_rnd = []
                hist_len_all_rnd = []                
                
                for j,cur_rnd_option in enumerate(cur_rnd_options):
                    cur_rnd_option, start_segment = pruneRounds(cur_rnd_option, self.params['visdial_tot_rounds'])
                    tokens, segments, sep_indices, mask = encode_input(cur_rnd_option, start_segment,self.CLS, 
                    self.SEP, self.MASK ,max_seq_len=MAX_SEQ_LEN, mask_prob=0)

                    tokens_all_rnd.append(tokens)
                    mask_all_rnd.append(mask)
                    segments_all_rnd.append(segments)
                    sep_indices_all_rnd.append(sep_indices)
                    hist_len_all_rnd.append(torch.LongTensor([len(cur_rnd_option)-1]))
                
                tokens_all.append(torch.cat(tokens_all_rnd,0).unsqueeze(0))
                mask_all.append(torch.cat(mask_all_rnd,0).unsqueeze(0))
                segments_all.append(torch.cat(segments_all_rnd,0).unsqueeze(0))
                sep_indices_all.append(torch.cat(sep_indices_all_rnd,0).unsqueeze(0))
                hist_len_all.append(torch.cat(hist_len_all_rnd,0).unsqueeze(0))
            
            tokens_all = torch.cat(tokens_all,0)
            mask_all = torch.cat(mask_all,0)
            segments_all = torch.cat(segments_all, 0)
            sep_indices_all = torch.cat(sep_indices_all, 0)
            hist_len_all = torch.cat(hist_len_all,0)

            item = {}
            item['tokens'] = tokens_all
            item['segments'] = segments_all
            item['sep_indices'] = sep_indices_all
            item['mask'] = mask_all
            item['hist_len'] = hist_len_all
                        
            item['gt_option_inds'] = torch.LongTensor(gt_option_inds)            

            # return dense annotation data as well
            item['round_id'] = torch.LongTensor([self.visdial_data_val_dense[index]['round_id']])
            item['gt_relevance'] = gt_relevance

            # add image features. Expand them to create batch * num_rounds * num options * num bbox * img feats
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
            features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, \
                                     image_target, max_regions=self._max_region_num, mask_prob=0)

            item['image_feat'] = features
            item['image_loc'] = spatials
            item['image_mask'] = image_mask
            item['image_target'] = image_target
            item['image_label'] = image_label

            item['image_id'] = torch.LongTensor([img_id])
            
            return item
        
        else:
            assert num_options == 100
            cur_rnd_utterance = [self.tokenizer.encode(dialog['caption'])]
            options_all = []
            for rnd,utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance.append(self.tokenizer.encode(cur_questions[utterance['question']]))
                if rnd != len(dialog['dialog'])-1:
                    cur_rnd_utterance.append(self.tokenizer.encode(cur_answers[utterance['answer']]))
            for answer_option in dialog['dialog'][-1]['answer_options']:
                cur_option = cur_rnd_utterance.copy()
                cur_option.append(self.tokenizer.encode(cur_answers[answer_option]))
                options_all.append(cur_option)

            tokens_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            hist_len_all = []
            
            for j, option in enumerate(options_all):
                option, start_segment = pruneRounds(option, self.params['visdial_tot_rounds'])
                print("option: {} {}".format(j, tokens2str(option)))
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
            item['round_id'] = torch.LongTensor([dialog['round_id']])

            # add image features. Expand them to create batch * num_rounds * num options * num bbox * img feats
            features, num_boxes, boxes, _ , image_target = self._image_features_reader[img_id]
            features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes, boxes, \
                                     image_target, max_regions=self._max_region_num, mask_prob=0)

            item['image_feat'] = features
            item['image_loc'] = spatials
            item['image_mask'] = image_mask
            item['image_target'] = image_target
            item['image_label'] = image_label

            return item