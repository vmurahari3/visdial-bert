import copy
import json
import torch
from torch import nn
from models.vilbert_dialog import BertForMultiModalPreTraining, BertConfig

class VisualDialogEncoder(nn.Module):

    def __init__(self, config_path):
        super(VisualDialogEncoder, self).__init__()
        config = BertConfig.from_json_file(config_path)

        self.bert_pretrained = BertForMultiModalPreTraining.from_pretrained('bert-base-uncased',config)
        self.bert_pretrained.train()
        
    def forward(self, input_ids, image_feat, image_loc, sep_indices=None, sep_len=None, token_type_ids=None,
         attention_mask=None, masked_lm_labels=None, next_sentence_label=None, head_mask=None,random_round_indices=None,
         output_nsp_scores=False, output_lm_scores=False,image_attention_mask=None,image_label=None, image_target=None):                      

        masked_lm_loss = None
        masked_img_loss = None
        nsp_loss = None
        prediction_scores_t = None
        seq_relationship_score = None

        if next_sentence_label is not None and masked_lm_labels \
            is not None and image_target is not None:
            # train mode, output losses
            masked_lm_loss, masked_img_loss, nsp_loss, _, prediction_scores_t, seq_relationship_score  = \
                self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, \
                 token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                            next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                                image_label=image_label, image_target=image_target)
        else:
            #inference, output scores
            prediction_scores_t, _, seq_relationship_score, _, _ = \
                self.bert_pretrained(input_ids, image_feat, image_loc, sep_indices=sep_indices, sep_len=sep_len, \
                    token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels, \
                    next_sentence_label=next_sentence_label, image_attention_mask=image_attention_mask,\
                    image_label=image_label, image_target=image_target)

        out = (masked_lm_loss, masked_img_loss, nsp_loss)

        if output_nsp_scores:
            out  = out + (seq_relationship_score,)
        if output_lm_scores:
            out = out + (prediction_scores_t,)
        return out