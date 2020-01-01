import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_transformers.modeling_bert import BertPredictionHeadTransform
from models.language_only_dialog import BertForPretrainingDialog

class DialogEncoder(nn.Module):

    def __init__(self):
        super(DialogEncoder, self).__init__()
        self.bert_pretrained = BertForPretrainingDialog.from_pretrained('bert-base-uncased',output_hidden_states=True)
        self.bert_pretrained.train()
        # add additional layers for the inconsistency loss
        assert self.bert_pretrained.config.output_hidden_states == True
        
    def forward(self, input_ids, sep_indices=None, sep_len=None, token_type_ids=None, attention_mask=None, 
                 masked_lm_labels=None, next_sentence_label=None, head_mask=None, output_nsp_scores=False, output_lm_scores=False):

        outputs = self.bert_pretrained(input_ids,sep_indices=sep_indices, sep_len=sep_len, \
                 token_type_ids=token_type_ids, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels,
                 next_sentence_label=next_sentence_label, position_ids=None, head_mask=head_mask)
        
        loss = None
        if next_sentence_label is not None:
            loss, lm_scores, nsp_scores, hidden_states = outputs
        else:
            lm_scores, nsp_scores, hidden_states = outputs

        loss_fct = CrossEntropyLoss(ignore_index=-1)

        lm_loss = None
        nsp_loss = None
        if next_sentence_label is not None:
            nsp_loss = loss_fct(nsp_scores, next_sentence_label)
        if masked_lm_labels is not None:            
            lm_loss = loss_fct(lm_scores.view(-1,lm_scores.shape[-1]), masked_lm_labels.view(-1))

        out = (loss,lm_loss, nsp_loss)
        if output_nsp_scores:
            out  = out + (nsp_scores,)
        if output_lm_scores:
            out = out + (lm_scores,)
        return out