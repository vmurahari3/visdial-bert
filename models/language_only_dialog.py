import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_transformers.modeling_bert import BertForPreTraining, BertPredictionHeadTransform, BertEmbeddings, BertPreTrainingHeads, BertLayerNorm, \
    BertModel, BertEncoder, BertPooler
from utils.data_utils import sequence_mask

class BertEmbeddingsDialog(BertEmbeddings):
    def __init__(self, config):
        super(BertEmbeddingsDialog, self).__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # add support for additional segment embeddings. Supporting 10 additional embedding as of now
        self.token_type_embeddings_extension = nn.Embedding(10,config.hidden_size)
        # adding specialized embeddings for sep tokens
        self.sep_embeddings = nn.Embedding(50,config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        
    def forward(self, input_ids, sep_indices=None, sep_len=None, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        token_type_ids_extension = token_type_ids - self.config.type_vocab_size
        token_type_ids_extension_mask = (token_type_ids_extension >= 0).float()
        token_type_ids_extension = (token_type_ids_extension.float() * token_type_ids_extension_mask).long()

        token_type_ids_mask = (token_type_ids < self.config.type_vocab_size).float()
        assert torch.sum(token_type_ids_extension_mask + token_type_ids_mask) ==  \
            torch.numel(token_type_ids) == torch.numel(token_type_ids_mask)
        token_type_ids = (token_type_ids.float() * token_type_ids_mask).long()

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        token_type_embeddings_extension = self.token_type_embeddings_extension(token_type_ids_extension)
        
        token_type_embeddings = (token_type_embeddings * token_type_ids_mask.unsqueeze(-1)) + \
                                (token_type_embeddings_extension * token_type_ids_extension_mask.unsqueeze(-1))
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModelDialog(BertModel):

    def __init__(self, config):
        super(BertModelDialog, self).__init__(config)

        self.embeddings = BertEmbeddingsDialog(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, sep_indices=None, sep_len=None, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, sep_indices=sep_indices, sep_len=sep_len, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class BertForPretrainingDialog(BertForPreTraining):
    def __init__(self, config):
        super(BertForPretrainingDialog, self).__init__(config)
        self.bert = BertModelDialog(config)
        self.cls = BertPreTrainingHeads(config)
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, sep_indices=None, sep_len=None, token_type_ids=None,attention_mask=None,
        masked_lm_labels=None, next_sentence_label=None, position_ids=None, head_mask=None):

        outputs = self.bert(input_ids, sep_indices=sep_indices, sep_len=sep_len,token_type_ids=token_type_ids\
                    ,attention_mask=attention_mask)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)
