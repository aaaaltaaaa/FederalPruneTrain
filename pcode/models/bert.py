from collections import namedtuple
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import (
    BertForSequenceClassification,
    BertForMultipleChoice,
)
from transformers.modeling_roberta import (
    RobertaForSequenceClassification,
    RobertaForMultipleChoice,
    RobertaModel,
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
)
from transformers import DistilBertModel
from transformers.configuration_roberta import RobertaConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer
import torch
import torch.nn as nn

__all__ = ["distilbert"]

Classes = namedtuple("Classes", "seqcls postag multiplechoice tokenizer")


class LinearPredictor(BertPreTrainedModel):
    def __init__(self, bert_config, out_dim, dropout):
        super(LinearPredictor, self).__init__(bert_config)
        self.bert = BertModel(bert_config)
        self.classifier = nn.Linear(768, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(self):
        raise NotImplementedError


class BertPOSTagger(LinearPredictor):
    def __init__(self, bert_config, out_dim, dropout=0.1):
        super(BertPOSTagger, self).__init__(bert_config, out_dim, dropout)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            if_tgts=None,
            **kwargs,
    ):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        bert_out = bert_out[0]
        bert_out = self.dropout(bert_out)
        bert_out = self.classifier(bert_out)

        logits = []
        for b_step in range(input_ids.shape[0]):
            preds = bert_out[b_step][if_tgts[b_step]]
            logits.append(preds)
        return (torch.cat(logits, dim=0), torch.argmax(bert_out, dim=-1, keepdim=False))


class RobertaPOSTagger(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, roberta_config, out_dim, dropout=0.1):
        super().__init__(roberta_config)
        self.roberta = RobertaModel(roberta_config)
        self.classifier = nn.Linear(768, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            if_tgts=None,
            **kwargs,
    ):
        roberta_out = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        roberta_out = roberta_out[0]
        roberta_out = self.dropout(roberta_out)
        roberta_out = self.classifier(roberta_out)

        logits = []
        for b_step in range(input_ids.shape[0]):
            preds = roberta_out[b_step][if_tgts[b_step]]
            logits.append(preds)
        return (
            torch.cat(logits, dim=0),
            torch.argmax(roberta_out, dim=-1, keepdim=False),
        )


from transformers.modeling_distilbert import DistilBertPreTrainedModel


class DistilBertCLS(DistilBertPreTrainedModel):
    def __init__(self,config, projection=False):
        super(DistilBertCLS, self).__init__(config)

        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.projection = projection
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        if self.projection:
            self.projection_layer = nn.Linear(config.dim,256)
            self.classifier = nn.Linear(256,config.num_labels)
        else:
            self.classifier = nn.Linear(config.dim, config.num_labels)
        print(config.num_labels)

    def forward(self,input,start_layer_idx = 0):
        if start_layer_idx >= 0:
            input_ids, attention_mask = input
            distilbert_output = self.distilbert(input_ids=input_ids,
                                                attention_mask=attention_mask)

            hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)

            if self.projection:
                pooled_output = self.projection_layer(pooled_output)
        else:
            pooled_output = input
        logits = self.classifier(pooled_output)  # (bs, dim)

        return pooled_output,logits

from transformers import DistilBertConfig

def distilbert(conf):
    dataset = conf.data
    class_num = 4

    if "ag_news" in dataset:
        class_num = 4
    elif "sst" in dataset:
        class_num = 5
    elif "dbpedia" in dataset:
        class_num = 14

    bert_config = DistilBertConfig(num_labels=class_num)

    model = DistilBertCLS(bert_config,conf.projection)

    return model
