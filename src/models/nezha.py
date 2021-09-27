import torch
import torch.nn as nn

import sys
import numpy as np
import torch.nn.functional as F
sys.path.append('../../src')
from src.util.modeling.modeling_nezha.modeling import NeZhaPreTrainedModel, NeZhaModel


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2

    return loss


class NeZhaSequenceClassification_F(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.bert = NeZhaModel(config)
        self.rdrop_coef = 0.0
        self.r_dropout = nn.Sequential(nn.Dropout(0.3), nn.Dropout(0.3))
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        attention_mask = torch.ne(input_ids, 0)
        encoder_out, pooled_output, all_hidden_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = self.r_dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + (pooled_output,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # r-dropout
            if self.rdrop_coef > 0:
                encoder_out1, pooled_output1, all_hidden_outputs1 = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                pooled_output1 = self.r_dropout(pooled_output1)
                logits1 = self.classifier(pooled_output1)

                ce_loss = 0.5 * (loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) +
                                 loss_fct(logits1.view(-1, self.num_labels), labels.view(-1)))
                kl_loss = compute_kl_loss(logits, logits1)
                loss = ce_loss + self.rdrop_coef * kl_loss
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs


class NeZhaSequenceClassification_P(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.bert = NeZhaModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None
    ):
        attention_mask = torch.ne(input_ids, 0)
        encoder_out, pooled_output, all_hidden_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = self.classifier(pooled_output)
        outputs = (logits,) + (pooled_output,)

        return outputs

