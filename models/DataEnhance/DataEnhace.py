# -*- coding: utf-8 -*-

from torch import nn
from transformers import BertForMaskedLM


class DataEnhance(nn.Module):
    def __init__(self, config):
        super(DataEnhance, self).__init__()
        self.config = config
        self.bert = BertForMaskedLM.from_pretrained(config.pretrained_model_path)

    def forward(self, inputIds, attentionMask, tokenTypeIds):
        output = self.bert(inputIds, attentionMask, tokenTypeIds)
        return output
