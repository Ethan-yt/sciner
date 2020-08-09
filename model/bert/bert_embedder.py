#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-09-05 17:14
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : bert_embedder.py
import torch

from allennlp.modules import TokenEmbedder
from allennlp.nn.util import get_text_field_mask
from transformers import AutoModel


class BertEmbedder(TokenEmbedder):
    def __init__(self, bert_type="'", trainable=True):
        super(BertEmbedder, self).__init__()
        self.trainable = trainable
        path = {
            "scibert": "embeddings/scibert_scivocab_uncased",
            "biobert": "embeddings/biobert_large",
            "bert": "embeddings/bert"
        }[bert_type]
        from_tf = bert_type == 'biobert'
        self.bert_model = AutoModel.from_pretrained(path, from_tf=from_tf)

    def get_output_dim(self) -> int:
        return self.bert_model.config.hidden_size

    def forward(self, inputs) -> torch.Tensor:
        mask = get_text_field_mask({'tokens': inputs})
        bert_embeddings = self.bert_model(inputs, attention_mask=mask)[0]
        if not self.trainable:
            bert_embeddings.detach_()
        return bert_embeddings
