#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-09-05 17:14
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : bert_embedder.py
import torch

from allennlp.modules import TokenEmbedder
from allennlp.nn.util import get_text_field_mask
from pytorch_pretrained_bert import BertModel


class BertEmbedder(TokenEmbedder):
    def __init__(self, trainable=True):
        super(BertEmbedder, self).__init__()
        self.trainable = trainable
        self.bert_model = BertModel.from_pretrained('embeddings/scibert_scivocab_uncased/weights.tar.gz')

    def get_output_dim(self) -> int:
        return self.bert_model.config.hidden_size

    def forward(self, inputs) -> torch.Tensor:
        mask = get_text_field_mask({'tokens': inputs})
        bert_embeddings, _ = self.bert_model(input_ids=inputs,
                                             attention_mask=mask,
                                             output_all_encoded_layers=False)
        if not self.trainable:
            bert_embeddings.detach_()
        return bert_embeddings
