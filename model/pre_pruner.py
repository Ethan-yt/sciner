#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-09-18 19:37
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : dep_pruner.py
import torch
from typing import Optional, Dict

from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from allennlp.nn import RegularizerApplicator
import torch.nn.functional as F
from allennlp.training.metrics import FBetaMeasure, Average

from model.embeddings import get_embeddings
import torch
from overrides import overrides

from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util

from model.span_extractor import PoolingSpanExtractor


class PrePruner(Model):

    def __init__(self, vocab,
                 feature_size: int,
                 max_span_width: int,
                 keep_rate: int,
                 mlp_dropout: float = 0.4,
                 embedder_type=None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(PrePruner, self).__init__(vocab, regularizer)
        self.keep_rate = keep_rate
        self.embedder = get_embeddings(embedder_type, self.vocab)
        self.ffn = FeedForward(300, 2, 300, F.relu, 0.5)
        embedding_dim = self.embedder.get_output_dim()

        self._span_extractor = PoolingSpanExtractor(embedding_dim,
                                                    num_width_embeddings=max_span_width,
                                                    span_width_embedding_dim=feature_size,
                                                    bucket_widths=False)
        entity_feedforward = FeedForward(self._span_extractor.get_output_dim(), 2, 150,
                                         F.relu, mlp_dropout)

        self.feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(entity_feedforward),
            TimeDistributed(torch.nn.Linear(entity_feedforward.get_output_dim(), 1)),
        )
        self._lexical_dropout = torch.nn.Dropout(p=0.1)

        self.loss = torch.nn.BCELoss()
        self._metric_f1 = FBetaMeasure()

    def forward(self, text: Dict[str, torch.LongTensor],
                spans: torch.IntTensor,
                labels: torch.IntTensor = None,
                **kwargs):
        text_embeddings = self._lexical_dropout(self.embedder(text))

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        spans = F.relu(spans.float()).long()

        span_embeddings = self._span_extractor(text_embeddings, spans, span_indices_mask=span_mask)

        span_scores = self.feedforward_scorer(span_embeddings)

        span_scores = span_scores.squeeze(-1)
        span_scores += span_mask.log()
        span_scores = span_scores.sigmoid()
        topk_idx = torch.topk(span_scores, int(self.keep_rate * spans.shape[1]))[-1]
        predict_true = span_scores.new_zeros(span_scores.shape).scatter_(1, topk_idx, 1).bool()
        is_entity = (labels != 0).float()
        span_scores = span_scores.reshape(-1)
        is_entity = is_entity.reshape(-1)
        loss = self.loss(span_scores, is_entity)

        predict_true_flatten = predict_true.reshape(-1)
        predict_true_flatten = predict_true_flatten.unsqueeze(-1)
        predict_false_flatten = ~predict_true_flatten
        predict = torch.cat([predict_false_flatten, predict_true_flatten], -1)
        self._metric_f1(predict, is_entity, mask=span_mask.reshape(-1))

        predict_true |= labels.bool()
        output_dict = {"loss": loss,
                       "predict_true": predict_true}
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False):
        metric = self._metric_f1.get_metric(reset)
        metric['precision'] = metric['precision'][1]
        metric['recall'] = metric['recall'][1]
        metric['fscore'] = metric['fscore'][1]

        return metric
