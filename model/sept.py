#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-08-19 19:25
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : sept.py
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from allennlp.training.metrics import FBetaMeasure
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from metrics.ner_f1 import NERF1Metric
from model.embeddings import get_embeddings
from model.span_extractor import PoolingSpanExtractor


class SEPT(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedding_dim: int,
                 feature_size: int,
                 max_span_width: int,
                 lexical_dropout: float = 0.2,
                 mlp_dropout: float = 0.4,
                 sampling_rate: float = 1,
                 regularizer: Optional[RegularizerApplicator] = None, embedder_type=None) -> None:
        super(SEPT, self).__init__(vocab, regularizer)
        self.sampling_rate = sampling_rate
        self.class_num = self.vocab.get_vocab_size('labels')
        word_embeddings = get_embeddings(embedder_type, self.vocab, embedding_dim, True)
        embedding_dim = word_embeddings.get_output_dim()
        self._text_field_embedder = word_embeddings
        self._span_extractor = PoolingSpanExtractor(embedding_dim)
        # self._endpoint_span_extractor = EndpointSpanExtractor(embedding_dim,
        #                                                       combination="x,y",
        #                                                       num_width_embeddings=max_span_width,
        #                                                       span_width_embedding_dim=feature_size,
        #                                                       bucket_widths=False)
        # self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=embedding_dim)

        # entity_feedforward = FeedForward(self._span_extractor.get_output_dim()
        #                                  + self._endpoint_span_extractor.get_output_dim()
        #                                  + self._attentive_span_extractor.get_output_dim(), 3, feature_size,
        #                                  F.relu, mlp_dropout)
        entity_feedforward = FeedForward(self._span_extractor.get_output_dim(), 3, feature_size,
                                         F.relu, mlp_dropout)

        self._entity_scorer = torch.nn.Sequential(
            TimeDistributed(entity_feedforward),
            TimeDistributed(torch.nn.Linear(entity_feedforward.get_output_dim(), self.class_num)))

        self._max_span_width = max_span_width
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        self._metric_all = FBetaMeasure()
        self._metric_avg = NERF1Metric()

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                spans: torch.IntTensor,
                labels: torch.IntTensor,
                metadata: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        # Shape: (batch_size, document_length)
        span_mask = spans[:, :, 0] >= 0
        spans = F.relu(spans.float()).long()
        span_embeddings = self._span_extractor(text_embeddings, spans, span_indices_mask=span_mask)
        # endpoint_span_embeddings = self._endpoint_span_extractor(text_embeddings, spans)
        # attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # span_embeddings = torch.cat([pooling_span_embeddings, endpoint_span_embeddings, attended_span_embeddings], -1)

        # Shape: (batch_size, num_spans, class_num)
        ne_scores = self._entity_scorer(span_embeddings)

        # Shape: (batch_size, num_spans)
        _, predicted_named_entities = ne_scores.max(2)

        output_dict = {
            "predicted_named_entities": predicted_named_entities}
        if labels is not None:
            ne_scores = ne_scores.reshape(-1, self.class_num)
            labels = labels.reshape(-1)
            span_mask = span_mask.reshape(-1)

            if self.sampling_rate > 0:
                neg_mask = labels == 0
                neg_sampling_mask = neg_mask & span_mask & (
                        torch.rand(labels.shape, device=labels.device) < self.sampling_rate)
                sampling_mask = neg_sampling_mask | ~neg_mask
                negative_log_likelihood = F.cross_entropy(ne_scores[sampling_mask], labels[sampling_mask])
            else:
                negative_log_likelihood = F.cross_entropy(ne_scores, labels)
            output_dict["loss"] = negative_log_likelihood

            self._metric_all(ne_scores, labels.reshape(-1), span_mask)
            self._metric_avg(ne_scores, labels.reshape(-1), span_mask)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False, prefix=""):
        metric = self._metric_all.get_metric(reset)
        metric2 = self._metric_avg.get_metric(reset)
        metric.update(metric2)
        return metric
