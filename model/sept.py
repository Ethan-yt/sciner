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
                 spans_per_word: float,
                 lexical_dropout: float = 0.2,
                 mlp_dropout: float = 0.4,
                 sampling_rate: float = 1,
                 regularizer: Optional[RegularizerApplicator] = None, embedder_type=None) -> None:
        super(SEPT, self).__init__(vocab, regularizer)
        self.sampling_rate = sampling_rate
        class_num = self.vocab.get_vocab_size('ner_span_labels') - 1
        word_embeddings = get_embeddings(embedder_type, self.vocab, embedding_dim, True)
        embedding_dim = word_embeddings.get_output_dim()
        self._text_field_embedder = word_embeddings
        # self._span_extractor = PoolingSpanExtractor(embedding_dim,
        #                                             num_width_embeddings=max_span_width,
        #                                             span_width_embedding_dim=feature_size,
        #                                             bucket_widths=False)
        # self._span_extractor = PoolingSpanExtractor(embedding_dim)
        self._endpoint_span_extractor = EndpointSpanExtractor(embedding_dim,
                                                              combination="x,y",
                                                              num_width_embeddings=max_span_width,
                                                              span_width_embedding_dim=feature_size,
                                                              bucket_widths=False)
        # self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=embedding_dim)
        entity_feedforward = FeedForward(self._endpoint_span_extractor.get_output_dim(), 3, feature_size,
                                         F.relu, mlp_dropout)

        # entity_feedforward = FeedForward(self._span_extractor.get_output_dim(), 3, feature_size,
        #                                  F.relu, mlp_dropout)

        self._entity_scorer = torch.nn.Sequential(
            TimeDistributed(entity_feedforward),
            TimeDistributed(torch.nn.Linear(entity_feedforward.get_output_dim(), class_num)))

        self._max_span_width = max_span_width
        self._spans_per_word = spans_per_word
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        self._metric1 = FBetaMeasure()
        self._metric2 = NERF1Metric()

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                spans: torch.IntTensor,
                ner_span_labels: torch.IntTensor,
                pos_spans=None,
                pos_labels=None,
                pos_mask=None,
                metadata: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text).float()

        if self.training:
            pos_mask = pos_mask.byte()
            span_mask = spans[:, :, 0] >= 0
            for i in range(pos_mask.shape[0]):
                sampling_mask = pos_mask.new_zeros(pos_mask.shape[1])
                sampling_mask[:span_mask[i].sum()] = torch.rand(span_mask[i].sum()) < self.sampling_rate
                pos_mask[i] = pos_mask[i] | sampling_mask
            lengths = pos_mask.sum(1)
            pos_spans = spans.new_full((pos_mask.shape[0], lengths.max(), 2), -1)
            pos_labels = spans.new_zeros(pos_mask.shape[0], lengths.max())
            flatten_spans = spans[pos_mask]
            flatten_labels = ner_span_labels[pos_mask]
            start = 0
            for i, l in enumerate(lengths):
                pos_spans[i, :l] = flatten_spans[start:start + l]
                pos_labels[i, :l] = flatten_labels[start:start + l]
                start += l.item()

        span_mask = pos_spans[:, :, 0] >= 0
        pos_spans = F.relu(pos_spans.float()).long()
        # span_embeddings = self._span_extractor(text_embeddings, pos_spans, span_indices_mask=span_mask)

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(text_embeddings, pos_spans)
        # Shape: (batch_size, num_spans, emebedding_size)
        # attended_span_embeddings = self._attentive_span_extractor(text_embeddings, pos_spans)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = endpoint_span_embeddings
        # span_embeddings = self._span_extractor(text_embeddings, spans, span_indices_mask=span

        # Shape: (batch_size, num_spans_to_keep, class_num + 1)
        ne_scores = self._compute_named_entity_scores(span_embeddings)

        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_named_entities = ne_scores.max(2)

        output_dict = {
            "predicted_named_entities": predicted_named_entities}
        if ner_span_labels is not None:
            # Find the gold labels for the spans which we kept.
            # Shape: (batch_size, num_spans_to_keep, 1)
            _, _, class_num = ne_scores.shape
            ne_scores = ne_scores.reshape(-1, class_num)
            sliced_ne_scores = ne_scores[span_mask.reshape(-1)]
            pos_labels = pos_labels.reshape(-1)
            sliced_pruned_labels = pos_labels[span_mask.reshape(-1)]
            negative_log_likelihood = F.cross_entropy(sliced_ne_scores, sliced_pruned_labels)
            output_dict["loss"] = negative_log_likelihood

            spans = spans.reshape(-1, 2)
            all_scores = ne_scores.new_zeros([spans.shape[0], class_num])
            all_scores[:, 0] = 1
            all_scores[pos_mask.byte().reshape(-1)] = sliced_ne_scores

            self._metric1(all_scores, ner_span_labels.reshape(-1))
            self._metric2(all_scores, ner_span_labels.reshape(-1))

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False, prefix=""):
        metric = self._metric1.get_metric(reset)
        metric2 = self._metric2.get_metric(reset)
        metric.update({k + '2': v for k, v in metric2.items()})
        return metric

    def _compute_named_entity_scores(self, span_embeddings: torch.FloatTensor) -> torch.Tensor:
        scores = self._entity_scorer(span_embeddings)
        shape = [scores.size(0), scores.size(1), 1]
        dummy_scores = scores.new_full(shape, 0)
        ne_scores = torch.cat([dummy_scores, scores], -1)
        return ne_scores
