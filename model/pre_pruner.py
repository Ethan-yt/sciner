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


class PrePruner(Model):

    def __init__(self, vocab,
                 feature_size: int,
                 max_span_width: int,
                 mlp_dropout: float = 0.4,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(PrePruner, self).__init__(vocab, regularizer)
        self.embedder = get_embeddings('bert', self.vocab, 300, bert_trainable=True)
        self.ffn = FeedForward(300, 2, 300, F.relu, 0.5)
        embedding_dim = self.embedder.get_output_dim()

        context_layer = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(embedding_dim, 300, batch_first=True, bidirectional=True))
        self._context_layer = context_layer
        endpoint_span_extractor_input_dim = context_layer.get_output_dim()
        attentive_span_extractor_input_dim = embedding_dim
        self._endpoint_span_extractor = EndpointSpanExtractor(endpoint_span_extractor_input_dim,
                                                              combination="x,y",
                                                              num_width_embeddings=max_span_width,
                                                              span_width_embedding_dim=feature_size,
                                                              bucket_widths=False)
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=attentive_span_extractor_input_dim)

        entity_feedforward = FeedForward(self._endpoint_span_extractor.get_output_dim() +
                                         self._attentive_span_extractor.get_output_dim(), 2, 150,
                                         F.relu, mlp_dropout)

        self.feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(entity_feedforward),
            TimeDistributed(torch.nn.Linear(entity_feedforward.get_output_dim(), 1)),
        )
        self._lexical_dropout = torch.nn.Dropout(p=0.2)

        self.loss = torch.nn.BCELoss()
        self._metric_f1 = FBetaMeasure()
        self._metric_pos_num = Average()
        self._metric_span_num = Average()

    def forward(self, text: Dict[str, torch.LongTensor],
                spans: torch.IntTensor,
                ner_span_labels: torch.IntTensor = None,
                threshold: float = 0.5,
                **kwargs):
        text_embeddings = self._lexical_dropout(self.embedder(text))

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text).float()

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        spans = F.relu(spans.float()).long()
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, emebedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)

        span_scores = self.feedforward_scorer(span_embeddings)

        span_scores = span_scores.squeeze(-1)
        span_scores += span_mask.log()
        span_scores = span_scores.sigmoid()

        labels = (ner_span_labels != 0).float()
        span_scores = span_scores.reshape(-1)
        labels = labels.reshape(-1)
        loss = self.loss(span_scores, labels)

        predict_true = span_scores > threshold
        predict_true = predict_true.unsqueeze(-1)
        predict_false = 1 - predict_true
        predict = torch.cat([predict_false, predict_true], -1)
        self._metric_f1(predict, labels, mask=span_mask.reshape(-1))
        self._metric_pos_num(predict_true.sum())
        self._metric_span_num(predict_true.shape[0])

        output_dict = {"loss": loss,
                       "predict_true": predict_true}

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False):
        metric = self._metric_f1.get_metric(reset)
        metric['f1'] = metric['fscore'][1]

        pos_num = self._metric_pos_num.get_metric(reset)
        metric.update({'pos_num': pos_num})
        span_num = self._metric_span_num.get_metric(reset)
        metric.update({'span_num': span_num})

        return metric
