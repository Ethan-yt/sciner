#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-08-19 19:25
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : span_ner.py
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.training.metrics import FBetaMeasure
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, Pruner
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator, Activation

from metrics.ner_f1 import NERF1Metric
from model.embeddings import get_embeddings
from model.span_extractor import PoolingSpanExtractor


class SCIIE(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward: ``FeedForward``
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width: ``int``
        The maximum width of candidate spans.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 embedding_dim: int,
                 feature_size: int,
                 max_span_width: int,
                 spans_per_word: float,
                 lexical_dropout: float = 0.2,
                 mlp_dropout: float = 0.4,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None, embedder_type=None) -> None:
        super(SCIIE, self).__init__(vocab, regularizer)
        class_num = self.vocab.get_vocab_size('ner_span_labels') - 1
        word_embeddings = get_embeddings(embedder_type, self.vocab, embedding_dim, True)
        embedding_dim = word_embeddings.get_output_dim()
        self._text_field_embedder = word_embeddings

        context_layer = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(embedding_dim, feature_size, batch_first=True, bidirectional=True))
        self._context_layer = context_layer

        endpoint_span_extractor_input_dim = context_layer.get_output_dim()
        attentive_span_extractor_input_dim = word_embeddings.get_output_dim()

        self._endpoint_span_extractor = EndpointSpanExtractor(endpoint_span_extractor_input_dim,
                                                              combination="x,y",
                                                              num_width_embeddings=max_span_width,
                                                              span_width_embedding_dim=feature_size,
                                                              bucket_widths=False)
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=attentive_span_extractor_input_dim)

        # self._span_extractor = PoolingSpanExtractor(embedding_dim,
        #                                             num_width_embeddings=max_span_width,
        #                                             span_width_embedding_dim=feature_size,
        #                                             bucket_widths=False)

        entity_feedforward = FeedForward(self._endpoint_span_extractor.get_output_dim() +
                                         self._attentive_span_extractor.get_output_dim(), 2, 150,
                                         F.relu, mlp_dropout)
        # entity_feedforward = FeedForward(self._span_extractor.get_output_dim(), 2, 150,
        #                                  F.relu, mlp_dropout)

        feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(entity_feedforward),
            TimeDistributed(torch.nn.Linear(entity_feedforward.get_output_dim(), 1)))
        self._mention_pruner = Pruner(feedforward_scorer)

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
        # initializer(self)

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                spans: torch.IntTensor,
                ner_span_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        document_length = text_embeddings.size(1)
        num_spans = spans.size(1)

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text).float()

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, document_length, encoding_dim)
        contextualized_embeddings = self._context_layer(text_embeddings, text_mask)

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        # Shape: (batch_size, num_spans, emebedding_size)
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
        # span_embeddings = self._span_extractor(text_embeddings, spans, span_indices_mask=span_mask)

        # Prune based on mention scores.
        num_spans_to_keep = int(math.floor(self._spans_per_word * document_length))
        num_spans_to_keep = min(num_spans_to_keep, span_embeddings.shape[1])

        # Shape:    (batch_size, num_spans_to_keep, emebedding_size + 2 * encoding_dim + feature_size)
        #           (batch_size, num_spans_to_keep)
        #           (batch_size, num_spans_to_keep)
        #           (batch_size, num_spans_to_keep, 1)
        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores) = self._mention_pruner(span_embeddings,
                                                                           span_mask,
                                                                           num_spans_to_keep)
        # (batch_size, num_spans_to_keep, 1)
        top_span_mask = top_span_mask.unsqueeze(-1)
        # Shape: (batch_size * num_spans_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select spans for each element in the batch.
        # This reformats the indices to take into account their
        # index into the batch. We precompute this here to make
        # the multiple calls to util.batched_index_select below more efficient.
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans,
                                              top_span_indices,
                                              flat_top_span_indices)

        # Shape: (batch_size, num_spans_to_keep, class_num + 1)
        ne_scores = self._compute_named_entity_scores(top_span_embeddings)

        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_named_entities = ne_scores.max(2)

        output_dict = {"top_spans": top_spans,
                       "predicted_named_entities": predicted_named_entities}
        if ner_span_labels is not None:
            # Find the gold labels for the spans which we kept.
            # Shape: (batch_size, num_spans_to_keep, 1)
            pruned_gold_labels = util.batched_index_select(ner_span_labels.unsqueeze(-1),
                                                           top_span_indices,
                                                           flat_top_span_indices).squeeze(-1)
            _, _, class_num = ne_scores.shape
            negative_log_likelihood = F.cross_entropy(ne_scores.reshape(-1, class_num), pruned_gold_labels.reshape(-1))

            pruner_loss = F.binary_cross_entropy_with_logits(top_span_mention_scores.reshape(-1),
                                                             (pruned_gold_labels.reshape(-1) != 0).float())
            loss = negative_log_likelihood + pruner_loss
            output_dict["loss"] = loss
            output_dict["pruner_loss"] = pruner_loss
            batch_size, _ = ner_span_labels.shape
            all_scores = ne_scores.new_zeros([batch_size * num_spans, class_num])
            all_scores[:, 0] = 1
            all_scores[flat_top_span_indices] = ne_scores.reshape(-1, class_num)
            all_scores = all_scores.reshape([batch_size, num_spans, class_num])
            self._metric1(all_scores, ner_span_labels)
            self._metric2(all_scores, ner_span_labels)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False, prefix=""):
        metric = self._metric1.get_metric(reset)
        metric2 = self._metric2.get_metric(reset)
        metric.update({k + '2': v for k, v in metric2.items()})
        return metric

    def _compute_named_entity_scores(self, span_embeddings: torch.FloatTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        span_embeddings: ``torch.FloatTensor``, required.
            Embedding representations of spans. Has shape
            (batch_size, num_spans_to_keep, encoding_dim)
        """
        # Shape: (batch_size, num_spans_to_keep, class_num)
        scores = self._entity_scorer(span_embeddings)
        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [scores.size(0), scores.size(1), 1]
        dummy_scores = scores.new_full(shape, 0)
        ne_scores = torch.cat([dummy_scores, scores], -1)
        return ne_scores
