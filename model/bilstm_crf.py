#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-08-06 14:01
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : bilstm_crf.py
from typing import Dict, List, Any

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions, ConditionalRandomField
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.training.metrics import SpanBasedF1Measure
from overrides import overrides
from torch.nn import Linear

import allennlp.nn.util as util

from model.embeddings import get_embeddings

"""
flatten BiLSTM-CRF model
"""


class BiLSTMCRF(Model):

    def __init__(self, vocab: Vocabulary, embedding_dim=300, embedder_type=None, bert_trainable=True, **kwargs):
        super().__init__(vocab)
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        text_field_embedder = get_embeddings(embedder_type, self.vocab, embedding_dim, bert_trainable)
        embedding_dim = text_field_embedder.get_output_dim()

        encoder = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(embedding_dim, self.num_rnn_units, batch_first=True, bidirectional=True, dropout=self.dropout_rate))

        self.label_namespace = label_namespace = 'ner_bio_labels'
        self.num_tags = self.vocab.get_vocab_size(label_namespace)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(self.dropout_rate)

        output_dim = self.encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(output_dim,
                                                           self.num_tags))

        self.label_encoding = label_encoding = 'BIO'
        labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
        constraints = allowed_transitions(self.label_encoding, labels)

        self.include_start_end_transitions = True
        self.crf = ConditionalRandomField(
            self.num_tags, constraints,
            include_start_end_transitions=True
        )

        self._f1_metric = SpanBasedF1Measure(self.vocab,
                                             tag_namespace=label_namespace,
                                             label_encoding=label_encoding)
        self._verbose_metrics = False

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                ner_bio: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.text_field_embedder(text)
        mask = util.get_text_field_mask(text)
        tags = ner_bio
        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        lstm_feature = self.encoder(embedded_text_input, mask)

        if self.dropout:
            lstm_feature = self.dropout(lstm_feature)
        logits = self.tag_projection_layer(lstm_feature)

        best_paths = self.crf.viterbi_tags(logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if tags is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, tags, mask)

            output["loss"] = -log_likelihood
            batch_size = tags.shape[0]
            output['loss'] /= batch_size

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            self._f1_metric(class_probabilities, tags, mask.float())
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        f1_dict = self._f1_metric.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update(f1_dict)
        else:
            metrics_to_return.update({
                x: y for x, y in f1_dict.items() if
                "overall" in x})
        return metrics_to_return

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]
        ]

        return output_dict
