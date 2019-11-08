#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-10-11 15:52
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : span_extractor


import torch
from torch.nn.parameter import Parameter
from overrides import overrides

from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn import util

from allennlp.common.checks import ConfigurationError


class PoolingSpanExtractor(SpanExtractor):
    def __init__(self,
                 input_dim: int,
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(num_width_embeddings, span_width_embedding_dim)
        elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
            raise ConfigurationError("To use a span width embedding representation, you must"
                                     "specify both num_width_buckets and span_width_embedding_dim.")
        else:
            self._span_width_embedding = None

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = self._input_dim
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:
        # shape (batch_size, num_spans)
        # span_starts, span_ends = span_indices.split(1, dim=-1)
        # batch_size, max_seq_len, _ = sequence_tensor.shape
        # max_span_num = span_indices.shape[1]
        # range_vector = util.get_range_vector(max_seq_len, util.get_device_of(sequence_tensor)).repeat(
        #     (batch_size, max_span_num, 1))
        # att_mask = (span_ends >= range_vector) - (span_starts > range_vector)
        # att_mask = att_mask * span_mask.unsqueeze(-1)
        # res = self._attention(sequence_tensor.repeat((max_span_num,1,1)), att_mask)

        # combined_tensors = util.combine_tensors(self._combination, [start_embeddings, end_embeddings])

        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # shape (batch_size, sequence_length, 1)
        # global_attention_logits = self._global_attention(sequence_tensor)

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = util.get_range_vector(max_batch_span_width,
                                                       util.get_device_of(sequence_tensor)).view(1, 1, -1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        span_embeddings = span_embeddings * span_mask.unsqueeze(-1)
        span_embeddings = span_embeddings.max(2)[0]

        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            if self._bucket_widths:
                span_widths = util.bucket_values(span_ends - span_starts,
                                                 num_total_buckets=self._num_width_embeddings)
            else:
                span_widths = span_ends - span_starts
            span_widths = span_widths.squeeze(-1)
            span_width_embeddings = self._span_width_embedding(span_widths)
            combined_tensors = torch.cat([span_embeddings, span_width_embeddings], -1)
        else:
            combined_tensors = span_embeddings
        if span_indices_mask is not None:
            return combined_tensors * span_indices_mask.unsqueeze(-1).float()

        return combined_tensors
