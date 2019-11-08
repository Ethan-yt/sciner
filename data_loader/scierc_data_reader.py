#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-08-15 16:29
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : data_readers.py.py

import json
from typing import Dict, Iterator, Any
from allennlp.data import Token, Instance
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import TextField, SequenceLabelField, SpanField, ListField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer

from data_loader.base_data_reader import BaseDataReader
from utils.util import span2bio

"""
Read SciERC dataset
Output:
    text: {'tokens': (batch_size, max_words)}
    ner_bio: (batch_size, max_words)
    ner_span: (batch_size, max_span_num, 2)
    ner_tag: (batch_size, max_span_num)
"""


class SciercDataReader(BaseDataReader):
    def __init__(self, data_dir, batch_size: int, shuffle=False, small_data=False, test=False, max_span_width=8,
                 output_type="bio", use_neg_sampling=False, train_name='train.json', dev_name='dev.json',
                 test_name='test.json', use_elmo=False) -> None:
        self.use_elmo = use_elmo
        if use_elmo:
            token_indexers = {"tokens": SingleIdTokenIndexer(), "elmo": ELMoTokenCharactersIndexer()}
        else:
            token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.token_indexers = token_indexers
        # self.cache_data('./elmo' if self.use_elmo else 'glove' + '.' + self._output_type)

        super().__init__(data_dir, batch_size, shuffle, small_data, test, max_span_width, output_type, use_neg_sampling,
                         train_name, dev_name, test_name)

    def text_to_instance(self, example) -> Instance:
        mapping_dict = {"-LRB-": '(', "-RRB-": ')'}
        tokens = [word for sentence in example['sentences'] for word in sentence]
        tokens = [mapping_dict[token] if token in mapping_dict else token for token in tokens]
        tokens = [Token(token) for token in tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {"text": text_field}

        ner_spans = [span for spans in example['ner'] for span in spans]

        if self._output_type == 'bio':
            self.build_bio(fields, ner_spans, tokens, text_field)
        if self._output_type == 'span':
            self.build_spans(fields, ner_spans, example["sentences"], text_field)

        metadata_dict: Dict[str, Any] = {"words": [x.text for x in tokens]}
        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        instances = []
        with open(file_path) as file:
            for i, line in enumerate(file):
                example = json.loads(line)
                instance = self.text_to_instance(example)
                instances.append(instance)
                yield instance

    def build_spans(self, fields, ner_spans, sentences, text_field):
        ner_spans_dict = {}
        for start, end, tag in ner_spans:
            ner_spans_dict[(start, end)] = tag
        spans = []
        ner_span_labels = []
        sentence_offset = 0
        for sentence in sentences:
            for start, end in enumerate_spans(sentence,
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                if (start, end) in ner_spans_dict:
                    ner_span_labels.append(ner_spans_dict[(start, end)])
                    spans.append(SpanField(start, end, text_field))
                else:
                    ner_span_labels.append("O")
                    spans.append(SpanField(start, end, text_field))
            sentence_offset += len(sentence)

        span_field = ListField(spans)
        ner_span_labels_field = SequenceLabelField(ner_span_labels, span_field, 'ner_span_labels')
        fields["spans"] = span_field
        fields["ner_span_labels"] = ner_span_labels_field

    def build_bio(self, fields, ner_spans, tokens, text_field):
        ner_bio_tags = span2bio(ner_spans, len(tokens))
        ner_bio_field = SequenceLabelField(ner_bio_tags,
                                           sequence_field=text_field,
                                           label_namespace='ner_bio_labels')
        fields["ner_bio"] = ner_bio_field
