#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-08-16 16:00
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : bert_data_reader.py
import json
from typing import Dict, List, Iterable, Tuple, Any

from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from transformers import AutoTokenizer

from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, SpanField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from data_loader.base_data_reader import BaseDataReader


def _convert_span_to_wordpiece_span(ner_spans, start_offsets, end_offsets):
    result = []
    for start, end, tag in ner_spans:
        start = start_offsets[start]
        end = end_offsets[end]
        result.append((start, end, tag))
    return result


class BaseSpanDataReader(BaseDataReader):
    def __init__(self, data_dir, batch_size, shuffle, small_data, max_span_width, train_name, dev_name, test_name,
                 embedder_type='scibert') -> None:
        self._max_span_width = max_span_width
        self._token_indexers = {"bert": SingleIdTokenIndexer()}
        path = {
            "scibert": "embeddings/scibert_scivocab_uncased",
            "biobert": "embeddings/biobert_large",
            "bert": "embeddings/bert"
        }[embedder_type]
        self.bert_tokenizer = AutoTokenizer.from_pretrained(path)
        self.lowercase_input = True
        super().__init__(data_dir, batch_size, shuffle, small_data,
                         train_name, dev_name, test_name)

    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the begining and end of the sentence.
        A slight oddity with this function is that it also returns the wordpiece offsets
        corresponding to the _start_ of words as well as the end.
        We need both of these offsets (or at least, it's easiest to use both), because we need
        to convert the labels to tags using the end_offsets. However, when we are decoding a
        BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
        because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
        wordpieces (this happens in the case that a word is split into multiple word pieces,
        and then we take the last tag of the word, which might correspond to, e.g, I-V, which
        would not be allowed as it is not preceeded by a B tag).
        For example:
        `annotate` will be bert tokenized as ["anno", "##tate"].
        If this is tagged as [B-V, I-V] as it should be, we need to select the
        _first_ wordpiece label to be the label for the token, because otherwise
        we may end up with invalid tag sequences (we cannot start a new tag with an I).
        Returns
        -------
        wordpieces : List[str]
            The BERT wordpieces from the words in the sentence.
        end_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each word being chosen.
        start_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)
        if len(word_piece_tokens) > 510:
            word_piece_tokens = word_piece_tokens[:510]
        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets

    def text_to_instance(self,
                         sentence_list: List[List],
                         tokens: List[Token],
                         ner_spans: List[Tuple]) -> Instance:
        wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input([t.text for t in tokens])
        metadata_dict: Dict[str, Any] = {
            "sentences": sentence_list,
            "offsets": start_offsets,
            "words": [x.text for x in tokens],
            "sent_len": [len(sent) for sent in sentence_list]}
        text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                               token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {'text': text_field}

        self.build_spans(fields, ner_spans, wordpieces, start_offsets, offsets, text_field, sentence_list)

        fields["metadata"] = MetadataField(metadata_dict)

        return Instance(fields)

    def build_spans(self, fields, ner_spans, wordpieces, start_offsets, offsets, text_field, sentence_list):
        new_spans = _convert_span_to_wordpiece_span(ner_spans, start_offsets, offsets)
        ner_spans_dict = {}
        for start, end, tag in new_spans:
            ner_spans_dict[(start, end)] = tag
        spans = []
        labels = []
        sentence_offset = 0
        wordpiece_offset = 1
        for sentence in sentence_list:
            wordpiece_sentence = wordpieces[wordpiece_offset:
                                            offsets[sentence_offset + len(sentence) - 1] + 1]
            for start, end in enumerate_spans(wordpiece_sentence,
                                              offset=wordpiece_offset,
                                              max_span_width=self._max_span_width):
                if (start, end) in ner_spans_dict:
                    labels.append(ner_spans_dict[(start, end)])
                    spans.append(SpanField(start, end, text_field))
                else:
                    labels.append("O")
                    spans.append(SpanField(start, end, text_field))

            sentence_offset += len(sentence)
            wordpiece_offset += len(wordpiece_sentence)

        span_field = ListField(spans)
        labels_field = SequenceLabelField(labels, span_field, 'labels')
        fields["spans"] = span_field
        fields["labels"] = labels_field


class SpanDataReader(BaseSpanDataReader):
    def __init__(self, data_dir, batch_size: int, shuffle=False, small_data=False, max_span_width=8,
                 train_name='train.json', dev_name='dev.json',
                 test_name='test.json', embedder_type='scibert') -> None:
        super().__init__(data_dir, batch_size, shuffle, small_data, max_span_width,
                         train_name, dev_name, test_name, embedder_type)

    def _read(self, file_path: str) -> Iterable[Instance]:
        instances = []
        with open(file_path) as file:
            for i, line in enumerate(file):
                example = json.loads(line)
                mapping_dict = {"-LRB-": '(', "-RRB-": ')'}
                tokens = [word for sentence in example['sentences'] for word in sentence]
                tokens = [mapping_dict[token] if token in mapping_dict else token for token in tokens]
                tokens = [Token(token) for token in tokens]
                if 'ner' in example:
                    ner_spans = [span for spans in example['ner'] for span in spans]
                else:
                    ner_spans = []
                instance = self.text_to_instance(example['sentences'], tokens, ner_spans)

                instances.append(instance)
                yield instance

        avg_spans = sum([len(i['spans']) for i in instances]) / len(instances)
        print('avg_span num:', avg_spans)


class PrunedSpanDataReader(BaseSpanDataReader):
    def __init__(self, data_dir, batch_size: int, shuffle=False, small_data=False, max_span_width=8,
                 train_name='train.json', dev_name='dev.json',
                 test_name='test.json', embedder_type='scibert') -> None:
        super().__init__(data_dir, batch_size, shuffle, small_data, max_span_width,
                         train_name, dev_name, test_name, embedder_type)

    def _read(self, file_path: str) -> Iterable[Instance]:
        instances = []
        with open(file_path) as file:
            for i, line in enumerate(file):
                example = json.loads(line)
                mapping_dict = {"-LRB-": '(', "-RRB-": ')'}
                tokens = [word for sentence in example['sentences'] for word in sentence]
                tokens = [mapping_dict[token] if token in mapping_dict else token for token in tokens]
                tokens = [Token(token) for token in tokens]

                instance = self.text_to_instance(example['sentences'], tokens, example['pruned_spans'])

                instances.append(instance)
                yield instance

    def text_to_instance(self,
                         sentence_list: List[List],
                         tokens: List[Token],
                         pruned_spans: List[Tuple]) -> Instance:
        wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input([t.text for t in tokens])
        metadata_dict: Dict[str, Any] = {
            "sentences": sentence_list,
            "offsets": start_offsets,
            "words": [x.text for x in tokens],
            "sent_len": [len(sent) for sent in sentence_list]}
        text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                               token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {'text': text_field}

        spans = []
        labels = []
        for start, end, label in pruned_spans:
            spans.append(SpanField(start, end, text_field))
            labels.append(label)
        span_field = ListField(spans)
        labels_field = SequenceLabelField(labels, span_field, 'labels')
        fields["spans"] = span_field
        fields["labels"] = labels_field
        fields["metadata"] = MetadataField(metadata_dict)

        return Instance(fields)
