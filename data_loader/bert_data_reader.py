#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-08-16 16:00
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : bert_data_reader.py
import json
from typing import Dict, List, Iterable, Tuple, Any

from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, SpanField, ListField, ArrayField, \
    LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from data_loader.base_data_reader import BaseDataReader
from utils.util import span2bio
import scipy.sparse as sp
import numpy as np


def _convert_tags_to_wordpiece_tags(tags: List[str], offsets: List[int]) -> List[str]:
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.
    This is only used if you pass a `bert_model_name` to the dataset reader below.
    Parameters
    ----------
    tags : `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces
    offsets : `List[int]`
        The wordpiece offsets.
    Returns
    -------
    The new BIO tags.
    """
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        is_o = tag == "O"
        is_start = True
        while j < offset:
            if is_o:
                new_tags.append("O")

            elif tag.startswith("I"):
                new_tags.append(tag)

            elif is_start and tag.startswith("B"):
                new_tags.append(tag)
                is_start = False

            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                new_tags.append("I-" + label)
            j += 1

    # Add O tags for cls and sep tokens.
    return ["O"] + new_tags + ["O"]


def _convert_verb_indices_to_wordpiece_indices(verb_indices: List[int],
                                               offsets: List[int]):  # pylint: disable=invalid-name
    """
    Converts binary verb indicators to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.
    This is only used if you pass a `bert_model_name` to the dataset reader below.
    Parameters
    ----------
    verb_indices : `List[int]`
        The binary verb indicators, 0 for not a verb, 1 for verb.
    offsets : `List[int]`
        The wordpiece offsets.
    Returns
    -------
    The new verb indices.
    """
    j = 0
    new_verb_indices = []
    for i, offset in enumerate(offsets):
        indicator = verb_indices[i]
        while j < offset:
            new_verb_indices.append(indicator)
            j += 1

    # Add 0 indicators for cls and sep tokens.
    return [0] + new_verb_indices + [0]


def _convert_span_to_wordpiece_span(ner_spans, start_offsets, end_offsets):
    result = []
    for start, end, tag in ner_spans:
        start = start_offsets[start]
        end = end_offsets[end]
        result.append((start, end, tag))
    return result


class BertSciercDataReader(BaseDataReader):
    """
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    Returns
    -------
    A ``Dataset`` of ``Instances``.
    """

    def __init__(self, data_dir, batch_size: int, shuffle=False, small_data=False, test=False, max_span_width=8,
                 output_type="bio", use_neg_sampling=False, train_name='train.json', dev_name='dev.json',
                 test_name='test.json') -> None:
        self._token_indexers = {"bert": SingleIdTokenIndexer()}
        self.bert_tokenizer = BertTokenizer.from_pretrained('embeddings/scibert_scivocab_uncased/vocab.txt')
        self.lowercase_input = True
        # self.cache_data('./bert.' + self._output_type)

        super().__init__(data_dir, batch_size, shuffle, small_data, test, max_span_width, output_type, use_neg_sampling,
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

        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets

    def _read(self, file_path: str) -> Iterable[Instance]:
        data_type = file_path.split('/')[-1].split('.')[0]

        prepruned_spans = None
        # if self.use_neg_sampling:
        #     if data_type != 'train':
        #         with open(f'data/scierc/pruned_{data_type}.pkl', 'rb') as file:
        #             prepruned_spans = pickle.load(file)

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
                instance = self.text_to_instance(example['sentences'], tokens, ner_spans, data_type == 'train',
                                                 prepruned_spans[i] if prepruned_spans else None)

                instances.append(instance)
                yield instance

    def text_to_instance(self,
                         sentence_list: List[List],
                         tokens: List[Token],
                         ner_spans: List[Tuple],
                         is_training_set,
                         prepruned_spans=None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input([t.text for t in tokens])
        metadata_dict: Dict[str, Any] = {"offsets": start_offsets,
                                         "words": [x.text for x in tokens],
                                         "sent_len": [len(sent) for sent in sentence_list]}
        text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                               token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {'text': text_field}

        if self._output_type == 'bio':
            self.build_bio(fields, ner_spans, tokens, offsets, text_field, metadata_dict)
        if self._output_type == 'span':
            self.build_spans(fields, ner_spans, wordpieces, start_offsets, offsets, text_field, sentence_list,
                             is_training_set)
            # if self.use_prepruned_spans:
            #     if prepruned_spans:
            #         ner_spans = []
            #         ner_span_labels = []
            #         pruned_spans = prepruned_spans['pruned_spans']
            #         pruned_labels = prepruned_spans['pruned_labels']
            #         for span, label in zip(pruned_spans, pruned_labels):
            #             ner_span_labels.append(label)
            #             ner_spans.append(SpanField(span[0], span[1], text_field))
            #         span_field = ListField(ner_spans)
            #         ner_span_labels_field = SequenceLabelField(ner_span_labels, span_field, 'ner_span_labels')
            #         fields["pruned_spans"] = span_field
            #         fields["pruned_labels"] = ner_span_labels_field
            #         fields["pruned_mask"] = ArrayField(np.array(prepruned_spans['pruned_mask']), padding_value=0)

        fields["metadata"] = MetadataField(metadata_dict)

        return Instance(fields)

    def build_bio(self, fields, ner_spans, tokens, offsets, text_field, metadata_dict):
        ner_tags = span2bio(ner_spans, len(tokens))
        wordpiece_tags = _convert_tags_to_wordpiece_tags(ner_tags, offsets)
        fields['ner_bio'] = SequenceLabelField(wordpiece_tags, text_field, label_namespace='ner_bio_labels')
        metadata_dict["gold_ner_bio"] = ner_tags

    def build_spans(self, fields, ner_spans, wordpieces, start_offsets, offsets, text_field, sentence_list,
                    is_training_set):
        new_spans = _convert_span_to_wordpiece_span(ner_spans, start_offsets, offsets)
        ner_spans_dict = {}
        for start, end, tag in new_spans:
            ner_spans_dict[(start, end)] = tag
        spans = []
        ner_span_labels = []
        pos_spans = []
        pos_labels = []
        pruned_mask = []
        sentence_offset = 0
        wordpiece_offset = 1
        for sentence in sentence_list:
            wordpiece_sentence = wordpieces[wordpiece_offset:
                                            offsets[sentence_offset + len(sentence) - 1] + 1]
            for start, end in enumerate_spans(wordpiece_sentence,
                                              offset=wordpiece_offset,
                                              max_span_width=self._max_span_width):
                if (start, end) in ner_spans_dict:
                    ner_span_labels.append(ner_spans_dict[(start, end)])
                    spans.append(SpanField(start, end, text_field))
                    pos_labels.append(ner_spans_dict[(start, end)])
                    pos_spans.append(SpanField(start, end, text_field))
                    pruned_mask.append(1)
                else:
                    ner_span_labels.append("O")
                    spans.append(SpanField(start, end, text_field))
                    pruned_mask.append(0)

            sentence_offset += len(sentence)
            wordpiece_offset += len(wordpiece_sentence)

        span_field = ListField(spans)
        ner_span_labels_field = SequenceLabelField(ner_span_labels, span_field, 'ner_span_labels')
        fields["spans"] = span_field
        fields["ner_span_labels"] = ner_span_labels_field
        if self._use_neg_sampling:
            if is_training_set:
                if pos_spans:
                    pos_spans_field = ListField(pos_spans)
                    pos_labels_field = ListField(
                        [LabelField(label, label_namespace='ner_span_labels') for label in pos_labels])
                else:
                    pos_spans_field = ListField([SpanField(0, 0, text_field)]).empty_field()
                    pos_labels_field = ListField([LabelField('O', label_namespace='ner_span_labels')]).empty_field()

                fields["pos_spans"] = pos_spans_field
                fields["pos_labels"] = pos_labels_field
                fields["pos_mask"] = ArrayField(np.array(pruned_mask), padding_value=0)
            else:
                fields["pos_spans"] = span_field
                fields["pos_labels"] = ner_span_labels_field
                fields["pos_mask"] = ArrayField(np.ones(len(spans)), padding_value=0)
