#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2020-04-28 10:35
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : convert_iob_to_json.py
import itertools
import json
import os

from allennlp.data.dataset_readers.dataset_utils import bio_tags_to_spans
from transformers import AutoTokenizer


def _is_divider(line: str) -> bool:
    return line.strip() == ''


def _is_doc_start(line):
    return line.startswith('-DOCSTART-')


bert_tokenizer = AutoTokenizer.from_pretrained("embeddings/bert")


def convert(origin_path, save_path, max_len):
    result = []
    with open(origin_path) as file:
        for is_doc_start, doc in itertools.groupby(file, _is_doc_start):
            if not is_doc_start:
                sentences = []
                ner = []
                offset = 0
                document_word_piece = []
                for is_divider, lines in itertools.groupby(doc, _is_divider):
                    if not is_divider:
                        fields = [line.strip().split() for line in lines]
                        fields = [list(field) for field in zip(*fields)]
                        tokens, _, _, labels = fields
                        sentence_word_pieces = []
                        for token in tokens:
                            word_piece = bert_tokenizer.wordpiece_tokenizer.tokenize(token.lower())
                            sentence_word_pieces.extend(word_piece)
                        if len(document_word_piece) + len(sentence_word_pieces) + 2 > max_len:
                            # to prevent too long input for bert (max length 512)
                            result.append(json.dumps({"sentences": sentences, "ner": ner}))
                            offset = 0
                            sentences = []
                            ner = []
                            document_word_piece = []

                        spans = [[start + offset, end + offset, label] for label, (start, end) in
                                 bio_tags_to_spans(labels)]
                        sentences.append(tokens)
                        ner.append(spans)
                        offset += len(tokens)
                        document_word_piece.extend(sentence_word_pieces)

                if sentences:
                    result.append(json.dumps({"sentences": sentences, "ner": ner}))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wt') as file:
        print(f"length:{len(result)}")
        file.write("\n".join(result))


if __name__ == '__main__':
    dataset_name = 'JNLPBA'
    # dataset_name = 'NCBI-disease'
    # dataset_name = 'bc5cdr'
    for data_type in ['train', 'dev', 'test']:
        convert(f"data/{dataset_name}_bio/{data_type}.txt", f"data/{dataset_name}/{data_type}.json", 512)
