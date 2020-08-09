#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2020-04-28 10:35
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : convert_iob_to_json.py
import json
import os

from utils import span2bio


def convert(origin_path, save_path):
    result = []
    with open(origin_path) as file:
        for row in file.read().split("\n"):
            if not row:
                continue
            example = json.loads(row)
            offset = 0
            result.append('-DOCSTART-')
            result.append('')
            for tokens, spans in zip(example["sentences"], example["ner"]):
                spans = [[span[0] - offset, span[1] - offset, span[-1]] for span in spans]
                labels = span2bio(spans, len(tokens))
                offset += len(tokens)
                result.extend([f"{token} O O {label}" for token, label in zip(tokens, labels)])
                result.append('')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wt') as file:
        print(f"length:{len(result)}")
        file.write("\n".join(result))


if __name__ == '__main__':
    dataset_name = "scierc"
    for data_type in ['train', 'dev', 'test']:
        convert(f"data/{dataset_name}/{data_type}.json", f"data/{dataset_name}_bio/{data_type}.txt")
