import json
import torch
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def sequence_mask(lens, max_len):
    mask = torch.arange(max_len, device=lens.device).expand(len(lens), max_len) < lens.unsqueeze(1)
    return mask


def id_to_text(ids, vocab, mask, namespace='tokens'):
    return [vocab.get_token_from_index(_id, namespace) for _id, m in zip(ids, mask) if m]


def span2bio(spans, seq_len):
    ret = ['O'] * seq_len
    for span in spans:
        ret[span[0]] = "B-" + span[2]
        for i in range(span[0] + 1, span[1] + 1):
            ret[i] = "I-" + span[2]
    return ret
