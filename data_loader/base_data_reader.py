#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-10-16 16:49
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : base_data_reader
import os
from typing import Iterable

from allennlp.data import DatasetReader, Vocabulary, Instance
from allennlp.data.iterators import BasicIterator


class BaseDataReader(DatasetReader):
    def __init__(self, data_dir, batch_size: int, shuffle=False, small_data=False, test=False, max_span_width=8,
                 output_type="bio", use_neg_sampling=False, train_name='train.json',
                 dev_name='dev.json', test_name='test.json'):
        super().__init__()

        self._max_span_width = max_span_width
        self._use_neg_sampling = use_neg_sampling
        self._output_type = output_type

        self.train_dataset = self.read(os.path.join(data_dir, train_name))
        self.validation_dataset = self.read(os.path.join(data_dir, dev_name))
        self.vocab = Vocabulary.from_instances(self.train_dataset + self.validation_dataset)
        self.test_dataset = self.read(os.path.join(data_dir, test_name)) if test else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.small_data = small_data
        self.iterator = BasicIterator(batch_size=batch_size, cache_instances=True)
        self.iterator.index_with(self.vocab)

    def get_iterator_and_num_batches(self, data_type):
        dataset_type_map = {
            'train': self.train_dataset,
            'dev': self.validation_dataset,
            'test': self.test_dataset
        }
        dataset = dataset_type_map[data_type]
        if self.small_data and data_type == 'train':
            dataset = dataset[:int(len(dataset) / 10)]
        shuffle = self.shuffle if data_type == 'train' else False
        return self.iterator(dataset, shuffle=shuffle, num_epochs=1), self.iterator.get_num_batches(dataset)

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
        You are strongly encouraged to use a generator, so that users can
        read a dataset in a lazy way, if they so choose.
        """
        raise NotImplementedError

    def text_to_instance(self, *inputs) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        ``Instance``.  The primary intended use for this is with a
        :class:`~allennlp.service.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`_read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        ``DatasetReader`` to process new text lets us accomplish this, as we can just call
        ``DatasetReader.text_to_instance`` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The ``Predictor`` will
        have to make some assumptions about the kind of ``DatasetReader`` that it's using, in order
        to pass it the right information.
        """
        raise NotImplementedError
