#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2019-09-05 17:37
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : embeddings.py
import torch

from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder

from model.bert.bert_embedder import BertEmbedder
import torchtext.vocab

valid_embedders = ['random', 'glove', 'elmo', 'elmo_and_glove', 'bert', 'biobert', 'scibert']


def load_glove_weights(vocab):
    glove = torchtext.vocab.Vectors(name='embeddings/glove.6B/glove.6B.300d.txt')
    weights = torch.zeros(vocab.get_vocab_size(), 300)
    torch.nn.init.xavier_uniform_(weights)

    for token, idx in vocab.get_token_to_index_vocabulary().items():
        try:
            weights[idx] = glove.vectors[glove.stoi[token]]
        except KeyError:
            pass
    return weights


def get_embeddings(embedder_type, vocab, embedding_dim=300, bert_trainable=True):
    if embedder_type not in valid_embedders:
        raise Exception(f'Unknown embedder type {embedder_type}')
    vocab_size = vocab.get_vocab_size('tokens')
    token_embedders = {}
    if embedder_type == 'random':
        token_embedding = Embedding(vocab_size, embedding_dim, trainable=True)
        token_embedders['tokens'] = token_embedding
    if embedder_type in ['glove', 'elmo_and_glove']:
        weights = load_glove_weights(vocab)
        token_embedding = Embedding(vocab_size, embedding_dim, weight=weights,
                                    trainable=True)
        token_embedders['tokens'] = token_embedding
    if embedder_type in ['elmo', 'elmo_and_glove']:
        elmo_token_embedder = ElmoTokenEmbedder('embeddings/elmo_2x4096_512_2048cnn_2xhighway_options.json',
                                                'embeddings/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
                                                do_layer_norm=False, dropout=0.5)
        token_embedders['elmo'] = elmo_token_embedder
    if 'bert' in embedder_type:
        token_embedders['bert'] = BertEmbedder(bert_type=embedder_type, trainable=bert_trainable)

    word_embeddings = BasicTextFieldEmbedder(token_embedders)
    return word_embeddings
