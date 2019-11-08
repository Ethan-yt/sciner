import argparse
import collections
import os
os.environ["OMP_NUM_THREADS"] = "1"

from pytorch_pretrained_bert import BertTokenizer

import torch
from tqdm import tqdm
import data_loader as module_data
import model as module_arch
from parse_config import ConfigParser
from allennlp.nn import util as nn_util

from utils import util


def main(config, verbose=True):
    logger = config.get_logger('test')

    # setup data_loader instances
    config['data_loader']['args']['shuffle'] = False
    config['data_loader']['args']['test'] = True

    data_loader = getattr(module_data, config['data_loader']['type'])(
        **config['data_loader']['args'],
    )

    # build model architecture
    model = config.initialize('arch', module_arch, vocab=data_loader.vocab)
    if verbose:
        logger.info(model)
        logger.info('Loading checkpoint: {} ...'.format(config.resume))

    # get function handles of loss and metrics
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_iter, test_num_batches = data_loader.get_iterator_and_num_batches('test')

    total_loss = 0.0
    results_to_save = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_iter)):
            data = nn_util.move_to_device(data, 0)
            output = model(**data)
            loss = output['loss']
            total_loss += loss.item()

            if config.config.get('save_results'):
                if 'ner_bio' in data:
                    batch_size = data['ner_bio'].shape[0]
                    masks = nn_util.get_text_field_mask(data['text'])
                    for i in range(batch_size):
                        mask = masks[i].cpu().numpy()
                        sentence = data['text']['tokens'][i].cpu().numpy()
                        sentence = util.id_to_text(sentence, model.vocab, mask)
                        results_to_save.append(' '.join(sentence))
                        ground_truth = data['ner_bio'][i].cpu().numpy()
                        ground_truth = util.id_to_text(ground_truth, model.vocab, mask, namespace='ner_bio_labels')
                        results_to_save.append(' '.join(ground_truth))
                        predict = output['tags'][i]
                        predict = util.id_to_text(predict, model.vocab, mask, namespace='ner_bio_labels')
                        results_to_save.append(' '.join(predict))
                else:
                    batch_size = data['ner_span_labels'].shape[0]
                    masks = nn_util.get_text_field_mask(data['text'])
                    for i in range(batch_size):
                        mask = masks[i].cpu().numpy()
                        sentence = data['text']['bert'][i].cpu().numpy()
                        sentence = bert_id2tokens(sentence, mask)
                        results_to_save.append(' '.join(sentence))

                        ground_truth_labels = data['ner_span_labels'][i][data['ner_span_labels'][i] != 0]
                        ground_truth_spans = data['spans'][i][data['ner_span_labels'][i] != 0]
                        ground_truth_labels = label_id2tokens(ground_truth_labels.tolist(), model.vocab)
                        ground_truth = span2bio(ground_truth_spans, ground_truth_labels, sum(mask))
                        results_to_save.append(' '.join(ground_truth))

                        predicted_labels = output['predicted_named_entities'][i][
                            output['predicted_named_entities'][i] != 0]
                        if 'top_spans' in output:
                            predicted_spans = output['top_spans'][i][output['predicted_named_entities'][i] != 0]
                        else:
                            predicted_spans = data['spans'][i][output['predicted_named_entities'][i] != 0]
                        predicted_labels = label_id2tokens(predicted_labels.tolist(), model.vocab)
                        predicted = span2bio(predicted_spans, predicted_labels, sum(mask))
                        results_to_save.append(' '.join(predicted))

    metrics = model.get_metrics(True)
    metrics.update({'loss': total_loss / test_num_batches})
    if verbose:
        for key, value in metrics.items():
            logger.info('    {:15s}: {}'.format(str(key), value))
        if config.config.get('save_results'):
            results_path = config.resume.parent / 'result.txt'
            with open(results_path, 'wt') as f:
                f.write('\n'.join(results_to_save))
            print(results_path)
    return metrics


bert_tokenizer = BertTokenizer.from_pretrained('embeddings/scibert_scivocab_uncased/vocab.txt')


def bert_id2tokens(ids, mask):
    return [bert_tokenizer.ids_to_tokens[i] for i in ids[:sum(mask)]]


def label_id2tokens(ids, vocab):
    return [vocab.get_token_from_index(_id, 'ner_span_labels') for _id in ids]


def span2bio(spans, labels, seq_len):
    ret = ['O'] * seq_len
    for span, label in zip(spans, labels):
        ret[span[0]] = "B-" + label
        for i in range(span[0] + 1, span[1] + 1):
            ret[i] = "I-" + label
    return ret


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags target action')
    options = [
        CustomArgs(['-s', '--save_results'], target=('save_results',), action='store_true'),
    ]
    config = ConfigParser(args, options)
    main(config)
