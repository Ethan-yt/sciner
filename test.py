import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"

import torch
from tqdm import tqdm
import data_loader as module_data
import model as module_arch
from parse_config import ConfigParser
from allennlp.nn import util as nn_util


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    config['data_loader']['args']['shuffle'] = False

    data_loader = getattr(module_data, config['data_loader']['type'])(
        **config['data_loader']['args'],
    )

    # build model architecture
    model = config.initialize('arch', module_arch, vocab=data_loader.vocab)
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
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_iter)):
            data = nn_util.move_to_device(data, 0)
            output = model(**data)
            loss = output['loss']
            total_loss += loss.item()

    metrics = model.get_metrics(True)
    metrics.update({'loss': total_loss / test_num_batches})
    for key, value in metrics.items():
        logger.info('    {:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser(args)
    main(config)
