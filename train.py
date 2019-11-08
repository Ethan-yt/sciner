import argparse
import collections
import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import data_loader as module_data
import model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from test import main as test
from trainer import optimizer as custom_optimizer


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch, vocab=data_loader.vocab)
    logger.info(model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    try:
        lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    except AttributeError:
        lr_scheduler = config.initialize('lr_scheduler', custom_optimizer, optimizer)

    trainer = Trainer(model, optimizer,
                      data_loader=data_loader,
                      config=config,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    config.resume = config.save_dir / 'model_best.pth'
    result = test(config, verbose=False)
    for key, value in result.items():
        logger.info('{:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config/config.json', type=str,
                      help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    # prepare_environment(Params({'random_seed': 1, 'numpy_seed': 1, 'pytorch_seed': 1}))
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    main(config)
