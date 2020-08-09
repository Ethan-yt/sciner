import numpy as np
import torch
from allennlp.common.util import lazy_groups_of
from allennlp.training import util as training_util
from allennlp.nn import util as nn_util
from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, optimizer, config, data_loader,
                 lr_scheduler=None):
        super().__init__(model, optimizer, config)
        self.config = config
        self.data_loader = data_loader

        self.lr_scheduler = lr_scheduler

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.train_iter, self.train_num_batches = self.data_loader.get_iterator_and_num_batches('train')
        self.train_iter = lazy_groups_of(self.train_iter, self.n_gpu_use)

        self.model.train()

        total_loss = 0
        for batch_idx, data in enumerate(self.train_iter):
            self.optimizer.zero_grad()

            output = self._run_model(data)
            loss = output['loss']
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.train_num_batches + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()

            if batch_idx % (self.train_num_batches // 5) == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
        metrics = self.model.get_metrics(True)
        metrics = {'train_' + k: v for k, v in metrics.items()}
        metrics.update({
            'train_loss': total_loss / self.train_num_batches,
        })

        val_log = self._valid_epoch(epoch)
        metrics.update(val_log)

        # test_log = self._test_epoch()
        # metrics.update(test_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return metrics

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.valid_iter, self.valid_num_batches = self.data_loader.get_iterator_and_num_batches('dev')
        self.valid_iter = lazy_groups_of(self.valid_iter, self.n_gpu_use)

        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_iter):
                output = self._run_model(data)
                loss = output['loss']
                # predict = output['predict']

                self.writer.set_step((epoch - 1) * self.valid_num_batches + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        metrics = self.model.get_metrics(True)
        # metrics = {"val" + k: v for k, v in metrics.items()}
        metrics.update({
            'val_loss': total_val_loss / self.valid_num_batches,
        })

        return metrics

    def _test_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.valid_iter, self.valid_num_batches = self.data_loader.get_iterator_and_num_batches('test')
        self.valid_iter = lazy_groups_of(self.valid_iter, self.n_gpu_use)

        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_iter):
                output = self._run_model(data)
                loss = output['loss']
                total_val_loss += loss.item()

        metrics = self.model.get_metrics(True)
        metrics.update({
            'test_loss': total_val_loss / self.valid_num_batches,
        })

        return metrics

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.train_num_batches
        return base.format(current, total, 100.0 * current / total)

    def _run_model(self, batch_group):
        if self.n_gpu_use > 1:
            output_dict = training_util.data_parallel(batch_group, self.model, self.device)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self.device[0])
            output_dict = self.model(**batch)
        return output_dict

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")

        model_state_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if
                           k in model_state_dict and model_state_dict[k].shape == v.shape}
        model_state_dict.update(pretrained_dict)
        self.model.load_state_dict(model_state_dict)

        self.logger.info("Checkpoint loaded.")
