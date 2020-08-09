from typing import List, Optional, Union

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


class NERF1Metric(Metric):
    """
    micro averaged F1 metric
    excluding tag "O"
    """

    def __init__(self) -> None:
        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._true_positive_sum: Union[None, torch.Tensor] = None
        # the total number of instances
        # Shape: (num_classes, )
        self._total_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum: Union[None, torch.Tensor] = None

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to FBetaMeasure contains "
                                     f"an id >= {num_classes}, the number of classes.")

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes)
            self._true_sum = torch.zeros(num_classes)
            self._pred_sum = torch.zeros(num_classes)
            self._total_sum = torch.zeros(num_classes)

        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.to(torch.bool)
        gold_labels = gold_labels.float()

        argmax_predictions = predictions.max(dim=-1)[1].float()
        true_positives = (gold_labels == argmax_predictions) * mask
        true_positives_bins = gold_labels[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes)
        else:
            true_positive_sum = torch.bincount(true_positives_bins.long(), minlength=num_classes).float()

        pred_bins = argmax_predictions[mask].long()
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes)

        gold_labels_bins = gold_labels[mask].long()
        if gold_labels.shape[0] != 0:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum
        self._total_sum += mask.sum().to(torch.float)

    @overrides
    def get_metric(self,
                   reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precisions : List[float]
        recalls : List[float]
        f1-measures : List[float]

        If ``self.average`` is not ``None``, you will get ``float`` instead of ``List[float]``.
        """
        if self._true_positive_sum is None:
            raise RuntimeError("You never call this metric before.")

        tp_sum = self._true_positive_sum
        pred_sum = self._pred_sum
        true_sum = self._true_sum

        # Exclude 0 class
        tp_sum = tp_sum[1:].sum()
        pred_sum = pred_sum[1:].sum()
        true_sum = true_sum[1:].sum()

        beta2 = 1
        # Finally, we have all our sufficient statistics.
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        fscore = ((1 + beta2) * precision * recall /
                  (beta2 * precision + recall))
        fscore[tp_sum == 0] = 0.0

        if reset:
            self.reset()
        return {
            "avg_precision": precision.item(),
            "avg_recall": recall.item(),
            "avg_fscore": fscore.item()
        }

    @overrides
    def reset(self) -> None:
        self._true_positive_sum = None
        self._pred_sum = None
        self._true_sum = None
        self._total_sum = None

    @property
    def _true_negative_sum(self):
        if self._total_sum is None:
            return None
        else:
            true_negative_sum = self._total_sum - self._pred_sum - self._true_sum + self._true_positive_sum
            return true_negative_sum


def _prf_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result
