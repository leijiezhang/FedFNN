import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """

    def __init__(self):
        super(BalancedSoftmax, self).__init__()

    def forward(self, logits, labels):
        """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
            Args:
              labels: A int tensor of size [batch].
              logits: A float tensor of size [batch, no_of_classes].
            Returns:
              loss: A float tensor. Balanced Softmax Loss.
            """
        sample_per_class = torch.ones(logits.shape[0], logits.shape[1]).to(logits.device)
        y_list = labels.unique()
        for y_item in y_list:
            y_item_num = (labels == y_item).sum()
            sample_per_class[:, y_item] = sample_per_class[:, y_item] * y_item_num
        spc = sample_per_class.type_as(logits)
        # spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(input=logits, target=labels)
        return loss
