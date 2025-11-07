import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn

class DC_and_FC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, fc_kwargs, weight_fc=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for FC and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_FC_loss, self).__init__()
        if ignore_label is not None:
            fc_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_fc = weight_fc
        self.ignore_label = ignore_label

        self.fc = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            force_reload=False,
            **fc_kwargs
        )
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_FC_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        # robust
        target_fc = target[:,0]
        if target_fc.ndim == net_output.ndim:
            assert target_fc.shape[1] == 1
            target_fc = target_fc[:, 0]
        target_fc = target_fc.long()

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        fc_loss = self.fc(net_output, target_fc) \
            if self.weight_fc != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_fc * fc_loss + self.weight_dice * dc_loss
        return result