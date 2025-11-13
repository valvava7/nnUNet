import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import nnUNetTrainer_250epochs_NoMirroring
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
import numpy as np


class nnUNetTrainerDC_WCELoss_NM250(nnUNetTrainer_250epochs_NoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight=(1,1,1,1)
        self.ce_args = {}
        self.dc_args = {'batch_dice': self.configuration_manager.batch_dice,'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}
        self.weight_ce = 1
        self.weight_dice = 1

    def _build_loss(self):
        self.ce_args['weight'] = torch.tensor(self.weight, device=self.device, dtype=torch.float32)
        assert not self.label_manager.has_regions, 'not implemented'
        loss = DC_and_CE_loss(self.dc_args,
                                self.ce_args,
                                self.weight_ce, self.weight_dice,
                                ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

class nnUNetTrainerDC_WCELoss1114_NM250(nnUNetTrainerDC_WCELoss_NM250):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight=(1,1,1,4)

class nnUNetTrainerDC_WCELoss1124_NM250(nnUNetTrainerDC_WCELoss_NM250):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight=(1,1,2,4)

class nnUNetTrainerDC_WCE03Loss_NM250(nnUNetTrainerDC_WCELoss_NM250):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight=(1,1,1,1)
        self.weight_ce = 0.3

class nnUNetTrainerDC_WCELoss_5epochs__NM250(nnUNetTrainerDC_WCELoss_NM250):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5

