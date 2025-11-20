import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import nnUNetTrainer_250epochs_NoMirroring
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdam import nnUNetTrainerVanillaAdam, nnUNetTrainerAdam1en3, nnUNetTrainerAdam3en4
from nnunetv2.training.nnUNetTrainer.variants.optimizer.nnUNetTrainerAdan import nnUNetTrainerAdan1en3, nnUNetTrainerAdan3en4

class nnUNetTrainer_vadam_nm250(nnUNetTrainer_250epochs_NoMirroring, nnUNetTrainerVanillaAdam):
    pass

class nnUNetTrainer_adam1en3_nm250(nnUNetTrainer_250epochs_NoMirroring, nnUNetTrainerAdam1en3):
    pass

class nnUNetTrainer_adan1en3_nm250(nnUNetTrainer_250epochs_NoMirroring, nnUNetTrainerAdan1en3):
    pass

class nnUNetTrainer_adan3en4_nm250(nnUNetTrainer_250epochs_NoMirroring, nnUNetTrainerAdan3en4):
    pass

class nnUNetTrainer_adam3en4_nm250(nnUNetTrainer_250epochs_NoMirroring, nnUNetTrainerAdam3en4):
    pass

class nnUNetTrainer_1en3_nm250(nnUNetTrainer_250epochs_NoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr=1e-3

class nnUNetTrainer_adam1en3_nm250_wd0(nnUNetTrainer_250epochs_NoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight_decay = 0

class nnUNetTrainer_adam1en3_nm250_wd3en6(nnUNetTrainer_250epochs_NoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight_decay = 3e-6


