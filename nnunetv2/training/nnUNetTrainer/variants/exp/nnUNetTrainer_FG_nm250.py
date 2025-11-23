import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import nnUNetTrainer_250epochs_NoMirroring

# i reviewed code about foreground oversampling
# when use p=0.33 and batch_size = 2 , it was actually equals to p=0.5 ( 1 in per batch  )
# and when p=0.33 and batch_size =4 , it becomes to p = 0.25 (1 in per batch)
# so we want to fix this, but not using probabilistic_oversampling=True

class nnUNetTrainer_fg50(nnUNetTrainer_250epochs_NoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.5

class nnUNetTrainer_fg75(nnUNetTrainer_250epochs_NoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.75