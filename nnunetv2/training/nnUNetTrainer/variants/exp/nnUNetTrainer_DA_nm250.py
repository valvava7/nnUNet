from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import nnUNetTrainer_250epochs_NoMirroring
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_250epochs
import numpy as np

class nnUNetTrainer_onlyMirror01_250(nnUNetTrainer_250epochs):
    """
    Only mirrors along spatial axes 0 and 1 for 3D and 0 for 2D
    """
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 2:
            mirror_axes = (0, )
        else:
            mirror_axes = (0, 1)
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

class nnUNetTrainer_DA1_nm250(nnUNetTrainer_250epochs_NoMirroring):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
        
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes