import torch
import os
import torch.nn as nn
from nnunet_mednext.network.SFScopeNet import SFScopeNet
from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

class nnUNetTrainerV2_Optim_and_LR(nnUNetTrainerV2):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-3

    def process_plans(self, plans):
        super().process_plans(plans)
        num_of_outputs_in_mednext = 5
        self.net_num_pool_op_kernel_sizes = [[2,2,2] for i in range(num_of_outputs_in_mednext+1)]    
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4
                                        )
        self.lr_scheduler = None


class nnUNetTrainerV2_MedNeXt_S_kernel3(nnUNetTrainerV2_Optim_and_LR):   
    def initialize_network(self):
        self.network = SFScopeNet(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2                 ,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

        if torch.cuda.is_available():
            self.network.cuda()
