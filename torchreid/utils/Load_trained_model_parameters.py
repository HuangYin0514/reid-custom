# ！ python

import torchreid
import torch


class Load_trained_parameters():
    """
    Load trained parameters

    used method：

    load_path = '/home/hy/vscode/reid-custom/log/pcb_p4/model/model.pth.tar-8'
    Load_trained_parameters(load_path).load_trained_model_weights(model)
    """

    def __init__(self, load_path_url, device):
        self.__checkpoint_states = {
            'state_dict': None,
            'optimizer': None,
            'scheduler': None,
            'epoch': None,
            'rank1': None
        }
        self.load_path_url = load_path_url
        self.device = device
        self.load_saved_path(self.load_path_url)


    def load_saved_path(self, load_path_url):
        """load paramters with path.

        from load_path (str)
        """
        checkpoint = torch.load(load_path_url, map_location=self.device)
        for _ in checkpoint:
            if _ in checkpoint.keys() and self.__checkpoint_states:
                self.__checkpoint_states[_] = checkpoint[_]
                print(f'load state {_}')

    def load_trained_model_weights(self, model):
        """load model with trained weights.

        Layers that don't match with pretrained layers in name or size are kept unchanged.
        """
        model_dict = model.state_dict()
        pretrain_dict = {
            k: v
            for k, v in self.__checkpoint_states['state_dict'].items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        self.print_load_module(pretrain_dict)

    def print_load_module(self, module_dict):
        """
        print laoded module name of frist 
        """
        load_module = set(map(lambda i: i.split('.')[0], module_dict.keys()))
        print(f'load module name is {load_module}')
