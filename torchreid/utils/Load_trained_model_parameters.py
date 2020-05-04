# ！ python

import torchreid
import torch


class Load_trained_parameters():
    """
    Load trained parameters

    used method：
    >>>     load_path = '/home/hy/vscode/reid-custom/log/pcb_p4/model/model.pth.tar-8'
    >>>     Load_trained_parameters(load_path).load_trained_model_weights(model)
    """

    def __init__(self, load_path_url=None, device=None, model=None, is_pretrained=False):
        self.checkpoint_states = {
            'state_dict': None,
            'optimizer': None,
            'scheduler': None,
            'epoch': None,
            'rank1': None
        }
        self.load_path_url = load_path_url
        self.device = device
        self.is_pretrained = is_pretrained
        self.model = model
        self.is_pretrained_model()

    def is_pretrained_model(self):
        """
        assert model have a checkpoint file to load
        """
        if self.is_pretrained:
            assert self.load_path_url, "load_path_url is None"
            assert self.device, "device is None"
            assert self.model, "model is None"
            self.load_saved_path(self.load_path_url)
            self.load_trained_model_weights(self.model)
        else:
            return

    def load_saved_path(self, load_path_url):
        """load paramters with path.

        from load_path (str)
        """
        checkpoint = torch.load(load_path_url, map_location=self.device)
        load_state = []
        for _ in checkpoint:
            if _ in checkpoint.keys() and self.checkpoint_states:
                self.checkpoint_states[_] = checkpoint[_]
                load_state.append(_)
        rank1 = self.checkpoint_states['rank1']
        epoch = self.checkpoint_states['epoch']
        print(f'load state {load_state} of rank1 is {rank1} and epoch is {epoch}')

    def load_trained_model_weights(self, model):
        """load model with trained weights.

        Layers that don't match with pretrained layers in name or size are kept unchanged.
        """
        model_dict = model.state_dict()
        pretrain_dict = {
            k[7:]: v
            for k, v in self.checkpoint_states['state_dict'].items()
            if k[7:] in model_dict and model_dict[k[7:]].size() == v.size()
        }
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        self.print_load_module(pretrain_dict)
        
        # unload module
        unload_dict = {
            k[7:]: v
            for k, v in self.checkpoint_states['state_dict'].items()
            if not (k[7:] in model_dict and model_dict[k[7:]].size() == v.size())
        }

        self.print_unload_module(unload_dict)

    def print_load_module(self, module_dict):
        """
        print laoded module name of frist
        """
        load_module = set(map(lambda i: i.split('.')[0], module_dict.keys()))
        print(f'load module name is {load_module}')

    def print_unload_module(self, module_dict):
        """
        print unlaoded module name of frist 
        """
        load_module = sorted(set(map(lambda i: i, module_dict.keys())))
        print(f'unload module name is {load_module}')
