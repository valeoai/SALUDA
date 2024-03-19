import torch
import os
from torchsparse import SparseTensor

def get_dataset(base_class):

    # create a dataset class that will inherit from base_class
    class LCPDataset(base_class):

        def __init__(self, *args, **kwargs):

            if "network_function" in kwargs:
                net_func = kwargs["network_function"]
                del kwargs["network_function"]
            else:
                net_func = None

            super().__init__(*args, **kwargs)

            if net_func is not None:
                self.net = net_func()
            else:
                self.net = None


        def download(self):
            super().download()

        def process(self):
            super().process()

        def __getitem__(self, idx):

            data = super().__getitem__(idx)

            if (self.net is not None) and ("lcp_preprocess" in self.net.__dict__) and (self.net.__dict__["lcp_preprocess"]):

                with torch.no_grad():
                    return_data = self.net(data, spatial_only=True)

                for key in return_data.keys():
                    if return_data[key] is not None:
                        if isinstance(return_data[key], torch.Tensor):
                            data[key] = return_data[key].detach()
                        else:
                            data[key] = return_data[key]

            # remove None type keys
            to_delete_keys = []
            for key in data:
                if data[key] is None:
                    to_delete_keys.append(key)

            for key in to_delete_keys:
                data.pop(key, None)
            
            return data

    return LCPDataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def list_to_device(data, device):
    for key, value in enumerate(data):
        if torch.is_tensor(value):
            data[key] = value.to(device)
        elif isinstance(value, list):
            data[key] = list_to_device(value, device)
        elif isinstance(value, dict):
            data[key] = dict_to_device(value, device)
    return data

def dict_to_device(data, device):
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device)
        elif isinstance(value, list):
            data[key] = list_to_device(value, device)
        elif isinstance(value, dict):
            data[key] = dict_to_device(value, device)
        elif isinstance(value, SparseTensor):
            data[key] = data[key].to(device)
    return data

def logs_file(filepath, epoch, log_data):

    
    if not os.path.exists(filepath):
        log_str = f"epoch"
        for key, value in log_data.items():
            log_str += f", {key}"
        log_str += "\n"
        with open(filepath, "a+") as logs:
            logs.write(log_str)
            logs.flush()

    # write the logs
    log_str = f"{epoch}"
    for key, value in log_data.items():
        log_str += f", {value}"
    log_str += "\n"
    with open(filepath, "a+") as logs:
        logs.write(log_str)
        logs.flush()