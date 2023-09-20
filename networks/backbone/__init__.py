def get_backbone(name):
    if name == "TorchSparseMinkUNet":
        from .torchsparse_minkunet import TorchSparseMinkUNet
        return TorchSparseMinkUNet
    else:
        raise ValueError("Unknown backbone")
