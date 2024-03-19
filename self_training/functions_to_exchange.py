def load_model(checkpoint_path, model):
    # reloads model
    def clean_state_dict(state):
        # clean state dict from names of PL
        #Removes reconstruction head
        for k in list(ckpt.keys()):
            if "model." == k[0:6]:
                ckpt[k[6:]] = ckpt[k]
                del ckpt[k]
            elif "dual_seg_head" in k:
                #Keep semantic segmentation layer
                pass
            elif "net." == k[0:4]:
                ckpt[k[4:]] = ckpt[k]
                del ckpt[k]
            else:    
                del ckpt[k]
        return state

    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
    print(ckpt.keys())
    ckpt = clean_state_dict(ckpt)
    model.load_state_dict(ckpt, strict=True)
    return model