import torch
import yaml

def get_device():
    return  torch.device("cuda" if torch.cuda.is_available() else "cpu")

def estimate_model_vram(model, device='cuda'):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    
    print(f"Model weights: {(param_bytes + buffer_bytes) / 1e6:.1f} MB")
    ## torch.cuda.mem_get_info() returns: [0] free memory , [1] all available memory
    print(f"free Vram before loading: {torch.cuda.mem_get_info()[0]/1e6:.0f} MB")
    
    model.to(device)
    ##  GPU can operate asyncr. so we basically wait the model to be loaded
    torch.cuda.synchronize()

    
    print(f"VRAM after loading: {torch.cuda.memory_allocated()/1e6:.0f} MB")
    print(f"VRAM reserved: {torch.cuda.memory_reserved()/1e6:.0f} MB")

def list_models_feats(folder="models"):
    from pathlib import Path
    from collections import defaultdict

    groups = defaultdict(list)

    # takes every .pt file inside the folder
    for file in sorted(Path(folder).glob("*.pt")):
        # stem removes the extension of the file
        model_name = file.stem.replace("_gallery_feats", "").replace("_test_feats", "")
        groups[model_name].append(str(file))

    # groups is a dict like:
    # {
    #   "model_x": ["model_x_gallery_feats.pt", "model_x_test_feats.pt"],
    #   "model_y": [...],
    # }
    return [tuple(sorted(paths)) for paths in groups.values()]

def load_yaml(config_path):
    with open(config_path,'r') as f:
        config = yaml.safe_load(f)
    return config



