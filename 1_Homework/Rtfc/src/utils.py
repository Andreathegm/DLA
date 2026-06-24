import torch
from torchvision.models import get_model

def get_device():
    return  torch.device("cuda" if torch.cuda.is_available() else "cpu")

def estimate_model_vram(model, device='cuda'):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    
    print(f"Model weights: {(param_bytes + buffer_bytes) / 1e6:.1f} MB")
    print(f"free Vram before loading: {torch.cuda.mem_get_info()[0]/1e6:.0f} MB")
    
    model.to(device)
    torch.cuda.synchronize()
    
    print(f"VRAM after loading: {torch.cuda.memory_allocated()/1e6:.0f} MB")
    print(f"VRAM reserved (with overhead): {torch.cuda.memory_reserved()/1e6:.0f} MB")

def list_models_feats(folder="models"):
    from pathlib import Path
    from collections import defaultdict

    groups = defaultdict(list)

    for file in sorted(Path(folder).glob("*.pt")):
        model_name = file.stem.replace("_gallery_feats", "").replace("_test_feats", "")
        groups[model_name].append(str(file))

    return [tuple(sorted(paths)) for paths in groups.values()]
    
model = get_model("vit_b_16", weights='DEFAULT')
estimate_model_vram(model,get_device())



