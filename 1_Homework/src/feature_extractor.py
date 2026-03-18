import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import get_device
from tqdm import tqdm

def extract_features(model: nn.Module, dataloader: DataLoader, device=None):
    device = device or get_device()


    model.eval()

    replaced = False
    for name in ["fc", "classifier", "head"]:
        if hasattr(model, name):
            layer = getattr(model, name)
            if isinstance(layer, nn.Module):
                setattr(model, name, nn.Identity())
                replaced = True
                break

    if not replaced:
        raise RuntimeError(
            f"Model {model.__class__.__name__} not have a final 'standar' named layer "
            f"('fc', 'classifier', 'head'). Identify it manually using print(model)."
        )

    model = model.to(device)

    feats = []
    classes = []

    with torch.no_grad():
        for ims, cls in tqdm(dataloader):

            ims = ims.to(device)
            feats.append(model(ims).cpu())
            classes.append(cls)

    # 5. Concatena tutto
    feats = torch.vstack(feats)
    classes = torch.cat(classes)

    return feats, classes










