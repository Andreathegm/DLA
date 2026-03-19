import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import get_model
from tqdm import tqdm
from .utils import get_device
from src.dataset import get_train_GTSRB_dl,get_test_GTSRB_dl

def extract_features(model: nn.Module, dataloader: DataLoader, device=None):
    device = device or get_device()


    model.eval()

    replaced = False
    for name in ["fc", "classifier", "head","heads"]:
        if hasattr(model, name):
            layer = getattr(model, name)
            if isinstance(layer, nn.Module):
                setattr(model, name, nn.Identity())
                replaced = True
                break

    if not replaced:
        raise RuntimeError(
            f"Model {model.__class__.__name__} not have a final 'standar' named layer "
            f"('fc', 'classifier', 'head'). Identify it manually in the following \n."
            f"{model}"
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

def test_extract_features(model,dataloader):
    features,labels = extract_features(model,dataloader)
    print(f"shape of feature is {features.shape} \n & shape of labels is {labels.shape}")
    return features,labels


def save_feats(model_name,batch_size=16,transform_string=None):
    model_name = model_name
    model = get_model(model_name, weights='DEFAULT')
    dl_train = get_train_GTSRB_dl("dataset/",batch_size=batch_size,transform_string=transform_string)
    dl_test = get_test_GTSRB_dl("dataset/",batch_size=batch_size,transform_string=transform_string)
    print(f"Info about train dataloader : \n{dl_train}")
    print(f"Info about test dataloader : \n{dl_test}")

    train_features,train_labels = test_extract_features(model,dl_train)
    test_features,test_labels = test_extract_features(model,dl_test)
    
    torch.save([train_features,train_labels],f"models/{model_name}_gallery_feats.pt")
    torch.save([test_features,test_labels],f"models/{model_name}_test_feats.pt")










