import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import get_model
from tqdm import tqdm
from .utils import get_device
from .dataset import get_train_GTSRB_dl,get_test_GTSRB_dl
import os

def extract_features(model: nn.Module, dataloader: DataLoader,replace_head=False,device=None):
    device = device or get_device()
    model.eval()

    if replace_head :
        replaced = classifier_takeout(model)

        if not replaced:
            raise RuntimeError(
                f"Model {model.__class__.__name__} has no recognizable final layer "
                f"('fc', 'classifier', 'head', 'heads'). Inspect it manually:\n{model}"
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
    feats = torch.vstack(feats) ## (N,D)
    classes = torch.cat(classes) ## (N,)

    return feats, classes

def test_extract_features(model,dataloader,replace_head = False):
    features,labels = extract_features(model,dataloader,replace_head)
    print(f"shape of feature is {features.shape} \n & shape of labels is {labels.shape}")
    return features,labels


def save_feats(model_name,batch_size=128,transform_string=None,features_path = "features"):
    print(f"Start pipeline feature extraction for {model_name} ")
    model = get_model(model_name, weights='DEFAULT')
    dl_train = get_train_GTSRB_dl("dataset/",batch_size=batch_size,transform_string=transform_string)
    dl_test = get_test_GTSRB_dl("dataset/",batch_size=batch_size,transform_string=transform_string)
    
    if str(dl_train.dataset.transform) != str(dl_test.dataset.transform) :
        raise ValueError(f"train transform =  {dl_train.dataset.transform} is different than test transform = {dl_test.dataset.transform}")



        
    train_features,train_labels = test_extract_features(model,dl_train,replace_head = True)
    test_features,test_labels = test_extract_features(model,dl_test)
    
    os.makedirs(features_path, exist_ok=True)
    torch.save([train_features,train_labels],f"{features_path}/{model_name}_gallery_feats.pt")
    torch.save([test_features,test_labels],f"{features_path}/{model_name}_test_feats.pt")
    return (train_features,train_labels),(test_features,test_labels)


def classifier_takeout(model):

    replaced = False

    # heads: Sequential with final Linear (e.g. swin_t)
    if not replaced and hasattr(model, "heads") and isinstance(model.heads, nn.Sequential):
        last_key = list(model.heads._modules.keys())[-1]
        if isinstance(model.heads._modules[last_key], nn.Linear):
            setattr(model.heads, last_key, nn.Identity())
            replaced = True
            return replaced

    # classifier: Sequential with final Linear (e.g. convnext, efficientnet)
    if not replaced and hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last_key = list(model.classifier._modules.keys())[-1]
        if isinstance(model.classifier._modules[last_key], nn.Linear):
            setattr(model.classifier, last_key, nn.Identity())
            replaced = True
            return replaced

    # fc / classifier / head: direct Linear (e.g. resnet, googlenet)
    if not replaced:
        for name in ["fc", "classifier", "head"]:
            if hasattr(model, name) and isinstance(getattr(model, name), nn.Linear):
                setattr(model, name, nn.Identity())
                replaced = True
                return replaced

    return replaced







