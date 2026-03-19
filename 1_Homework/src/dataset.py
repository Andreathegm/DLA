from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
from torchvision.models import list_models, get_model
import os
import torchvision.transforms.v2 as T


mean = [0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
worker = min(8, os.cpu_count() // 2)


def get_GTSRB_ds(folder,split,transform,download=True):
    return GTSRB(folder, split=split, transform=transform, download=download)

def get_train_GTSRB_ds(folder,transform_string=None):
    trasform = get_transform(transform_string)
    return get_GTSRB_ds(folder,"train",trasform)

def get_test_GTSRB_ds(folder,transform_string=None):
    trasform = get_transform(transform_string)
    return get_GTSRB_ds(folder,"test",trasform)

def get_train_GTSRB_dl(folder,batch_size,transform_string=None):
    return DataLoader(dataset=get_train_GTSRB_ds(folder,transform_string),
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=worker,
                      pin_memory=True,   
                      persistent_workers=True )

def get_test_GTSRB_dl(folder,batch_size,transform_string=None):

    return DataLoader(
                      dataset=get_test_GTSRB_ds(folder,transform_string),
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=worker,
                      pin_memory=True,   
                      persistent_workers=True 
                      )

def get_transform(trasform_str):
    match trasform_str:
        case "train":
            return T.Compose([
                T.Resize(70),
                T.RandomCrop((64, 64)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        case "test":
            return T.Compose([
                T.Resize(70),
                T.CenterCrop((64, 64)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        
        case "vit": 
            return T.Compose([
                T.Resize(256),
                T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
            
        case _:
            return T.Compose([
                T.Resize(70),
                T.CenterCrop((64, 64)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        