from src.feature_extractor import extract_features
from src.dataset import get_train_GTSRB_dl,get_test_GTSRB_dl
from torchvision.models import get_model
import torch

def test_extract_features(model,dataloader):
    features,labels = extract_features(model,dataloader)
    print(f"shape of feature is {features.shape} \n & shape of labels is {labels.shape}")
    return features,labels


batch_size = 16
model_name = "resnet18"
model = get_model('resnet18', weights='DEFAULT')
dl_train = get_train_GTSRB_dl("dataset/",batch_size)
dl_test = get_test_GTSRB_dl("dataset/",batch_size)
print(f"Info about train dataloader : \n{dl_train}")
print(f"Info about test dataloader : \n{dl_test}")
train_features,train_labels = test_extract_features(model,dl_train)
test_features,test_labels = test_extract_features(model,dl_test)
torch.save([train_features,train_labels],f"{model_name}_gallery_feats.pt")
torch.save([test_features,test_labels],f"{model_name}_test_feats.pt")



