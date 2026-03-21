import torch
from src.retrieval import retrieval_evaluation
from src.feature_extractor import save_feats
import os
from src.utils import list_models_feats
from src.classifier import nearest_mean_classifier
import yaml

model_paths = []

def main():

    config_path = "config/models.yaml"
    config = load_yaml(config_path)

    for name in config:
        model_feats = (f"models/{name}_gallery_feats.pt",f"models/{name}_test_feats.pt")

        if model_feats not in list_models_feats():
            return
            print(f"Downloading and saving feats... from {name}")
            save_feats(model_name=name,transform_string = config[name])
            model_paths.append(model_feats)
        else:
            model_paths.append(model_feats)
            print(f"Already have features for {name} ")
            print(model_paths)
        

    if False:
        calculate_mAP(model_paths= model_paths,config=config)
    near_mean_classify(model_paths=model_paths,config=config)
    


def load_yaml(config_path):
    with open(config_path,'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_mAP(model_paths : tuple ,config : dict):
    model_names = list(config.keys())
    for i,path in enumerate(model_paths):
        gallery_feats,gallery_labels = torch.load(path[0])
        test_feats,test_labels = torch.load(path[1])
        avg_precision_per_class,mAP = retrieval_evaluation(test_feats=test_feats,test_labels=test_labels,gallery_feats=gallery_feats,gallery_labels=gallery_labels)
        print(f"Avg precision per class for model{path} : \n{avg_precision_per_class}")
        print(f"mAP : {mAP}")

        os.makedirs("results", exist_ok=True)
        torch.save(
            {
                "mAP": mAP,
                "avg_precision_per_class": avg_precision_per_class,
                "model_path": str(path[0]),
            },
            f"results/{model_names[i]}.pt"
        )
def near_mean_classify(model_paths : tuple,config):

    model_names = list(config.keys())
    for i,path in enumerate(model_paths):
        gallery_feats,gallery_labels = torch.load(path[0])
        test_feats,test_labels = torch.load(path[1])
        acc,acc_per_class, _ = nearest_mean_classifier(gallery_feats=gallery_feats,
                                                       gallery_labels=gallery_labels,
                                                       test_feats=test_feats,
                                                       test_labels=test_labels)
        
        print(f"Accuracy for model{path} : \n{acc}")
        print(f"Accuracy per class : {acc_per_class}")

        os.makedirs("classify", exist_ok=True)
        torch.save(
            {
                "acc" : acc,
                "acc_per_class" : acc_per_class,
            },f"classify/{model_names[i]}.pt"
        )





if __name__ == "__main__":
    main()
