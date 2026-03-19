import torch
from src.retrieval import retrieval_evaluation
from src.feature_extractor import save_feats

# model_paths = [("models/resnet18_gallery_feats.pt","models/resnet18_test_feats.pt"),
#                ("models/resnet50_gallery_feats.pt","models/resnet50_test_feats.pt")
               
#                ]
model_paths = []

def main():
    model_names = ["vit_b_16"]
    model_transform = {"vit_b_16" : "vit"}
    folder = "models"
    for name in model_names:
        model_feats = (f"models/{name}_gallery_feats.pt",f"models/{name}_test_feats.pt")
        if model_feats not in model_paths:
            print(f"Downloading and saving feats... from {name}")
            save_feats(model_name=name,transform_string = model_transform[name])
            model_paths.append((f"models/{name}_gallery_feats.pt",f"models/{name}_test_feats.pt"))
        else:
            print(f"Already have features for {name} ")
        

    for path in model_paths:
        gallery_feats,gallery_labels = torch.load(path[0])
        test_feats,test_labels = torch.load(path[1])
        avg_precision_per_class,mAP = retrieval_evaluation(test_feats=test_feats,test_labels=test_labels,gallery_feats=gallery_feats,gallery_labels=gallery_labels)
        print(f"Avg precision per class for model{path} : \n{avg_precision_per_class}")
        print(f"mAP : {mAP}")
    


    pass
if __name__ == "__main__":
    main()
