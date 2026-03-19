import torch
from src.retrieval import retrieval_evaluation
from src.feature_extractor import save_feats
import os
from src.utils import list_models_feats

model_paths = []

def main():
    model_names = [
                # 'efficientnet_b0',    # baseline moderna leggera
                # 'efficientnet_v2_s',  # v2 più efficiente      # CNN moderna stile transformer
                # 'resnet50',           # classico di riferimento
                # 'swin_t',
                #'vit_b_16', 
                # resnet18    

                    ]
    
    model_transform = {name : "vit" for name in model_names}
    print(model_transform)
    for name in model_names:
        model_feats = (f"models/{name}_gallery_feats.pt",f"models/{name}_test_feats.pt")
        if model_feats not in list_models_feats():
         
            print(f"Downloading and saving feats... from {name}")
            save_feats(model_name=name,transform_string = model_transform[name])
            model_paths.append(model_feats)
        else:
            model_paths.append(model_feats)
            print(f"Already have features for {name} ")
            print(model_paths)
        

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
    


    pass
if __name__ == "__main__":
    main()
