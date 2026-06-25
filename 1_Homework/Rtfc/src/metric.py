import torch
import os
from .retrieval import retrieval_evaluation

def calculate_mAP(model_paths : tuple ,config : dict,save_dir = "results"):
    model_names = list(config.keys())
    for i,path in enumerate(model_paths):
        gallery_feats,gallery_labels = torch.load(path[0])
        test_feats,test_labels = torch.load(path[1])
        avg_precision_per_class,mAP = retrieval_evaluation(test_feats=test_feats,test_labels=test_labels,gallery_feats=gallery_feats,gallery_labels=gallery_labels)
        print(f"Avg precision per class for model{path} : \n{avg_precision_per_class}")
        print(f"mAP : {mAP}")

        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "mAP": mAP,
                "avg_precision_per_class": avg_precision_per_class,
                "model_path": str(path[0]),
            },
            f"{save_dir}/{model_names[i]}.pt"
        )