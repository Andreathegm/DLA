import torch 
import torch.nn.functional as F
import os

def nearest_mean_classifier(gallery_feats, gallery_labels, test_feats, test_labels, num_classes=43):
    # Normalization of vectors
    #Already normalized
    #gallery_feats = F.normalize(gallery_feats, p=2, dim=1) # (N,D)
    #test_feats    = F.normalize(test_feats,    p=2, dim=1) # (M,D)

    # stack evey class mean vector in a new tensor
    means = torch.stack([
        gallery_feats[gallery_labels == c].mean(dim=0)
        for c in range(num_classes)
    ])
    means = F.normalize(means, p=2, dim=1)  # (num_classes, D)

    # basically calculated the cosine similiraty between test_feats and every class mean
    sims  = test_feats @ means.T   # (M, num_classes)
    preds = sims.argmax(dim=1)

    acc = (preds == test_labels).float().mean().item()
    per_class_acc = torch.tensor([
        (preds[test_labels == c] == c).float().mean().item()
        for c in range(num_classes)
    ])

    return acc, per_class_acc, preds

def near_mean_classify(model_paths : tuple,config ,save_dir = "classify"):

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

        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "acc" : acc,
                "acc_per_class" : acc_per_class,
            },f"{save_dir}/{model_names[i]}.pt"
        )