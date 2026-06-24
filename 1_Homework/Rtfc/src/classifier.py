import torch 
import torch.nn.functional as F

def nearest_mean_classifier(gallery_feats, gallery_labels, test_feats, test_labels, num_classes=43):
    # Normalization of vectors
    gallery_feats = F.normalize(gallery_feats, p=2, dim=1) # (N,D)
    test_feats    = F.normalize(test_feats,    p=2, dim=1) # (M,D)

    # stach evey class mean vector in a new tensor
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