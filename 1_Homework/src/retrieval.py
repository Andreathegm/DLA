import torch

def compute_similarity_matrix(test_feats,gallery_feats):
    ## test_feats (N,embed_dim) , gallery_feats (M,embed_dim)
    test_norm = test_feats / test_feats.norm(dim=1,keepdim=True)
    gallery_norm = gallery_feats / gallery_feats.norm(dim=1,keepdim=True)

    ## sim_matrix must be a tensor of shape N*M
    sim_matrix = test_norm @ gallery_norm.T
    return sim_matrix

def compute_ranking(sim_matrix):
    # ranking will be a tensor of shape (N,M)
    # it returns a tensor that each value in it represents where is the value in the oginal tensor by row
    ranking = torch.argsort(sim_matrix,dim=1)
    return ranking

def average_precision(ranking,gallery_labels,true_label):
    # gallery_label shape is (M,)
    class_bitmap = (gallery_labels[ranking] == true_label).float()
    cumulative_sum = torch.cumsum(class_bitmap,dim=0)
    n_up_to_k = torch.arange(1,len(cumulative_sum)+1)
    precision_at_k = cumulative_sum / n_up_to_k

    if class_bitmap.sum() == 0 :
        return 0.0
    else:
        return (precision_at_k*class_bitmap).sum()/class_bitmap.sum().item()

def retrieval_evaluation(test_feats, test_labels, gallery_feats, gallery_labels, num_classes=43):
    sim_matrix = compute_similarity_matrix(test_feats, gallery_feats)
    ranking = compute_ranking(sim_matrix)

    class_AP = torch.zeros(num_classes)

    for c in range(num_classes):
        # Select test samples of class c
        idxs = torch.where(test_labels == c)[0]

        if len(idxs) == 0:
            continue

        APs = []
        for idx in idxs:
            ap = average_precision(ranking[idx], gallery_labels, c)
            APs.append(ap)

        class_AP[c] = torch.tensor(APs).mean()

    mAP = class_AP.mean().item()
    return class_AP, mAP

     


    



