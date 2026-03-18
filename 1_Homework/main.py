import torch
from src.retrieval import retrieval_evaluation
def main():
    gallery_feats,gallery_labels = torch.load("resnet18_gallery_feats.pt")
    test_feats,test_labels = torch.load("resnet18_test_feats.pt")
    avg_precision_per_class,mAP = retrieval_evaluation(test_feats=test_feats,test_labels=test_labels,gallery_feats=gallery_feats,gallery_labels=gallery_labels)
    print(avg_precision_per_class)
    print(mAP)


    pass
if __name__ == "__main__":
    main()
