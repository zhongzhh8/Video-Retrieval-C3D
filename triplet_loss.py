import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import itertools

class TripletLoss(nn.Module):
    #pass
    def __init__(self,margin,device):
        super(TripletLoss, self).__init__()
        self.margin=margin
        self.device=device
    def similarity(self,label1,label2):
        return label1==label2 # default with singe label

    def forward(self,x,labels):
        self.batch_size=x.size()[0]
        self.feature_size=x.size()[1]
        triplet_loss=torch.tensor(0.0).to(self.device)
        semihard_triplet_loss=torch.tensor(0.0).to(self.device)
        #start=time.clock()
        labels_=labels.cpu().data.numpy()
        triplets=[]
        for label in labels_:
            label_mask=(labels_==label)
            label_indices=np.where(label_mask)[0]
            if len(label_indices)<2:
                continue
            negative_indices=np.where(np.logical_not(label_mask))[0]
            if len(negative_indices)<1:
                continue
            anchor_positives=list(itertools.combinations(label_indices, 2))
            temp=[[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                        for neg_ind in negative_indices]
            triplets+=temp
            #end=time.clock()
        #print ("triplets mining time: %s Seconds"%(end-start))
        if triplets:
            triplets=np.array(triplets)
            #print triplets
            sq_ap=(x[triplets[:, 0]]-x[triplets[:, 1]]).pow(2).sum(1)  
            sq_an=(x[triplets[:, 0]]-x[triplets[:, 2]]).pow(2).sum(1)  
            losses=F.relu(self.margin+sq_ap-sq_an)
            triplet_count=torch.tensor(losses.size()[0]).float().to(self.device)
            semihard_triplet_count=(losses!=0).sum().float().to(self.device)
            if triplet_count>0:
                triplet_loss=losses.sum()/triplet_count
            if semihard_triplet_count>0:
                semihard_triplet_loss=losses.sum()/semihard_triplet_count
            # print ("triplet_count", triplet_count)
            # print ("semihard_triplet_count", semihard_triplet_count)
            # print ("triplet_loss:",triplet_loss.item())
            # print ("semihard_triplet_loss",semihard_triplet_loss.item())
        
        return triplet_loss
        # return semihard_triplet_loss


