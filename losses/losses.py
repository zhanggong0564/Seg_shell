import torch.nn as nn
import torch
import torch.nn.functional as F



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1

    def forward(self,output,target):
        output = torch.sigmoid(output)
        target = F.one_hot(target,num_classes=4)

        N,C,H,W = output.size()
        output = output.view(N,C,-1)
        output = output.transpose(1,2)
        output = output.contiguous().view(-1,C)

        target = target.view(-1,C)

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = (2*intersect+self.eps) / (denominator+self.eps)
        dice = torch.mean(dice)
        return 1 - dice
class FacalLoss(nn.Module):
    def __init__(self,gamma=2):
        super(FacalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss()
    def forward(self,outputs,targets):
        pt = F.softmax(outputs,dim=1)
        factor = torch.pow(1-pt,self.gamma)
        log_socre = F.log_softmax(outputs,dim=1)
        log_socre = factor*log_socre
        loss = self.nll(log_socre,targets)
        return loss

class Focal_Dice(nn.Module):
    def __init__(self,w=0.5):
        super(Focal_Dice, self).__init__()
        self.w = w
        self.focal =FacalLoss()
        self.dice = DiceLoss()
    def forward(self,outputs,targets):
        loss1 = self.focal(outputs,targets)
        loss2 = self.dice(outputs,targets)
        loss = self.w*loss1+(1-self.w)*loss2
        return loss



# class OhemCELoss(nn.Module):
#     def __init__(self,thresh,n_min):
#         super(OhemCELoss, self).__init__()
#         self.thresh = -torch.log(torch.tensor(thresh,torch.float)).cuda()
#         self.n_min = n_min
#         self.