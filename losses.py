import torch
import torch.nn as nn
import torch.nn.functional as F

#  color loss that uses color histogram
class ColorLoss(nn.Module):
    def __init__(self,cluster_number=32):
        super(ColorLoss, self).__init__()
        self.cluster_number=cluster_number
        self.criterion = nn.L1Loss()
        self.spacing=2/cluster_number

    def calc_hist(self,data_ab):
        H = data_ab.size(0)
        grid_a = torch.linspace(-1, 1, self.cluster_number + 1).view(self.cluster_number + 1, 1).expand(
            self.cluster_number + 1, H).cuda()
        hist_a = torch.max(self.spacing - torch.abs(grid_a - data_ab.view(-1)), torch.Tensor([0]).cuda()) * 10
        # return hist_a.mean(dim=1).view(-1) * H
        return hist_a.mean(dim=1).view(-1)  # removal H
    def forward(self,A_img,A_mask,B_img,B_mask):

        b,c,h,w=A_img.size()
        loss = 0
        for j in range(b):
            for i in range(3):
                temp_A = torch.masked_select(A_img[j,i:i+1,::],A_mask[j,i:i+1,::]>0.5).cuda()
                temp_B = torch.masked_select(B_img[j, i:i + 1, ::], B_mask[j, i:i + 1, ::] > 0.5).cuda()
                if temp_A.size(0)==0 or temp_B.size(0)==0:
                    continue
                loss+=self.criterion(self.calc_hist(temp_A),self.calc_hist(temp_B))
        return loss/(b*3)