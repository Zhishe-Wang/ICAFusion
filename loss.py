from torch import nn
import torch


class g_content_loss(nn.Module):
    def __init__(self):
        super(g_content_loss, self).__init__()
        self.F_loss = nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss(reduction="mean")
    def forward(self, img_ir, img_vi, img_fusion):
        x = 1
        F_loss = self.F_loss(img_fusion,img_ir)
        L1_loss = x*self.L1_loss(img_fusion,img_vi)
        content_loss = F_loss + L1_loss
        return content_loss,  L1_loss , F_loss


