import torch
import torch.nn as nn

class Dice_Coef_Loss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(Dice_Coef_Loss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        batch_size = y_true.shape[0]
        y_pred = torch.clamp(y_pred, 0, 1)
        y_true_f = y_true.view(batch_size, -1)
        y_pred_f = y_pred.view(batch_size, -1)

        intersection = torch.sum(y_true_f * y_pred_f, dim=-1)
        mask_sum = torch.sum(y_true_f, dim=-1) + torch.sum(y_pred_f, dim=-1)
        dice = (2. * intersection + self.smooth) / (mask_sum + self.smooth)
        
        return 1 - torch.mean(dice)
        
#Smoth는 zero division을 방지하기 위함