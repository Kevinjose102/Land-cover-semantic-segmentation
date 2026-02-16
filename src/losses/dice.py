import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        # convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # convert targets to one-hot
        targets_one_hot = F.one_hot(targets, self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # compute intersection and union
        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs, dims) + torch.sum(targets_one_hot, dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1.0 - dice.mean()

        return dice_loss
