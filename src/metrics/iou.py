import torch

def compute_iou(preds, masks, num_classes):
    ious = []

    preds = preds.view(-1)
    masks = masks.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = masks == cls

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(torch.tensor(1.0))  # if class not present
        else:
            ious.append(intersection / union)

    return torch.stack(ious)
