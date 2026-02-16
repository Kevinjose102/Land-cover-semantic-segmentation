import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import LandCoverDataset
from src.models.unet import UNet
from src.losses.dice import DiceLoss
from src.metrics.iou import compute_iou

import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_ROOT = r"D:\kevin stuff\Brave_Downloads\mlproject\datasets"
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-3
NUM_CLASSES = 5


def main():

    # dataset object
    full_dataset = LandCoverDataset(
        images_dir=f"{DATASET_ROOT}/images",
        masks_dir=f"{DATASET_ROOT}/masks",
        patch_size=512
    )

    # dividing the dataset into training and validation subset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # loads the data into batches
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # fixed validation batch for visualization
    fixed_images, fixed_masks = next(iter(val_loader))
    fixed_images = fixed_images.to(DEVICE)
    fixed_masks = fixed_masks.to(DEVICE)

    # ---------------- CLASS DISTRIBUTION ----------------
    print("Computing class distribution...")

    class_counts = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for images, masks in train_loader:
            masks = masks.view(-1)
            counts = torch.bincount(masks, minlength=NUM_CLASSES)
            class_counts += counts.cpu()

    total_pixels = class_counts.sum()

    for i in range(NUM_CLASSES):
        percentage = (class_counts[i] / total_pixels) * 100
        print(f"Class {i}: {int(class_counts[i].item())} pixels "
              f"({percentage:.2f}%)")

    # compute class weights
    freq = class_counts / total_pixels
    weights = 1.0 / freq
    weights = weights / weights.sum() * NUM_CLASSES
    weights = weights.to(DEVICE)

    print("Class Weights:", weights)

    # ---------------- MODEL ----------------
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)

    # loss and optimizer
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_loss = DiceLoss(NUM_CLASSES)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---------------- TRAINING LOOP ----------------
    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            ce = ce_loss(outputs, masks)
            dice = dice_loss(outputs, masks)

            loss = 0.7 * ce + 0.3 * dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0
        iou_sum = torch.zeros(NUM_CLASSES).to(DEVICE)

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)

                ce = ce_loss(outputs, masks)
                dice = dice_loss(outputs, masks)
                loss = 0.7 * ce + 0.3 * dice

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                ious = compute_iou(preds, masks, NUM_CLASSES)

                iou_sum += ious.to(DEVICE)

        val_loss /= len(val_loader)

        mean_ious = iou_sum / len(val_loader)
        mIoU = mean_ious.mean().item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"mIoU: {mIoU:.4f}")

        # ---- VISUALIZATION (FIXED PATCH) ----
        with torch.no_grad():
            outputs = model(fixed_images)
            preds = torch.argmax(outputs, dim=1)

        images = fixed_images.cpu()
        masks = fixed_masks.cpu()
        preds = preds.cpu()

        idx = 0

        image = images[idx].permute(1, 2, 0).numpy()
        gt_mask = masks[idx].numpy()
        pred_mask = preds[idx].numpy()

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(gt_mask)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred_mask)
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    main()
