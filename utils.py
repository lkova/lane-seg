import torch
import torchvision
from dataset import CarlaLaneDataset
from torch.utils.data import DataLoader
import numpy as np


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarlaLaneDataset(
        image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarlaLaneDataset(
        image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")

    return float(num_correct) / float(num_pixels)


def apply_mask(image_real, predicted_mask):
    n_channels, height, width = predicted_mask.shape
    predicted_mask = np.reshape(predicted_mask, (height, width, n_channels))
    image_real = image_real.resize((width, height))
    image_real = np.array(image_real)
    predicted_mask = np.squeeze(predicted_mask * 255.0, axis=2)
    predicted_mask_rgb = np.zeros((*predicted_mask.shape, 3))
    predicted_mask_rgb[:, :, 1] = predicted_mask
    image_with_mask = image_real + predicted_mask_rgb
    image_with_mask[image_with_mask > 255.0] = 0.0

    return image_with_mask


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
