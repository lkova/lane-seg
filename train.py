from numpy.core.fromnumeric import shape
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import *
import argparse
from torch.utils.tensorboard import SummaryWriter

# Default hyperparameters.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 10
IMAGE_HEIGHT = 128  # 512 originally
IMAGE_WIDTH = 256  # 1024 originally
PIN_MEMORY = True

TRAIN_IMG_DIR = "train"
TRAIN_MASK_DIR = "train_label"
VAL_IMG_DIR = "val"
VAL_MASK_DIR = "val_label"

"""
To do:
1. Completely switch to parsing
2. Switch to training the network normally
3. Add Tensorboard
4. Upload to git
5. Write Readme
"""

step = 0


def train(loader, model, optimizer, loss_fn, writer):
    model.train()
    loop = tqdm(loader)
    global step
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        predictions = model(data)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Training Loss", loss, global_step=step)
        step += 1
        loop.set_postfix(loss=loss.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train for lane segmentation")
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Spcifies learing rate for optimizer. (default: 1e-3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set resumes training from provided checkpoint. (default: None)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs. (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for data loaders. (default: 32)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers for data loader. (default: 8)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help='Path to the dataset. (default: "")',
    )
    args = parser.parse_args()

    if args.dataset_path == "":
        raise ("Path to the dataset is missing")

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    root_dir = args.dataset_path
    train_img_dir = root_dir + TRAIN_IMG_DIR
    train_mask_dir = root_dir + TRAIN_MASK_DIR
    val_img_dir = root_dir + VAL_IMG_DIR
    val_mask_dir = root_dir + VAL_MASK_DIR

    train_loader, val_loader = get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        args.batch_size,
        train_transform,
        val_transforms,
        args.num_workers,
        PIN_MEMORY,
    )

    if args.resume:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, writer)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        val_accuracy = check_accuracy(val_loader, model, device=DEVICE)
        writer.add_scalar("Validation Accuracy", val_accuracy, global_step=step)
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
