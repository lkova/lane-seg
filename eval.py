from numpy.core.fromnumeric import shape
import torch
from model import UNET
from utils import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMSIZE = 128
loader = transforms.Compose([transforms.Scale(IMSIZE), transforms.ToTensor()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test for lane segmentation")
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help='Path to the image that needs to be tested. (default: "")',
    )
    parser.add_argument(
        "--save_img",
        type=str,
        default="/saved_images/eval_img.png",
        help='Path to the image that needs to be tested. (default: "")',
    )

    args = parser.parse_args()
    if args.image_path == "":
        raise ("no image path specified")

    img_path = args.image_path
    image_real = Image.open(img_path)
    image = loader(image_real).float()
    image = image.unsqueeze(0)
    if DEVICE == "cuda":
        image = image.cuda()

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    model.eval()
    output = model(image)
    probabilities = torch.sigmoid(output.squeeze(1))
    predicted_mask = (probabilities >= 0.5).float() * 1

    if DEVICE == "cuda":
        predicted_mask = predicted_mask.cpu().numpy()
    else:
        predicted_mask = predicted_mask.numpy()

    image_with_mask = apply_mask(image_real=image_real, predicted_mask=predicted_mask)
    image_with_mask = Image.fromarray((image_with_mask).astype(np.uint8)).convert("RGB")

    if args.save_img != "":
        image_with_mask.save(args.save_img)

    plt.imshow(image_with_mask)
    plt.show()
