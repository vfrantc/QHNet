import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from model import QHNet
import utils


def process_image(image, model, patch_size):
    w, h = image.size
    processed_image = Image.new('RGB', (w, h))
    transform = transforms.ToTensor()

    for i in range(0, w, patch_size):
        for j in range(0, h, patch_size):
            patch = image.crop((i, j, i + patch_size, j + patch_size))
            patch_tensor = transform(patch).unsqueeze(0).cuda()
            with torch.no_grad():
                restored_patch_tensor = model(patch_tensor)
            restored_patch = transforms.ToPILImage()(restored_patch_tensor.squeeze(0).cpu())
            processed_image.paste(restored_patch, (i, j))

    return processed_image


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    torch.backends.cudnn.benchmark = True

    model = QHNet(base_channels=48, enc_blocks=[4, 4, 8, 8], dec_blocks=[2, 2, 2, 2]).cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    if not os.path.exists(args.dst_folder):
        os.makedirs(args.dst_folder)

    for filename in os.listdir(args.source_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(args.source_folder, filename)
            image = Image.open(image_path).convert('RGB')
            processed_image = process_image(image, model, args.patch_size)
            save_path = os.path.join(args.dst_folder, os.path.splitext(filename)[0] + '.png')
            processed_image.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo QHNet model")
    parser.add_argument('--source_folder', type=str, required=True, help='folder with source images')
    parser.add_argument('--dst_folder', type=str, required=True, help='folder to save processed images')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--patch_size', type=int, default=64, help='patch size for processing')
    parser.add_argument('--cuda_visible_devices', type=str, default='0', help='CUDA visible devices')

    args = parser.parse_args()
    main(args)
