import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from dataset_load import Dataload
import utils
from losses import PSNRLoss
from model import QHNet

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    torch.backends.cudnn.benchmark = True

    model = QHNet(base_channels=48, enc_blocks=[4, 4, 8, 8], dec_blocks=[2, 2, 2, 2], mode=args.mode).cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()

    criterion_psnr = PSNRLoss().cuda()
    criterion_ssim = utils.SSIM().cuda()

    dataset_test = Dataload(data_dir=args.test_dir, patch_size=args.patch_size)
    test_loader = DataLoader(dataset=dataset_test, num_workers=1, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    psnr_test_rgb = []
    ssim_test_rgb = []

    with torch.no_grad():
        for data in tqdm(test_loader, unit='img'):
            target = data[1].cuda()
            input_ = data[0].cuda()
            restored = model(input_)

            for res, tar in zip(restored, target):
                psnr_test_rgb.append(utils.torchPSNR(res, tar))
                ssim_test_rgb.append(criterion_ssim(res.unsqueeze(0), tar.unsqueeze(0)).item())

    psnr_test_rgb = torch.stack(psnr_test_rgb).mean().item()
    ssim_test_rgb = torch.tensor(ssim_test_rgb).mean().item()

    print(f"Test PSNR: {psnr_test_rgb:.4f}")
    print(f"Test SSIM: {ssim_test_rgb:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test QHNet model")
    parser.add_argument('--test_dir', type=str, default='./ds/test', help='directory for test data')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--patch_size', type=int, default=64, help='patch size for testing')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size for testing')
    parser.add_argument('--cuda_visible_devices', type=str, default='0', help='CUDA visible devices')
    parser.add_argument('--mode', type=str, default='QHNet', help='model mode')

    args = parser.parse_args()
    main(args)
