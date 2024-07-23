import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataset_load import Dataload
import utils
from warmup_scheduler import GradualWarmupScheduler
from losses import PSNRLoss
from model import QHNet

torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



def train(args):
    start_epoch = 1
    utils.mkdir(args.model_dir)

    model = QHNet(base_channels=48, enc_blocks=[4, 4, 8, 8], dec_blocks=[2, 2, 2, 2]).cuda()

    ######### Model ###########
    model_restoration = model
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    optimizer = optim.AdamW(model_restoration.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs - args.warmup_epochs,
                                                            eta_min=args.min_lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs,
                                       after_scheduler=scheduler_cosine)

    ######### Pretrain ###########
    if args.pretrain:
        utils.load_checkpoint(model_restoration, args.model_pre_dir)
        print('------------------------------------------------------------------------------')
        print("==> Retrain Training with: " + args.model_pre_dir)
        print('------------------------------------------------------------------------------')

    ######### Resume ###########
    if args.resume:
        path_chk_rest = utils.get_last_path(args.model_pre_dir, '_last.pth')
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    ######### Loss ###########
    criterion_ssim = utils.SSIM().cuda()
    criterion_psnr = PSNRLoss().cuda()

    ######### DataLoaders ###########
    dataset_train = Dataload(data_dir=args.train_dir, patch_size=args.patch_size_train)
    train_loader = DataLoader(dataset=dataset_train, num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=True, drop_last=False, pin_memory=True)

    dataset_val = Dataload(data_dir=args.val_dir, patch_size=args.patch_size_test)
    val_loader = DataLoader(dataset=dataset_val, num_workers=1, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, args.num_epochs + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0
    writer = SummaryWriter(args.model_dir)
    iter = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1
        train_psnr_val_rgb = []
        scaled_loss = 0
        model_restoration.train()

        for i, data in enumerate(tqdm(train_loader, unit='img'), 0):
            for param in model_restoration.parameters():
                param.grad = None
            target_ = data[1].cuda()
            input_ = data[0].cuda()
            restored = model_restoration(input_)
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    ssim = criterion_ssim(restored, target_)
                    loss = 1 - ssim
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                ssim = criterion_ssim(restored, target_)
                psnr = criterion_psnr(restored, target_)
                loss = 1 - ssim
                loss.backward()
                scaled_loss += loss.item()
                optimizer.step()
            torch.cuda.synchronize()
            epoch_loss += loss.item()
            iter += 1
            for res, tar in zip(restored, target_):
                train_psnr_val_rgb.append(utils.torchPSNR(res, tar))
            psnr_train = torch.stack(train_psnr_val_rgb).mean().item()

            writer.add_scalar('loss/iter_loss', loss.item(), iter)
            writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)
            writer.add_scalar('lr/epoch_loss', scheduler.get_lr()[0], epoch)

        #### Evaluation ####
        if epoch % args.val_epochs == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate(tqdm(val_loader, unit='img'), 0):
                target = data_val[1].cuda()
                input_ = data_val[0].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            writer.add_scalar('val/psnr', psnr_val_rgb, epoch)
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(model_restoration.state_dict(), os.path.join(args.model_dir, "model_best.pth"))

            print("[epoch %d Training PSNR: %.4f --- best_epoch %d Test_PSNR %.4f]" % (
            epoch, psnr_train, best_epoch, best_psnr))

        if epoch % 50 == 0:
            torch.save(
                {'epoch': epoch, 'state_dict': model_restoration.state_dict(), 'optimizer': optimizer.state_dict()},
                os.path.join(args.model_dir, f"model_epoch_{epoch}.pth"))

        torch.save({'epoch': epoch, 'state_dict': model_restoration.state_dict(), 'optimizer': optimizer.state_dict()},
                   os.path.join(args.model_dir, "model_last.pth"))

        scheduler.step()
        print("-" * 150)
        print(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tTrain_PSNR: {:.4f}\tSSIM: {:.4f}\tLearningRate {:.8f}\tTest_PSNR: {:.4f}".format(
                epoch, time.time() - epoch_start_time, loss.item(), psnr_train, ssim, scheduler.get_lr()[0], best_psnr))
        print("-" * 150)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QHNet model")
    parser.add_argument('--train_dir', type=str, default='./ds/train', help='directory for training data')
    parser.add_argument('--val_dir', type=str, default='./ds/test', help='directory for validation data')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints/',
                        help='directory to save the model checkpoints')
    parser.add_argument('--pretrain_weights', type=str, default='./checkpoints/model_best.pth',
                        help='path to pre-trained weights')
    parser.add_argument('--mode', type=str, default='QHNet', help='model mode')
    parser.add_argument('--session', type=str, default='mydataset', help='training session identifier')
    parser.add_argument('--patch_size_train', type=int, default=64, help='patch size for training')
    parser.add_argument('--patch_size_test', type=int, default=64, help='patch size for testing')
    parser.add_argument('--num_epochs', type=int, default=250, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size for training')
    parser.add_argument('--val_epochs', type=int, default=1, help='frequency of validation epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='number of warmup epochs')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='gradient clipping value')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading')
    parser.add_argument('--resume', action='store_true', help='resume training from last checkpoint')
    parser.add_argument('--pretrain', action='store_true', help='use pre-trained weights')
    parser.add_argument('--model_pre_dir', type=str, default='./weights/ESDNet.pth',
                        help='directory of pre-trained model')

    args = parser.parse_args()

    args.model_dir = os.path.join(args.model_save_dir, args.mode, 'models', args.session)
    train(args)
