import os
import cv2
import torch
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from utils.DCHFRDataset import DCHFRDataset
from utils.utils import set_device, data_process, get_current_visuals, save_network, tensor2img, calculate_psnr
from model.DefineNet import define_DCHFRnet

def main(args):
    device = torch.device('cuda')

    # data set
    train_set = DCHFRDataset(args.root, 'train', if_transform=False, dataset_type=args.dataset_type, base_size=args.base_size)
    val_set = DCHFRDataset(args.root, 'val', if_transform=False, dataset_type=args.dataset_type, base_size=args.base_size)
    print("---------------finished loading dataset---------------")
    print('train data number:',len(train_set))
    print('val data number:', len(val_set))

    # data loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_worker)
    print("---------------finished loading dataloader---------------")

    # model
    DCHFRNet = set_device(define_DCHFRnet(args),device)
    DCHFRNet.set_loss(device)
    DCHFRNet.set_new_noise_schedule(device)
    
    DCHFRNet.train()
    optim_params = list(DCHFRNet.parameters())
    DCHFRopt = torch.optim.Adam(optim_params, lr=1e-4)
    scheduler = lr_scheduler.MultiStepLR(DCHFRopt, milestones=[1000], gamma=0.2)
    print("---------------finished loading model---------------")

    # train
    current_step = 0
    print('begin step',current_step)
    current_epoch = 0
    print('begin epoch', current_epoch)
    n_epoch = args.n_epoch
    print("---------------finished setting noise schedule---------------")
    if args.phase == 'train':
        max_psnr = -1e18
        while current_epoch < n_epoch:
            train_loss = 0.0
            scheduler.step()
            for _, train_data in tqdm(enumerate(train_loader), total=len(train_loader)):
                current_step += 1
                feed_data = data_process(train_data, mode='train')
                feed_data = set_device(feed_data,device)
                loss = DCHFRNet(feed_data)
                b, c, h, w = feed_data['hr'].shape
                loss = loss.sum() / int(b * c * h * w)
                DCHFRopt.zero_grad()
                loss.backward()
                DCHFRopt.step()
                train_loss += loss
            wandb.log({'train_epoch': current_epoch, 'train_loss': train_loss/len(train_loader)})
            current_epoch += 1
            if current_epoch < 301:
                val_freq = 1
            else:
                val_freq = 30
            if current_epoch % val_freq == 0:
                print("---------------validation---------------")
                avg_psnr = 0.0
                idx = 0
                if args.dataset_type == "BigReal" or args.dataset_type == "BigSim":
                    mean = [-0.1246]
                    std = [1.0923]
                elif args.dataset_type == "NUAA" or args.dataset_type == "NUDT" or args.dataset_type == "IRSTD":
                    mean = [.485, .456, .406]
                    std = [.229, .224, .225]
                result_hr_path = args.results_hr.rsplit('/', 1)[0] + '/{}/'.format(current_epoch) + args.results_hr.rsplit('/', 1)[1]
                result_sr_path = args.results_sr.rsplit('/', 1)[0] + '/{}/'.format(current_epoch) + args.results_sr.rsplit('/', 1)[1]
                result_lr_path = args.results_lr.rsplit('/', 1)[0] + '/{}/'.format(current_epoch) + args.results_lr.rsplit('/', 1)[1]
                os.makedirs('{}'.format(result_hr_path), exist_ok=True)
                os.makedirs('{}'.format(result_sr_path), exist_ok=True)
                os.makedirs('{}'.format(result_lr_path), exist_ok=True)
                DCHFRNet.set_new_noise_schedule(device)
                for _, val_data in tqdm(enumerate(val_loader), total=len(val_loader)):
                    idx += 1
                    feed_data = data_process(val_data, mode='val')
                    feed_data = set_device(feed_data, device)
                    DCHFRNet.eval()
                    with torch.no_grad():
                        SR_imgs = DCHFRNet.super_resolution(feed_data, continous=False, use_ddim=True)
                        visuals = get_current_visuals(SR_imgs,feed_data)
                        sr_img = tensor2img(visuals['SR'])
                        hr_img = tensor2img(visuals['HR'])
                        avg_psnr += calculate_psnr(sr_img, hr_img)

                        sr_img = visuals['SR'].squeeze().cpu().numpy()
                        for i in range(3):
                            sr_img[i] = sr_img[i] * std[i] + mean[i]
                        sr_img = sr_img * 255
                        sr_img = np.transpose(sr_img, (1, 2, 0))

                        hr_img = visuals['HR'].squeeze().cpu().numpy()
                        for i in range(3):
                            hr_img[i] = hr_img[i] * std[i] + mean[i]
                        hr_img = hr_img * 255
                        hr_img = np.transpose(hr_img, (1, 2, 0))

                        lr_img = visuals['LR'].squeeze().cpu().numpy()
                        for i in range(3):
                            lr_img[i] = lr_img[i] * std[i] + mean[i]
                        lr_img = lr_img * 255
                        lr_img = np.transpose(lr_img, (1, 2, 0))
                        cv2.imwrite('{}/{}_hr.png'.format(result_hr_path, idx), cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY))
                        cv2.imwrite('{}/{}_sr.png'.format(result_sr_path, idx), cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY))
                        cv2.imwrite('{}/{}_lr.png'.format(result_lr_path, idx), cv2.cvtColor(lr_img, cv2.COLOR_RGB2GRAY))
                print("---------------result---------------")
                avg_psnr = avg_psnr / idx 
                wandb.log({'train_epoch': current_epoch, 'psnr': avg_psnr})
                print("psnr:", avg_psnr)
                if avg_psnr >= max_psnr:
                    max_psnr = avg_psnr
                    save_network(args.checkpoint, current_epoch, current_step, DCHFRNet, SRopt, best='psnr_{}'.format(max_psnr))
                    wandb.log({'train_epoch': current_epoch, 'max_psnr': max_psnr})
                    print("---------------finished saving best weights---------------")
                DCHFRNet.train()
                DCHFRNet.set_new_noise_schedule(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/Datasets/', help='choose datasets')
    parser.add_argument('--dataset_type', type=str, default='BigReal', help='BigReal or BigSim or NUAA or NUDT or IRSTD')
    parser.add_argument('--base_size', type=int, default=512, help='base_size')
    parser.add_argument('--crop_size', type=int, default=480, help='crop_size')
    parser.add_argument('--batch_size', type=int, default=8, help='train_batch_size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='val_batch_size')
    parser.add_argument('--num_worker', type=int, default=8, help='num_workers')
    parser.add_argument('--n_epoch', type=int, default=2000, help='iter number')
    parser.add_argument('--phase', type=str, default='train', help='train or val')
    parser.add_argument('--results_hr', type=str, default='results/hr', help='hr result fold')
    parser.add_argument('--results_sr', type=str, default='results/sr', help='sr result fold')
    parser.add_argument('--results_lr', type=str, default='results/lr', help='lr result fold')
    parser.add_argument('--checkpoint', type=str, default='checkpoints', help='checkpoint fold')
    args = parser.parse_args()
    os.environ["WANDB_API_KEY"] = "xxxx"
    wandb.login()
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project='train_DCHFR', config=args.__dict__, name='BigReal' + nowtime, save_code=False)
    num_gpus = torch.cuda.device_count()
    print("Number of available GPUs:", num_gpus)
    main(args)
