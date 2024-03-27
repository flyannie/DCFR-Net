import os
import torch
import wandb
import argparse
from tqdm import tqdm
import os.path as ops
from datetime import datetime
from utils.ISDTDDataset import ISDTDDataset
from torch.utils.data import DataLoader
from model.DefineNet import define_trained_DCHFRnet, define_ISDTDnet
from utils.utils import set_device
from utils.BCEDiceloss import BCEDiceloss
from torch.optim import lr_scheduler
from utils.Imetrics import SigmoidMetric, SamplewiseSigmoidMetric
from utils.Pmetric import ROCMetric, PD_FA

def main(args):
    device = torch.device('cuda')

    train_set = ISDTDDataset(args.root, 'train', if_transform=True, base_size=args.base_size, dataset_type=args.dataset_type)
    val_set = ISDTDDataset(args.root, 'val', if_transform=True, base_size=args.base_size, dataset_type=args.dataset_type)
    print("---------------finished loading dataset---------------")
    print('train data number:',len(train_set))
    print('val data number:', len(val_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_worker)
    print("---------------finished loading dataloader---------------")

    DCHFRnet = set_device(define_trained_DCHFRnet(args), device)
    DCHFR_checkpoint_path = args.checkpoint_path
    DCHFR_checkpoint = torch.load(DCHFR_checkpoint_path)
    DCHFRnet.load_state_dict(DCHFR_checkpoint, strict=False)
    for param in DCHFRnet.parameters():
        param.requires_grad = False
    print("---------------finished loading DCHFR checkpoint---------------")

    net = set_device(define_ISDTDnet(args), device)
    loss_func = BCEDiceloss().to(device)
    eval_iou = SigmoidMetric()
    eval_niou = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    eval_ROC = ROCMetric(1, 10)
    eval_PD_FA = PD_FA(1, 10)
    net.train()
    optim_params = list(net.parameters())
    opt = torch.optim.Adam(optim_params, lr=1e-4)
    scheduler = lr_scheduler.MultiStepLR(opt, milestones=[50,100,150], gamma=0.5)
    print("---------------finished loading ISDTD model---------------")

    current_epoch = 0
    n_epoch = args.n_epoch
    best_iou = 0
    best_niou = 0
    current_step = 0
    if args.phase == 'train':
        while current_epoch < n_epoch:
            train_loss = 0.0
            tbar = tqdm(train_loader)
            for _, train_data in enumerate(tbar):
                train_data = set_device(train_data, device)
                img = train_data['img']
                feat = DCHFRnet.encoder(img, img.shape[2:])
                pred = net(feat,img)
                label = train_data['label']
                label = label[:, 0:1, :, :]/255
                loss_ISDTD, bce, dice = loss_func(pred,label)
                current_step += 1
                opt.zero_grad()
                loss_ISDTD.backward()
                opt.step()
                scheduler.step()
                train_loss += loss_ISDTD
                tbar.set_description('Epoch %d, train loss %.4f, bce %.4f, dice %.4f' % (current_epoch, loss_ISDTD, bce, dice))
            wandb.log({'train_epoch': current_epoch, 'train_loss': train_loss/len(train_loader)})
            current_epoch += 1
            if current_epoch % args.val_freq == 0:
                print("---------------validation---------------")
                val_tbar = tqdm(val_loader)
                eval_iou.reset()
                eval_niou.reset()
                eval_PD_FA.reset()
                eval_ROC.reset()
                eval_loss = 0.0
                net.eval()
                for _, val_data in enumerate(val_tbar):
                    val_img = val_data['img'] 
                    val_img = set_device(val_img, device)
                    with torch.no_grad():
                        val_feat = DCHFRnet.encoder(val_img, val_img.shape[2:])
                        val_pred = net(val_feat,val_img)
                        val_label = val_data['label']
                        val_label = set_device(val_label, device)
                        val_label = val_label[:, 0:1, :, :]/255
                        val_pred = val_pred
                        val_loss_ISDTD, val_bce, val_dice = loss_func(val_pred, val_label)
                        eval_loss += val_loss_ISDTD
                        val_label = val_label.cpu()
                        val_pred = val_pred.cpu()
                        eval_iou.update(val_pred, val_label)
                        eval_niou.update(val_pred, val_label)
                        eval_ROC.update(val_pred, val_label)
                        eval_PD_FA.update(val_pred, val_label)
                    val_tbar.set_description('Epoch %d, val loss %.4f, bce %.4f, dice %.4f' % (current_epoch, val_loss_ISDTD, val_bce, val_dice))
                FA, PD = eval_PD_FA.get(len(val_loader))
                _, IoU = eval_iou.get()
                _, nIoU = eval_niou.get()
                _, _, _, _, F1_score = eval_ROC.get()
                FA = FA[5] * 1000000
                PD = PD[5] * 100
                F1_score = F1_score[5]
                wandb.log({'epoch': current_epoch, "ioU": IoU, "nioU": nIoU, 'test_loss': eval_loss/len(val_loader), "PD": PD, "FA": FA, "F1_score": F1_score})
                if IoU > best_iou:
                    best_iou = IoU
                    if IoU > 7.0:
                        pkl_name = 'best-Epoch-%3d_IoU-%.4f_nIoU-%.4f.pkl' % (current_epoch, best_iou, nIoU)
                        torch.save(net.state_dict(), ops.join(args.save_path, pkl_name))
                if nIoU > best_niou:
                    best_niou = nIoU
                wandb.log({'epoch': current_epoch, "best_ioU": best_iou, "best_nioU": best_niou})
                net.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/Datasets/', help='choose datasets')
    parser.add_argument('--dataset_type', type=str, default='BigReal', help='BigReal or BigSim or NUAA or NUDT or IRSTD')
    parser.add_argument('--checkpoint_path', type=str, default='/xxxx_gen.pth', help='DCHFR checkpoint path')
    parser.add_argument('--ISDTD_checkpoint_path', type=str, default='.pth', help='trained ISDTD checkpoint path')
    parser.add_argument('--base_size', type=int, default=512, help='img_size')
    parser.add_argument('--batch_size', type=int, default=4, help='train_batch_size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='val_batch_size')
    parser.add_argument('--num_worker', type=int, default=8, help='num_workers')
    parser.add_argument('--n_epoch', type=int, default=200, help='iter number')
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--val_freq', type=int, default=1, help='validation frequent')
    parser.add_argument('--results_mask', type=str, default='results', help='mask result fold')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='checkpoint fold')
    args = parser.parse_args()
    os.environ["WANDB_API_KEY"] = "xxxx"
    wandb.login()
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project='ISDTD', config=args.__dict__, name='BigReal' + nowtime, save_code=False)

    main(args)