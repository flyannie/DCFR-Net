import os
import glob
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter

class DCHFRDataset(Dataset):
    def __init__(self, root, mode, if_transform, base_size, dataset_type, crop_size=288):
        self.mode = mode
        self.if_transform = if_transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.dataset_type = dataset_type

        if self.dataset_type == "BigReal" or self.dataset_type == "BigSim":
            self.imgs_dir = os.path.join(root, 'images')
            self.names = []
            if self.mode == 'train':
                self.list_dir = os.path.join(root, 'train.txt')
            elif self.mode == 'val':
                self.list_dir = os.path.join(root, 'test_hr.txt')
            with open(self.list_dir, 'r') as f:
                self.names += [line.strip() for line in f.readlines()]
            self.dataset_len = len(self.names)
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([-0.1246], [1.0923])])
        elif self.dataset_type == "NUAA" or self.dataset_type == "NUDT" or self.dataset_type == "IRSTD":
            if self.mode == 'train':
                self.imgs = sorted(glob.glob(os.path.join(root, 'train_imgs', "*.png")))
            elif self.mode == 'val':
                self.imgs = sorted(glob.glob(os.path.join(root, 'test_imgs', "*.png")))
            self.dataset_len = len(self.imgs)
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]) 
        else:
            raise NotImplementedError( 'dataset_type is wrong!')
        

    def __len__(self):
        return self.dataset_len

    def _sync_transform(self, img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        base_size = self.base_size
        long_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        img_lr = img.resize((base_size/4, base_size/4), Image.BICUBIC)
        img_hr = np.array(img)
        img_lr = np.array(img_lr)
        return img_lr, img_hr


    def _testval_sync_transform(self, img):
        base_size = self.base_size
        img_hr  = img.resize ((base_size, base_size), Image.BILINEAR)
        img_lr = img.resize((64, 64), Image.BICUBIC)
        img_hr = np.array(img_hr)
        img_lr = np.array(img_lr)
        return img_lr, img_hr

    def __getitem__(self, index):
        if self.dataset_type == "BigReal" or self.dataset_type == "BigSim":
            name = self.names[index]
            img_path = self.imgs_dir+'/'+name+'.png'
        elif self.dataset_type == "NUAA" or self.dataset_type == "NUDT" or self.dataset_type == "IRSTD":
            img_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")
        
        if self.if_transform == True:
            if self.mode == 'train':
                img_lr, img_hr = self._sync_transform(img)
                img_lr = self.transform(img_lr)
                img_hr = self.transform(img_hr)
                if random.random() < 0.5:
                    img_lr = img_lr.flip(-1)
                    img_hr = img_hr.flip(-1)
                return {'lr': img_lr, 'hr': img_hr}
            elif self.mode == 'val':
                img_lr, img_hr = self._testval_sync_transform(img)
                img_lr = self.transform(img_lr)
                img_hr = self.transform(img_hr)
                if random.random() < 0.5:
                    img_lr = img_lr.flip(-1)
                    img_hr = img_hr.flip(-1)
                return {'lr': img_lr, 'hr': img_hr}
        else:
            base_size = self.base_size
            lr_base_size = int(base_size/4)
            img_hr  = img.resize ((base_size, base_size), Image.BILINEAR)
            img_lr = img.resize((lr_base_size, lr_base_size), Image.BICUBIC)
            img_hr = self.transform(img_hr)
            img_lr = self.transform(img_lr)
            return {'lr': img_lr, 'hr': img_hr}
            




