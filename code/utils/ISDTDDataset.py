import os
import glob
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter

# NUAA
def NUAA_extract_number(filename):
    parts = filename.split('_')
    number_str = parts[-1].split('.')[0]
    return int(number_str)

def NUAAlabel_extract_number(filename):
    parts = filename.split('_')
    number_str = parts[-2]
    return int(number_str)
# NUDT
def NUDT_extract_number(filename):
   parts = filename.split('/')
   number_str = parts[-1].split('.')[0]
   return int(number_str)

def NUDTlabel_extract_number(filename):
   parts = filename.split('/')
   number_str = parts[-1].split('.')[0]
   return int(number_str)


class ISDTDDataset(Dataset):
    def __init__(self, root, mode, if_transform, base_size, dataset_type, crop_size=480):
        self.mode = mode
        self.if_transform = if_transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.dataset_type = dataset_type

        if self.dataset_type == "BigReal" or self.dataset_type == "BigSim":
            self.imgs_dir = os.path.join(root, 'images')
            self.mask_dir = os.path.join(root, 'masks')
            self.names = []
            if self.mode == 'train':
                self.list_dir = os.path.join(root, 'train.txt')
            elif self.mode == 'val':
                self.list_dir = os.path.join(root, 'test.txt')
            with open(self.list_dir, 'r') as f:
                self.names += [line.strip() for line in f.readlines()]
            self.dataset_len = len(self.names)
        elif self.dataset_type == "NUAA":
            if self.mode == 'train':
                self.imgs = sorted(glob.glob(os.path.join(root, 'train_imgs', "*.png")),key=NUAA_extract_number)
                self.labels = sorted(glob.glob(os.path.join(root, 'train_labels', "*.png")),key=NUAAlabel_extract_number)
            if self.mode == 'val':
                self.imgs = sorted(glob.glob(os.path.join(root, 'test_imgs', "*.png")),key=NUAA_extract_number)
                self.labels = sorted(glob.glob(os.path.join(root, 'test_labels', "*.png")),key=NUAAlabel_extract_number)
            self.dataset_len = len(self.imgs)
        elif self.dataset_type == "NUDT":
            if self.mode == 'train':
                self.imgs = sorted(glob.glob(os.path.join(root, 'train_imgs', "*.png")),key=NUDT_extract_number)
                self.labels = sorted(glob.glob(os.path.join(root, 'train_labels', "*.png")),key=NUDTlabel_extract_number)
            if self.mode == 'val':
                self.imgs = sorted(glob.glob(os.path.join(root, 'test_imgs', "*.png")),key=NUDT_extract_number)
                self.labels = sorted(glob.glob(os.path.join(root, 'test_labels', "*.png")),key=NUDTlabel_extract_number)
            self.dataset_len = len(self.imgs)
        elif self.dataset_type == "IRSTD":
            if self.mode == 'train':
                self.imgs = sorted(glob.glob(os.path.join(root, 'train_imgs', "*.png")))
                self.labels = sorted(glob.glob(os.path.join(root, 'train_labels', "*.png")))
            if self.mode == 'val':
                self.imgs = sorted(glob.glob(os.path.join(root, 'test_imgs', "*.png")))
                self.labels = sorted(glob.glob(os.path.join(root, 'test_labels', "*.png")))
            self.dataset_len = len(self.imgs)
        else:
            raise NotImplementedError( 'dataset_type is wrong!')
        self.transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize([-0.1246], [1.0923])]) # transforms.ToTensor(),

    def __len__(self):
        return self.dataset_len

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
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
        mask = mask.resize((ow, oh), Image.NEAREST)
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask
    
    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, index):
        if self.dataset_type == "BigReal" or self.dataset_type == "BigSim":
            name = self.names[index]
            img_path = self.imgs_dir+'/'+name+'.png'
            label_path = self.mask_dir+'/'+name+'.png'
            img = Image.open(img_path).convert("RGB")
            label = Image.open(label_path).convert("RGB")
            out_name = name+'.png'
        elif self.dataset_type == "NUAA" or self.dataset_type == "NUDT" or self.dataset_type == "IRSTD":
            img = Image.open(self.imgs[index]).convert("RGB")
            label = Image.open(self.labels[index]).convert("RGB")
            out_name = self.imgs[index].split("/")[-1]

    
        if self.if_transform == True:
            if self.mode == 'train':
                img, label = self._sync_transform(img,label)
                img, label = self.transform(img), transforms.ToTensor()(label)    
                if random.random() < 0.5:
                    img = img.flip(-1)
                    label = label.flip(-1)
                return {'img': img, 'label': label}
            
            elif self.mode == 'val':
                width = img.width
                height = img.height
                img, label = self._testval_sync_transform(img,label)
                img, label = self.transform(img), transforms.ToTensor()(label)
                return {'img': img, 'label': label, 'name': out_name, 'width': width, 'height': height}
        
        else:
            if self.mode == 'train':
                img = img.resize((self.base_size, self.base_size))
                label = label.resize((self.base_size, self.base_size))
                img, label = self.transform(img), transforms.ToTensor()(label)
                if random.random() < 0.5:
                    img = img.flip(-1)
                    label = label.flip(-1)
                return {'img': img, 'label': label}
            
            elif self.mode == 'val':
                width = img.width
                height = img.height
                img, label = self.transform(img), transforms.ToTensor()(label)
                return {'img': img, 'label': label, 'name': out_name, 'width': width, 'height': height}




