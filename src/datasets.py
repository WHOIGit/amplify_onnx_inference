import os.path
import random
from typing import Union

import numpy as np
from PIL import Image

import ifcb


class MyDataLoader:
    def __init__(self, dataset, batch_size:int=1, transform:callable=None, shuffle:bool=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            imgs = [self.dataset[i] for i in batch_indices]
            pids = [self.dataset.roi_pids[i] for i in batch_indices]
            if self.transform:
                imgs = self.transform(imgs)
            yield imgs,pids


class IfcbBinDataset:
    def __init__(self, bin_accessor:str):
        if bin_accessor.startswith('http'):
            opener = ifcb.open_url
        else:
            opener = ifcb.open_raw
        with opener(bin_accessor) as ifcbbin:
            self.pid = ifcbbin.pid.pid
            self.roi_data = list(ifcbbin.images.values())
            self.roi_pids = [ifcbbin.pid.with_target(idx) for idx in ifcbbin.images.keys()]

    def __len__(self):
        return len(self.roi_pids)

    def __getitem__(self, idx):
        return self.roi_data[idx]

    def get_pid(self, idx):
        return self.roi_pids[idx]


class IfcbImagesDataset:
    def __init__(self, images:list[str]):
        self.image_acccessors = images

    def __len__(self):
        return len(self.image_acccessors)

    def __getitem__(self, idx):
        image_acccessor = self.image_acccessors[idx]
        if os.path.isfile(image_acccessor):
            with Image.open(image_acccessor) as img:
                img = img.convert("L")
                img = np.array(img)
        elif image_acccessor.startswith('http'):
            raise NotImplemented
        else:
            raise NotImplemented
        return img


class IfcbBinImageTransformer:
    def __init__(self, resize:Union[int,tuple], normalize:dict=None, dtype=np.float32):
        self.resize = (resize,resize) if isinstance(resize,int) else resize
        if normalize:
            assert 'mean' in normalize
            assert 'std' in normalize
            if isinstance(normalize['mean'],float):
                normalize['mean'] = (normalize['mean'],normalize['mean'],normalize['mean'])
            if isinstance(normalize['std'],float):
                normalize['std'] = (normalize['std'],normalize['std'],normalize['std'])
            assert len(normalize['mean']) == normalize['std'] == 3
        self.normalize = normalize
        self.dtype = dtype

    def transform_bin_image(self, img:np.ndarray):
        img = Image.fromarray(img, mode='L')
        img = img.resize(self.resize)
        img = img.convert('RGB') # from W,H to W,H,C where C is 3
        img = np.array(img) #  image is back to a np array now
        img = np.transpose(img, (2,0,1))  # from W,H,3 to 3,W,H
        img = img.astype(self.dtype) / 255.0  # scale uint8 to float
        if self.normalize:
            mean = np.array(self.normalize['mean']).reshape(3, 1, 1)
            std = np.array(self.normalize['std']).reshape(3, 1, 1)
            img = (img - mean) / std
        return img

    def __call__(self, batch:list):
        batch = [self.transform_bin_image(img) for img in batch]  # each is (3, H, W)
        batch = np.stack(batch, axis=0)  # shape: (N, 3, H, W)
        return batch

