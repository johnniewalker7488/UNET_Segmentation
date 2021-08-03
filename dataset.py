import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super(CarvanaDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        return image, mask
    

def prepare_datasets(data_dir):
    car_ids_1 = []
    paths = []
    for dirname, _, filenames in os.walk('./dataset/train'):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            paths.append(path)
            car_id = filename.split('.')[0]
            car_ids_1.append(car_id)
    d = {'id': car_ids_1, 'car_path': paths}
    df = pd.DataFrame(data=d)
    df = df.set_index('id')
    
    car_ids_2 = []
    mask_paths = []
    for dirname, _, filenames in os.walk('./dataset/train_masks'):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            mask_paths.append(path)
            car_id = filename.split('.')[0]
            car_id = car_id.split('_mask')[0]
            car_ids_2.append(car_id)
    d = {'id': car_ids_2, 'mask_path': mask_paths}
    mask_df = pd.DataFrame(data=d)
    mask_df = mask_df.set_index('id')
    
    df['mask_path'] = mask_df['mask_path']
    
    _, valid_df = train_test_split(df, random_state=1, test_size=.2)
    
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    
    val_masks_dir = os.path.join(data_dir, 'val_masks')
    if not os.path.exists(val_masks_dir):
        os.mkdir(val_masks_dir)
    
    valid_df = valid_df.reset_index()
    valid_masks_paths = list(valid_df['mask_path'])
    valid_imgs_paths = list(valid_df['car_path'])
    
    for path in valid_imgs_paths:
        src = path
        dst = os.path.join(data_dir, 'val')
        shutil.move(src, dst)
    print(f'{len(valid_imgs_paths)} images moved from {src} to {dst}')

    for path in valid_masks_paths:
        src = path
        dst = os.path.join(data_dir, 'val_masks')
        shutil.move(src, dst)
    print(f'{len(valid_masks_paths)} images moved from {src} to {dst}')