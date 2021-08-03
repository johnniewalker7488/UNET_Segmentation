import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import CarvanaDataset


def get_loaders(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True
):
    train_ds = CarvanaDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    
    val_ds = CarvanaDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    
    return train_loader, val_loader


def check_dice(loader, model, device='cuda:0'):
    dice_score = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds * y).sum()) / (preds + y).sum() + 1e-8
    print(f'Validation Dice score: {dice_score/len(loader)}')
    model.train()
    
    
def save_preds_img(loader, model, folder='saved_images/', device='cuda:0'):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f'{folder}/pred_{idx}.png')
        torchvision.utils.save_image(y.unsqueeze(1), f'{folder}{idx}.png')
    model.train()
    
