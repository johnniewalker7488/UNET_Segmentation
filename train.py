import torch
from tqdm import tqdm


def train_fn(loader, model, optimizer, loss_fn, scaler, device, checkpoint_path=None):
    
    loop = tqdm(enumerate(loader), total=len(loader))
    
#     if LOAD_MODEL:
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
    
    for batch_idx, (data, targets) in loop:
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)
        
        with torch.cuda.amp.autocast():
            preds = model(data)
            loss = loss_fn(preds, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loop.set_postfix(loss=loss.item())
    
    