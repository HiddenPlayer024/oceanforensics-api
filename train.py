import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
    def forward(self, y_pred, y_true):
        # SMP DiceLoss expects y_pred as logits
        return self.bce(y_pred, y_true) + self.dice(y_pred, y_true)

class OilSpillDataset(Dataset):
    def __init__(self, image_dirs, mask_dirs, transform=None):
        self.transform = transform
        self.samples = []
        for img_dir, msk_dir in zip(image_dirs, mask_dirs):
            for f in os.listdir(img_dir):
                if f.endswith(('.png', '.jpg')):
                    self.samples.append((os.path.join(img_dir, f), os.path.join(msk_dir, f)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # In this dataset masks end in .png but might be same name
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
        else:
            alt_mask_path = mask_path.replace('.jpg', '.png')
            if os.path.exists(alt_mask_path):
                mask = np.array(Image.open(alt_mask_path).convert("L"))
            else:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Binarize mask
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask is (1, H, W)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask

def main():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=1,                      
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # SAR specific augmentations via albumentations
    # MultiplicativeNoise simulates speckle noise typical in SAR
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    image_dirs = ['dataset/images/images/train', 'dataset/images/images/val']
    mask_dirs = ['dataset/masks/masks/train', 'dataset/masks/masks/val']

    # We need separate datasets to apply different transforms
    full_dataset_train = OilSpillDataset(image_dirs, mask_dirs, transform=train_transform)
    full_dataset_val = OilSpillDataset(image_dirs, mask_dirs, transform=val_transform)
    
    # Subset to allow completion on CPU within sandbox timeframe
    subset_size = min(len(full_dataset_train), 15)
    indices = torch.randperm(len(full_dataset_train))[:subset_size]
    
    # Split 80/20
    train_len = int(0.8 * subset_size)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    
    train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset_val, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"Starting optimized fine-tuning for {num_epochs} epochs on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step()
            
        # Validation Loop
        model.eval()
        val_iou = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.long(), mode='binary', threshold=0.5)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                
                val_iou += iou.item() if not torch.isnan(iou) else 0.0
                val_dice += dice.item() if not torch.isnan(dice) else 0.0

        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val mIoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/deeplabv3plus_resnet50_oilspill.pth")
    print("Model saved to models/deeplabv3plus_resnet50_oilspill.pth")

if __name__ == "__main__":
    main()
