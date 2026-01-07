"""
Phase 2: SIMPLIFIED Dual Self-Supervision
Focus on what works: Consistency + Weak Supervision
Remove complex losses that aren't helping
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--weak_ratio', type=float, default=0.5)
args = parser.parse_args()

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = args.epochs
LR = 3e-5
BATCH_SIZE = args.batch_size
WEAK_RATIO = args.weak_ratio

# SIMPLIFIED: Only two losses that actually work
LOSS_WEIGHTS = {
    "consistency": 0.5,
    "supervised": 0.5
}

PREPROCESSED_PATH = "./preprocessed_mpiigaze"

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

scaler = GradScaler()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Simple augmentations
class WeakAug:
    def __init__(self):
        self.t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ])
    def __call__(self, img):
        return self.t(img)

class StrongAug:
    def __init__(self):
        self.t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    def __call__(self, img):
        return self.t(img)

def compute_angular_error(pred, target):
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
    cos_sim = torch.clamp(torch.sum(pred_norm * target_norm, dim=1), -1.0, 1.0)
    angle_rad = torch.acos(cos_sim)
    return torch.rad2deg(angle_rad).mean()

class MPIIGazeDataset(Dataset):
    def __init__(self, data, dual_views=False):
        self.data = data
        self.dual_views = dual_views
        
        if dual_views:
            self.weak_aug = WeakAug()
            self.strong_aug = StrongAug()
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.data)
    
    def _process_image(self, image):
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        
        return cv2.resize(image, (224, 224))

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = self._process_image(sample['image'])
        gaze = torch.tensor(sample['gaze'], dtype=torch.float32)
        
        if self.dual_views:
            view_weak = self.weak_aug(image)
            view_strong = self.strong_aug(image)
            return view_weak, view_strong, gaze
        else:
            image = self.transform(image)
            return image, gaze

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train_epoch(model, train_loader, optimizer, scaler, epoch):
    model.train()
    
    total_consist = 0.0
    total_sup = 0.0
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in pbar:
        if batch is None:
            continue
        
        view_weak, view_strong, gaze_gt = batch
        view_weak = view_weak.to(DEVICE, non_blocking=True)
        view_strong = view_strong.to(DEVICE, non_blocking=True)
        gaze_gt = gaze_gt.to(DEVICE, non_blocking=True)
        
        batch_size = view_weak.shape[0]
        
        # Sample labeled indices
        num_labeled = max(1, int(batch_size * WEAK_RATIO))
        labeled_idx = torch.randperm(batch_size)[:num_labeled]
        
        with autocast():
            # Predict from both views
            gaze_weak, _ = model(view_weak)
            gaze_strong, _ = model(view_strong)
            
            # Average predictions
            gaze_avg = (gaze_weak + gaze_strong) / 2.0
            
            # Consistency loss
            consistency_loss = F.l1_loss(gaze_weak, gaze_strong)
            
            # Supervised loss (on labeled samples)
            supervised_loss = F.l1_loss(gaze_avg[labeled_idx], gaze_gt[labeled_idx])
            
            # Combined
            loss = (
                LOSS_WEIGHTS['consistency'] * consistency_loss +
                LOSS_WEIGHTS['supervised'] * supervised_loss
            )
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Track
        total_consist += consistency_loss.item() * batch_size
        total_sup += supervised_loss.item() * batch_size
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({
            'Loss': f"{total_loss/total_samples:.4f}",
            'Consist': f"{total_consist/total_samples:.4f}",
            'Sup': f"{total_sup/total_samples:.4f}"
        })
    
    return {
        'total': total_loss / total_samples,
        'consist': total_consist / total_samples,
        'sup': total_sup / total_samples
    }

@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    total_angular = 0.0
    total_samples = 0
    
    for batch in val_loader:
        if batch is None:
            continue
        
        x, y = batch
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        
        with autocast():
            gaze_pred, _ = model(x)
        
        angular = compute_angular_error(gaze_pred, y).item()
        total_angular += angular * x.shape[0]
        total_samples += x.shape[0]
    
    return total_angular / total_samples

def main():
    from rgb_mem_pretrain import RGB_DINO_Gaze
    
    # Seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("Loading data...")
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    
    # Normalize gaze
    print("Normalizing gaze vectors...")
    for split in ['train_data', 'val_data', 'test_data']:
        if split in data:
            for sample in data[split]:
                gaze = np.array(sample['gaze'])
                norm = np.linalg.norm(gaze)
                if norm > 1e-8:
                    sample['gaze'] = (gaze / norm).tolist()
    
    train_dataset = MPIIGazeDataset(data['train_data'], dual_views=True)
    val_dataset = MPIIGazeDataset(data['val_data'], dual_views=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=6, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn
    )
    
    print("Initializing model...")
    model = RGB_DINO_Gaze(unsupervised_pretrain=False).to(DEVICE)
    
    # Load RGB-MEM encoder if available
    if os.path.exists("./models/pretrained_rgb_mem/rgb_mem_final.pth"):
        weights = torch.load("./models/pretrained_rgb_mem/rgb_mem_final.pth", map_location=DEVICE)
        encoder_weights = {k: v for k, v in weights.items() if 'gaze_head' not in k and 'mem_head' not in k}
        model.load_state_dict(encoder_weights, strict=False)
        print("✓ Loaded RGB-MEM encoder")
    
    # Reinitialize gaze head with small weights
    for m in model.gaze_head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Freeze early layers
    for param in model.dinov2.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.dinov2.encoder.layer):
        if i < 6:
            for param in layer.parameters():
                param.requires_grad = False
    
    print("✓ Frozen: embeddings + layers 0-5")
    print("✓ Training: layers 6-11 + gaze_head")
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.gaze_head.parameters(), 'lr': LR * 10},
        {'params': [p for n, p in model.named_parameters() 
                   if 'gaze_head' not in n and p.requires_grad], 'lr': LR}
    ], weight_decay=1e-4)
    
    # Warmup + Cosine schedule
    def lr_lambda(epoch):
        warmup = 2
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (EPOCHS - warmup)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\n{'='*70}")
    print("SIMPLIFIED DUAL SELF-SUPERVISION")
    print(f"{'='*70}")
    print(f"Consistency: {LOSS_WEIGHTS['consistency']} | Supervised: {LOSS_WEIGHTS['supervised']}")
    print(f"Weak Ratio: {WEAK_RATIO} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"{'='*70}\n")
    
    best_angular = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        losses = train_epoch(model, train_loader, optimizer, scaler, epoch)
        val_angular = validate(model, val_loader)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train: Loss={losses['total']:.4f}, Consist={losses['consist']:.4f}, Sup={losses['sup']:.4f}")
        print(f"  Val Angular: {val_angular:.2f}°")
        
        if val_angular < best_angular:
            best_angular = val_angular
            patience_counter = 0
            torch.save(model.state_dict(), "./models/rgb_dino_gaze_simple.pth")
            print(f"  ✓ Best model saved ({val_angular:.2f}°)")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("  ⚠ Early stopping")
                break
    
    print(f"\n{'='*70}")
    print(f"Training Complete! Best: {best_angular:.2f}°")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()