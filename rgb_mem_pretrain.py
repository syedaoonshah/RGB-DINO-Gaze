import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from transformers import AutoModel

# --------------------------
# Global Config
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
PATCH_SIZE = 14
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) **2  # 256 total patches
MASK_RATIO = 0.30  # Target 30% total patches (77 patches)
BATCH_SIZE = 8
EPOCHS = 5
LR = 4e-5
WEIGHT_DECAY = 1e-5
PREPROCESSED_PATH = "./preprocessed_mpiigaze"

# Mixed Precision Scaler
scaler = GradScaler()

# RGB normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --------------------------
# 1. RGB-MEM Mask Generator
# --------------------------
class RGBMEMMaskGenerator:
    def __init__(self):
        self.patch_size = PATCH_SIZE
        self.image_size = IMAGE_SIZE
        self.num_patches = NUM_PATCHES
        self.mask_ratio = MASK_RATIO

    def generate_eye_region_mask(self):
        patch_grid = np.arange(self.num_patches).reshape(16, 16)
        
        # Pupil (center 5x5 patches)
        pupil_patches = patch_grid[5:11, 5:11].flatten()
        
        # Eyelid (perimeter 2 patches)
        eyelid_patches = np.concatenate([
            patch_grid[0:2, :].flatten(),
            patch_grid[14:16, :].flatten(),
            patch_grid[:, 0:2].flatten(),
            patch_grid[:, 14:16].flatten()
        ])
        
        # Combine keep patches
        keep_patches = np.unique(np.concatenate([pupil_patches, eyelid_patches]))
        non_keep_patches = np.setdiff1d(np.arange(self.num_patches), keep_patches)
        
        # Target 30% of total patches
        num_mask_total = int(self.mask_ratio * self.num_patches)
        num_mask = min(num_mask_total, len(non_keep_patches))
        
        mask_patches = np.random.choice(non_keep_patches, num_mask, replace=False)
        
        # Create boolean mask (True = keep, False = mask)
        mask = torch.ones(self.num_patches, dtype=torch.bool)
        mask[mask_patches] = False
        return mask

    def filter_illumination_artifacts(self, rgb_image, mask):
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(rgb_image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(rgb_image.device)
        rgb_denorm = rgb_image * std + mean
        rgb_denorm = torch.clamp(rgb_denorm, 0, 1)
        
        # Convert to numpy
        rgb_np = (rgb_denorm.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
        brightness = hsv[..., 2]
        
        # Compute brightness per patch
        patch_brightness = []
        for i in range(16):
            for j in range(16):
                y1, y2 = i*self.patch_size, (i+1)*self.patch_size
                x1, x2 = j*self.patch_size, (j+1)*self.patch_size
                patch_bright = np.mean(brightness[y1:y2, x1:x2])
                patch_brightness.append(patch_bright)
        
        # Unmask bright patches
        bright_patches = np.where(np.array(patch_brightness) > 200)[0]
        mask[bright_patches] = True
        
        return mask

# --------------------------
# 2. Dataset Class
# --------------------------
class MPIIGazeDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image']
        gaze = sample['gaze']
        
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        
        eye_crop = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
        if self.transform:
            eye_crop = self.transform(eye_crop)
        
        gaze_tensor = torch.tensor(gaze, dtype=torch.float32)
        return eye_crop, gaze_tensor

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --------------------------
# 3. RGB-DINO-Gaze Model (FIXED WITH INPUT MASKING)
# --------------------------
class RGB_DINO_Gaze(nn.Module):
    def __init__(self, unsupervised_pretrain=True):
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-base")
        self.unsupervised_pretrain = unsupervised_pretrain
        
        # Freeze first 4 layers
        for i, layer in enumerate(self.dinov2.encoder.layer):
            if i < 4:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # SIMPLE RGB-MEM Reconstruction Head
        self.mem_head = nn.Sequential(
            nn.Linear(self.dinov2.config.hidden_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 3 * PATCH_SIZE * PATCH_SIZE)
        )
        
        # Initialize properly
        for m in self.mem_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
        
        # Gaze Regression Head
        self.gaze_head = nn.Sequential(
            nn.Linear(self.dinov2.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )
        if unsupervised_pretrain:
            for param in self.gaze_head.parameters():
                param.requires_grad = False

    def forward(self, x, masks=None):
        # CRITICAL FIX: Mask input BEFORE DINOv2
        if self.unsupervised_pretrain and masks is not None:
            batch_size = x.shape[0]
            
            # Create masked input (replace masked patches with zeros)
            x_masked = x.clone()
            for b in range(batch_size):
                mask_b = masks[b]  # (256,) boolean tensor
                
                # Convert patch mask to pixel mask
                for patch_idx in range(NUM_PATCHES):
                    if not mask_b[patch_idx]:  # If this patch is masked
                        row = patch_idx // 16
                        col = patch_idx % 16
                        y1, y2 = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
                        x1, x2 = col * PATCH_SIZE, (col + 1) * PATCH_SIZE
                        # Replace with zeros (mask token)
                        x_masked[b, :, y1:y2, x1:x2] = 0.0
            
            # Pass MASKED input to DINOv2
            outputs = self.dinov2(pixel_values=x_masked, output_hidden_states=True)
            patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Skip [CLS]
            
            # Reconstruct masked patches
            recon_list = []
            for b in range(batch_size):
                mask_b = masks[b]
                num_masked = (~mask_b).sum().item()
                masked_emb = patch_embeddings[b][~mask_b]
                recon = self.mem_head(masked_emb)
                recon_list.append(recon)
            
            return recon_list
        else:
            # Supervised mode: use full image
            outputs = self.dinov2(pixel_values=x, output_hidden_states=True)
            patch_embeddings = outputs.last_hidden_state[:, 1:, :]
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            gaze_pred = self.gaze_head(cls_embedding)
            return gaze_pred, patch_embeddings

# --------------------------
# 4. RGB-MEM Loss Function
# --------------------------
def rgb_mem_recon_loss(recon_list, original_patches, masks):
    total_loss = 0.0
    batch_size = original_patches.shape[0]
    
    for b in range(batch_size):
        mask_b = masks[b]
        original_masked = original_patches[b][~mask_b]
        recon = recon_list[b]
        loss = F.mse_loss(recon, original_masked)
        total_loss += loss
    
    return total_loss / batch_size

# --------------------------
# 5. Training Loop
# --------------------------
def pretrain_rgb_mem(resume_from_checkpoint=True):
    print(f"Loading preprocessed data from {PREPROCESSED_PATH}...")
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = MPIIGazeDataset(data['train_data'], transform=train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=0, collate_fn=collate_fn
    )
    
    print("Initializing RGB-DINO-Gaze model...")
    model = RGB_DINO_Gaze(unsupervised_pretrain=True).to(DEVICE)
    mask_generator = RGBMEMMaskGenerator()
    
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "dinov2" in n and p.requires_grad], "weight_decay": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "mem_head" in n], "weight_decay": 5e-4}
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    os.makedirs("./models/pretrained_rgb_mem", exist_ok=True)
    os.makedirs("./reconstructions", exist_ok=True)
    
    start_epoch = 0
    if resume_from_checkpoint:
        checkpoint_dir = "./models/pretrained_rgb_mem"
        if os.path.exists(checkpoint_dir):
            existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pth")]
            if existing_checkpoints:
                latest_epoch = max([int(f.split("_")[1].split(".")[0]) for f in existing_checkpoints])
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{latest_epoch}.pth")
                print(f"Resuming from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"]
                for _ in range(start_epoch):
                    scheduler.step()
    
    print(f"Starting RGB-MEM pre-training on {DEVICE}...")
    print("CRITICAL FIX APPLIED: Input masking enabled (DINOv2 cannot see masked patches)")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        total_samples = 0
        pbar = tqdm(train_loader, desc=f"RGB-MEM Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            x, _ = batch
            x = x.to(DEVICE)
            batch_size = x.shape[0]
            
            # Generate unique masks per image
            masks = []
            for i in range(batch_size):
                mask_i = mask_generator.generate_eye_region_mask().to(DEVICE)
                mask_i = mask_generator.filter_illumination_artifacts(x[i], mask_i)
                masks.append(mask_i)
            masks = torch.stack(masks)
            
            # Debug
            if batch_idx == 0 and epoch == start_epoch:
                print(f"\nDebug: Masks shape {masks.shape}")
                print(f"✓ Unique masks: {len(set([m.sum().item() for m in masks]))}/{batch_size}")
                print("✓ Input masking: Enabled (masked patches set to zero)")
            
            with autocast():
                original_patches = F.unfold(x, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
                original_patches = original_patches.transpose(1, 2)
                
                recon_patches = model(x, masks)
                
                if batch_idx % 100 == 0:
                    all_recon = torch.cat(recon_patches, dim=0)
                    print(f"\nDebug: Recon mean: {all_recon.mean().item():.4f}, std: {all_recon.std().item():.4f}")
                
                loss = rgb_mem_recon_loss(recon_patches, original_patches, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.float().item() * batch_size
            total_samples += batch_size
            pbar.set_postfix({"Recon Loss": f"{loss.item():.4f}"})
            
            # Visualization
            if batch_idx % 1000 == 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        b = 0
                        mask_b = masks[b]
                        
                        # VIZ 1: Original
                        # VIZ 2: ONLY reconstructed patches
                        recon_only = torch.zeros_like(original_patches[b], dtype=torch.float32, device=DEVICE)
                        recon_only[~mask_b] = recon_patches[b].float()
                        
                        recon_only_img = F.fold(
                            recon_only.unsqueeze(0).transpose(1, 2),
                            output_size=(IMAGE_SIZE, IMAGE_SIZE),
                            kernel_size=PATCH_SIZE,
                            stride=PATCH_SIZE
                        )[0]
                        
                        # VIZ 3: Full reconstruction
                        recon_full = original_patches[b].clone().float()
                        recon_full[~mask_b] = recon_patches[b].float()
                        
                        recon_full_img = F.fold(
                            recon_full.unsqueeze(0).transpose(1, 2),
                            output_size=(IMAGE_SIZE, IMAGE_SIZE),
                            kernel_size=PATCH_SIZE,
                            stride=PATCH_SIZE
                        )[0]
                        
                        # Denormalize
                        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1).to(DEVICE)
                        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1).to(DEVICE)
                        
                        x_vis = x[b].float() * std + mean
                        recon_only_vis = recon_only_img.float() * std + mean
                        recon_full_vis = recon_full_img.float() * std + mean
                        
                        x_vis = torch.clamp(x_vis, 0.0, 1.0)
                        recon_only_vis = torch.clamp(recon_only_vis, 0.0, 1.0)
                        recon_full_vis = torch.clamp(recon_full_vis, 0.0, 1.0)
                        
                        # Save 3-panel
                        save_image(
                            torch.cat([x_vis, recon_only_vis, recon_full_vis], dim=2),
                            f"./reconstructions/recon_epoch{epoch}_batch{batch_idx}.png"
                        )
        
        avg_loss = total_loss / total_samples
        scheduler.step()
        
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, f"./models/pretrained_rgb_mem/epoch_{epoch+1}.pth")
        
        print(f"Epoch {epoch+1} | Avg Recon Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "./models/pretrained_rgb_mem/rgb_mem_final.pth")
    print("RGB-MEM Pre-Training Complete!")

if __name__ == "__main__":
    pretrain_rgb_mem()