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
import torchvision.transforms.functional as TF
from transformers import AutoModel

# --------------------------
# Global Config
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
PATCH_SIZE = 14
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) **2  # 256 total patches
MASK_RATIO = 0.4
BATCH_SIZE = 8
EPOCHS = 5
LR = 5e-5
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
        
        # Pupil (center 5x5 patches - expanded for fine detail preservation)
        pupil_patches = patch_grid[5:11, 5:11].flatten()
        
        # Eyelid (perimeter 2 patches)
        eyelid_patches = np.concatenate([
            patch_grid[0:2, :].flatten(),   # Top
            patch_grid[14:16, :].flatten(), # Bottom
            patch_grid[:, 0:2].flatten(),   # Left
            patch_grid[:, 14:16].flatten()  # Right
        ])
        
        # Combine keep patches (pupil + eyelid)
        keep_patches = np.unique(np.concatenate([pupil_patches, eyelid_patches]))
        non_keep_patches = np.setdiff1d(np.arange(self.num_patches), keep_patches)
        
        # ENFORCE 30% TOTAL MASK RATIO (77 patches)
        num_mask_total = int(0.3 * self.num_patches)  # 0.3 * 256 = 77
        # If non_keep_patches < 77, expand to include some eyelid patches (safe)
        if len(non_keep_patches) < num_mask_total:
            # Add 10% of eyelid patches to the mask pool (still preserve critical regions)
            eyelid_to_mask = np.random.choice(eyelid_patches, size=int(0.1*len(eyelid_patches)), replace=False)
            non_keep_patches = np.unique(np.concatenate([non_keep_patches, eyelid_to_mask]))
        
        num_mask = min(num_mask_total, len(non_keep_patches))
        mask_patches = np.random.choice(non_keep_patches, num_mask, replace=False)
        
        # Create boolean mask (True = keep, False = mask)
        mask = torch.ones(self.num_patches, dtype=torch.bool)
        mask[mask_patches] = False
        
        # Final check: Ensure mask ratio is ~30%
        actual_mask_ratio = (~mask).sum().float() / mask.numel()
        if abs(actual_mask_ratio - 0.3) > 0.02:
            # Adjust to hit 30% exactly
            current_masked = (~mask).sum().item()
            target_masked = int(0.3 * self.num_patches)
            if current_masked < target_masked:
                # Add more patches to mask (from non-keep)
                to_add = target_masked - current_masked
                available = [p for p in non_keep_patches if mask[p] == True]
                if len(available) >= to_add:
                    extra_mask = np.random.choice(available, size=to_add, replace=False)
                    mask[extra_mask] = False
            else:
                # Remove some patches from mask
                to_remove = current_masked - target_masked
                available = [p for p in non_keep_patches if mask[p] == False]
                if len(available) >= to_remove:
                    extra_unmask = np.random.choice(available, size=to_remove, replace=False)
                    mask[extra_unmask] = True
        
        return mask

    def filter_illumination_artifacts(self, rgb_image, mask):
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(rgb_image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(rgb_image.device)
        rgb_denorm = rgb_image * std + mean
        rgb_denorm = torch.clamp(rgb_denorm, 0, 1)
        
        # Convert to numpy for OpenCV
        rgb_np = (rgb_denorm.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Convert to HSV for brightness analysis
        hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
        brightness = hsv[..., 2]  # Value channel (0-255)
        
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
        mask[bright_patches] = True  # Keep bright patches
        
        # Re-check: Ensure at least 5 masked patches after filtering
        if (mask == False).sum() < 5:
            patch_grid = np.arange(self.num_patches).reshape(16, 16)
            pupil_patches = patch_grid[6:10, 6:10].flatten()
            eyelid_patches = np.concatenate([
                patch_grid[0:2, :].flatten(),
                patch_grid[14:16, :].flatten(),
                patch_grid[:, 0:2].flatten(),
                patch_grid[:, 14:16].flatten()
            ])
            keep_patches = np.unique(np.concatenate([pupil_patches, eyelid_patches]))
            non_keep_patches = np.setdiff1d(np.arange(self.num_patches), keep_patches)
            available = [p for p in non_keep_patches if mask[p] == True]
            if len(available) >= 5:
                extra_mask = np.random.choice(available, 5, replace=False)
                mask[extra_mask] = False
        
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
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        
        # Resize and enhance
        eye_crop = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # Fixed to 224x224
        
        # Apply transforms
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
# 3. RGB-DINO-Gaze Model with RGB-MEM
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
        
        # RGB-MEM: Masked Patch Reconstruction Head with upsampling for fine details
        self.mem_head_linear = nn.Sequential(
            nn.Linear(self.dinov2.config.hidden_size, 2048),  # Larger hidden dim
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced dropout for better texture recovery
            nn.Linear(2048, 3 * 4 * PATCH_SIZE * PATCH_SIZE),  # 4x more features for upsampling
        )
        
        # Convolutional upsampler for texture recovery
        self.mem_head_conv = nn.Sequential(
            nn.ConvTranspose2d(3 * 4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Constrain outputs to [-1,1]
        )
        
        # Backward compatibility wrapper
        self.mem_head = nn.Identity()  # Placeholder for state_dict loading
        
        # Initialize reconstruction head properly
        for m in list(self.mem_head_linear.modules()) + list(self.mem_head_conv.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
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

    def forward(self, x, mask=None):
        outputs = self.dinov2(pixel_values=x, output_hidden_states=True)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Skip [CLS]
        
        if self.unsupervised_pretrain and mask is not None:
            # Fixed masking logic - preserve batch structure
            batch_size, num_patches, embed_dim = patch_embeddings.shape
            mask_expanded = mask.unsqueeze(0).expand(batch_size, -1)  # [B, 256]
            
            # Count masked patches per batch
            num_masked = (~mask).sum().item()
            
            # Extract masked embeddings with proper reshaping
            masked_embeddings = patch_embeddings[~mask_expanded].view(batch_size, num_masked, embed_dim)
            
            # Pass through linear layers
            linear_out = self.mem_head_linear(masked_embeddings)  # (B, num_masked, 3*4*14*14)
            
            # Reshape for conv upsampling: (B*num_masked, 12, 14, 14)
            linear_out = linear_out.view(batch_size * num_masked, 3 * 4, PATCH_SIZE, PATCH_SIZE)
            
            # Apply conv upsampling for texture recovery
            conv_out = self.mem_head_conv(linear_out)  # (B*num_masked, 3, 14, 14)
            
            # Reshape back to (B, num_masked, 3*14*14)
            recon_patches = conv_out.view(batch_size, num_masked, 3 * PATCH_SIZE * PATCH_SIZE)
            
            return recon_patches
        else:
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            gaze_pred = self.gaze_head(cls_embedding)
            return gaze_pred, patch_embeddings

# --------------------------
# 4. RGB-MEM Loss Function
# --------------------------
def rgb_mem_recon_loss(recon_patches, original_patches, mask):
    # Fixed loss calculation - no more epsilon addition
    batch_size, num_patches, patch_dim = original_patches.shape
    mask_expanded = mask.unsqueeze(0).expand(batch_size, -1)  # [B, 256]
    
    # Extract original masked patches with proper alignment
    original_masked = original_patches[~mask_expanded].view(batch_size, -1, patch_dim)
    
    # Clean MSE loss calculation
    return F.mse_loss(recon_patches, original_masked)

# --------------------------
# 5. Unsupervised Pre-Training Loop
# --------------------------
def pretrain_rgb_mem(resume_from_checkpoint=True):
    print(f"Loading preprocessed data from {PREPROCESSED_PATH}...")
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create dataset
    train_dataset = MPIIGazeDataset(data['train_data'], transform=train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=0, collate_fn=collate_fn
    )
    
    # Initialize model and mask generator
    print("Initializing RGB-DINO-Gaze model...")
    model = RGB_DINO_Gaze(unsupervised_pretrain=True).to(DEVICE)
    mask_generator = RGBMEMMaskGenerator()
    
    # Optimizer/Scheduler (separate parameter groups for better regularization)
    param_groups = [
        # DINOv2 backbone (light weight decay)
        {"params": [p for n, p in model.named_parameters() if "dinov2" in n and p.requires_grad], "weight_decay": 1e-5},
        # Reconstruction head (strong weight decay)
        {"params": [p for n, p in model.named_parameters() if "mem_head" in n], "weight_decay": 5e-4}
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=4e-5)  # Lower LR from 5e-5 → 4e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Create save directories
    os.makedirs("./models/pretrained_rgb_mem", exist_ok=True)
    os.makedirs("./reconstructions", exist_ok=True)  # For visualization
    
    # Check for existing checkpoints
    start_epoch = 0
    if resume_from_checkpoint:
        checkpoint_dir = "./models/pretrained_rgb_mem"
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
    
    # Training Loop
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
            
            # Add mild Gaussian noise to prevent memorization
            x = x + torch.randn_like(x) * 0.01
            x = torch.clamp(x, -2.0, 2.0)  # Keep within reasonable range for normalized input
            
            # Generate eye-specific mask
            mask = mask_generator.generate_eye_region_mask().to(DEVICE)
            
            # Filter illumination artifacts
            mask = mask_generator.filter_illumination_artifacts(x[0], mask)
            
            # Debug mask information
            if batch_idx == 0 and epoch == 0:
                print(f"\nDebug: Mask shape {mask.shape}, Masked patches: { (~mask).sum().item() }")
                print(f"Debug: Mask ratio: { (~mask).sum().float() / mask.numel():.2f}")
            
            # Mixed Precision Forward Pass
            with autocast():
                # Extract original RGB patches
                original_patches = F.unfold(x, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
                original_patches = original_patches.transpose(1, 2)  # (B, 256, 3*14*14)
                
                # Reconstruct masked patches
                recon_patches = model(x, mask)
                
                # Debug output statistics
                if batch_idx % 100 == 0:
                    print(f"\nDebug: Recon mean: {recon_patches.mean().item():.4f}, std: {recon_patches.std().item():.4f}")
                
                # Compute loss
                loss = rgb_mem_recon_loss(recon_patches, original_patches, mask)
            
            # Backward Pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Track Metrics (accumulate in FP32 to avoid precision loss)
            total_loss += loss.float().item() * batch_size
            total_samples += batch_size
            pbar.set_postfix({"Recon Loss": f"{loss.item():.4f}"})
            
            # Visualization (fixed dtype mismatch)
            if batch_idx % 1000 == 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for visualization
                        # Create full reconstruction tensor (FP32)
                        recon_full = torch.zeros_like(original_patches, dtype=torch.float32, device=DEVICE)
                        
                        # Expand mask to 3D (match original_patches shape)
                        mask_3d = mask.unsqueeze(0).unsqueeze(-1).expand(original_patches.shape)
                        
                        # Cast recon_patches to FP32 to match recon_full
                        recon_patches_fp32 = recon_patches.float()
                        
                        # Assign reconstructed patches (fix index put error)
                        recon_full[~mask_3d] = recon_patches_fp32.flatten()
                        
                        # Convert patches back to image (FP32)
                        recon_img = F.fold(
                            recon_full.transpose(1, 2),
                            output_size=(IMAGE_SIZE, IMAGE_SIZE),
                            kernel_size=PATCH_SIZE,
                            stride=PATCH_SIZE
                        )
                        
                        # Denormalize for visualization (FP32)
                        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1).to(DEVICE)
                        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1).to(DEVICE)
                        
                        # Cast input image to FP32 and denormalize
                        x_vis = x.float() * std + mean
                        recon_vis = recon_img.float() * std + mean
                        
                        # Apply mild Gaussian blur to smooth pixelation
                        recon_vis = TF.gaussian_blur(recon_vis, kernel_size=3, sigma=0.5)
                        
                        # Clamp to valid RGB range [0,1]
                        x_vis = torch.clamp(x_vis, 0.0, 1.0)
                        recon_vis = torch.clamp(recon_vis, 0.0, 1.0)
                        
                        # Save comparison (only first image in batch to save space)
                        save_image(
                            torch.cat([x_vis[0], recon_vis[0]], dim=2),
                            f"./reconstructions/recon_epoch{epoch}_batch{batch_idx}.png"
                        )
        
        # Epoch Metrics
        avg_loss = total_loss / total_samples
        scheduler.step()
        
        # Save Checkpoint
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, f"./models/pretrained_rgb_mem/epoch_{epoch+1}.pth")
        
        print(f"Epoch {epoch+1} | Avg Recon Loss: {avg_loss:.4f}")
    
    # Save Final Model
    torch.save(model.state_dict(), "./models/pretrained_rgb_mem/rgb_mem_final.pth")
    print("RGB-MEM Pre-Training Complete!")

# --------------------------
# Run Pre-Training
# --------------------------
if __name__ == "__main__":
    pretrain_rgb_mem()
