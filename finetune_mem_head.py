"""
Fine-tuning script for RGB-MEM reconstruction head only.
Retrains just the mem_head for 1 epoch (1-2 hours) to improve reconstruction quality.
Freezes DINOv2 backbone to preserve learned global eye structure.
"""

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
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256 total patches
MASK_RATIO = 0.4
BATCH_SIZE = 8
EPOCHS = 1  # Only 1 epoch for fine-tuning
LR = 1e-4  # Higher LR for fine-tuning head only
PREPROCESSED_PATH = "./preprocessed_mpiigaze"

# Mixed Precision Scaler
scaler = GradScaler()

# RGB normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --------------------------
# 1. RGB-MEM Mask Generator (expanded pupil preservation)
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
        
        if len(non_keep_patches) < num_mask_total:
            eyelid_to_mask = np.random.choice(eyelid_patches, size=int(0.1*len(eyelid_patches)), replace=False)
            non_keep_patches = np.unique(np.concatenate([non_keep_patches, eyelid_to_mask]))
        
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
        eye_crop = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        
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
# 3. RGB-DINO-Gaze Model with Improved mem_head
# --------------------------
class RGB_DINO_Gaze(nn.Module):
    def __init__(self, unsupervised_pretrain=True):
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-base")
        self.unsupervised_pretrain = unsupervised_pretrain
        
        # Freeze all DINOv2 layers for fine-tuning
        for param in self.dinov2.parameters():
            param.requires_grad = False
        
        # RGB-MEM: Improved Reconstruction Head with upsampling for fine details
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
        self.mem_head = nn.Identity()
        
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
        
        # Gaze Regression Head (frozen for pretraining)
        self.gaze_head = nn.Sequential(
            nn.Linear(self.dinov2.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )
        for param in self.gaze_head.parameters():
            param.requires_grad = False

    def forward(self, x, mask=None):
        outputs = self.dinov2(pixel_values=x, output_hidden_states=True)
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Skip [CLS]
        
        if self.unsupervised_pretrain and mask is not None:
            batch_size, num_patches, embed_dim = patch_embeddings.shape
            mask_expanded = mask.unsqueeze(0).expand(batch_size, -1)
            
            num_masked = (~mask).sum().item()
            masked_embeddings = patch_embeddings[~mask_expanded].view(batch_size, num_masked, embed_dim)
            
            # Pass through linear layers
            linear_out = self.mem_head_linear(masked_embeddings)
            
            # Reshape for conv upsampling
            linear_out = linear_out.view(batch_size * num_masked, 3 * 4, PATCH_SIZE, PATCH_SIZE)
            
            # Apply conv upsampling
            conv_out = self.mem_head_conv(linear_out)
            
            # Reshape back
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
    batch_size, num_patches, patch_dim = original_patches.shape
    mask_expanded = mask.unsqueeze(0).expand(batch_size, -1)
    
    original_masked = original_patches[~mask_expanded].view(batch_size, -1, patch_dim)
    
    return F.mse_loss(recon_patches, original_masked)

# --------------------------
# 5. Fine-tuning Loop (mem_head only)
# --------------------------
def finetune_mem_head():
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
    
    # Initialize model
    print("Initializing RGB-DINO-Gaze model...")
    model = RGB_DINO_Gaze(unsupervised_pretrain=True).to(DEVICE)
    
    # Load pre-trained RGB-MEM backbone (DINOv2 features)
    pretrained_path = "./models/pretrained_rgb_mem/rgb_mem_final.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading pre-trained RGB-MEM from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path)
        
        # Load only DINOv2 weights (filter out old mem_head)
        dinov2_weights = {k: v for k, v in checkpoint.items() if "dinov2" in k}
        model.load_state_dict(dinov2_weights, strict=False)
        print(f"Loaded {len(dinov2_weights)} DINOv2 weights from checkpoint")
    else:
        print("WARNING: No pre-trained RGB-MEM found! Training from scratch.")
    
    mask_generator = RGBMEMMaskGenerator()
    
    # Optimizer only for mem_head (DINOv2 frozen)
    mem_head_params = list(model.mem_head_linear.parameters()) + list(model.mem_head_conv.parameters())
    optimizer = torch.optim.AdamW(mem_head_params, lr=LR, weight_decay=1e-4)
    
    # Create output directories
    os.makedirs("./models/pretrained_rgb_mem", exist_ok=True)
    os.makedirs("./reconstructions_improved", exist_ok=True)
    
    print(f"Starting mem_head fine-tuning on {DEVICE}...")
    print(f"Total trainable params: {sum(p.numel() for p in mem_head_params if p.requires_grad):,}")
    
    # Training Loop
    model.train()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(train_loader, desc="Fine-tuning mem_head")
    
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        
        x, _ = batch
        x = x.to(DEVICE)
        batch_size = x.shape[0]
        
        # Generate eye-specific mask
        mask = mask_generator.generate_eye_region_mask().to(DEVICE)
        mask = mask_generator.filter_illumination_artifacts(x[0], mask)
        
        # Mixed Precision Forward Pass
        with autocast():
            original_patches = F.unfold(x, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
            original_patches = original_patches.transpose(1, 2)
            
            recon_patches = model(x, mask)
            loss = rgb_mem_recon_loss(recon_patches, original_patches, mask)
        
        # Backward Pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(mem_head_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Track Metrics
        total_loss += loss.float().item() * batch_size
        total_samples += batch_size
        pbar.set_postfix({"Recon Loss": f"{loss.item():.4f}"})
        
        # Visualization every 2000 batches
        if batch_idx % 2000 == 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    recon_full = torch.zeros_like(original_patches, dtype=torch.float32, device=DEVICE)
                    mask_3d = mask.unsqueeze(0).unsqueeze(-1).expand(original_patches.shape)
                    recon_patches_fp32 = recon_patches.float()
                    recon_full[~mask_3d] = recon_patches_fp32.flatten()
                    
                    recon_img = F.fold(
                        recon_full.transpose(1, 2),
                        output_size=(IMAGE_SIZE, IMAGE_SIZE),
                        kernel_size=PATCH_SIZE,
                        stride=PATCH_SIZE
                    )
                    
                    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1).to(DEVICE)
                    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1).to(DEVICE)
                    
                    x_vis = x.float() * std + mean
                    recon_vis = recon_img.float() * std + mean
                    
                    # Apply Gaussian blur for smoother output
                    recon_vis = TF.gaussian_blur(recon_vis, kernel_size=3, sigma=0.5)
                    
                    x_vis = torch.clamp(x_vis, 0.0, 1.0)
                    recon_vis = torch.clamp(recon_vis, 0.0, 1.0)
                    
                    save_image(
                        torch.cat([x_vis[0], recon_vis[0]], dim=2),
                        f"./reconstructions_improved/recon_finetune_batch{batch_idx}.png"
                    )
    
    # Final Metrics
    avg_loss = total_loss / total_samples
    print(f"\nFine-tuning Complete | Avg Recon Loss: {avg_loss:.4f}")
    
    # Save Improved Model
    torch.save(model.state_dict(), "./models/pretrained_rgb_mem/rgb_mem_improved.pth")
    print("Saved improved model to ./models/pretrained_rgb_mem/rgb_mem_improved.pth")

# --------------------------
# Run Fine-tuning
# --------------------------
if __name__ == "__main__":
    finetune_mem_head()
