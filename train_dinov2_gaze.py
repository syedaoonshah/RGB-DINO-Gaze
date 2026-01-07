import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel
import numpy as np

# --------------------------
# 1. Configuration
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
BEST_MODEL_PATH = "./models/dinov2_gaze_baseline.pth"
PREPROCESSED_PATH = "./preprocessed_mpiigaze"

# Create model directory
os.makedirs("./models", exist_ok=True)

# RGB normalization (ImageNet stats for DINOv2)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --------------------------
# 2. Custom Dataset Class
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
        
        # Ensure RGB format (H, W, C)
        if len(image.shape) == 2:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[0] == 3:  # (C, H, W) format
            image = image.transpose(1, 2, 0)  # Convert to (H, W, C)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        gaze_tensor = torch.tensor(gaze, dtype=torch.float32)
        return image, gaze_tensor

# Collate function to filter None samples
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --------------------------
# 3. Load Preprocessed Data
# --------------------------
def load_data():
    print(f"Loading preprocessed data from {PREPROCESSED_PATH}...")
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create datasets
    train_dataset = MPIIGazeDataset(data['train_data'], transform=train_transform)
    val_dataset = MPIIGazeDataset(data['val_data'], transform=val_test_transform)
    test_dataset = MPIIGazeDataset(data['test_data'], transform=val_test_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

# --------------------------
# 4. DINOv2 + Gaze Regression Head
# --------------------------
class DINOv2Gaze(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained DINOv2-base
        print("Loading DINOv2-base model...")
        self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-base")
        
        # Freeze first 6 layers of DINOv2 (fine-tune top layers only)
        for i, layer in enumerate(self.dinov2.encoder.layer):
            if i < 6:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Lightweight regression head for 3D gaze vector (output size=3)
        hidden_size = self.dinov2.config.hidden_size  # 768 for dinov2-base
        self.gaze_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)  # 3D gaze vector (x, y, z)
        )

    def forward(self, x):
        # DINOv2 forward pass (extract [CLS] token embedding)
        outputs = self.dinov2(pixel_values=x)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token (batch_size, hidden_size)
        
        # Gaze prediction
        gaze_pred = self.gaze_head(cls_embedding)
        return gaze_pred

# --------------------------
# 5. Mean Angular Error (MAE) Calculation
# --------------------------
def gaze_angular_error(pred, target):
    """Compute mean angular error between predicted and ground-truth gaze vectors."""
    # Normalize vectors to unit length
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity (clamp to avoid numerical issues)
    cos_sim = torch.clamp(torch.sum(pred_norm * target_norm, dim=1), -1.0, 1.0)
    
    # Convert to radians, then degrees
    angle_rad = torch.acos(cos_sim)
    angle_deg = torch.rad2deg(angle_rad)
    
    return torch.mean(angle_deg)

# --------------------------
# 6. Training Loop
# --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    best_val_loss = float("inf")
    
    for epoch in range(EPOCHS):
        # Train phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            if batch is None:
                continue
            
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            train_mae += gaze_angular_error(y_pred, y).item() * batch_size
            train_samples += batch_size
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "mae": f"{gaze_angular_error(y_pred, y).item():.2f}°"
            })
        
        # Val phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for batch in pbar:
                if batch is None:
                    continue
                
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                y_pred = model(x)
                loss = criterion(y_pred, y)
                
                batch_size = x.size(0)
                val_loss += loss.item() * batch_size
                val_mae += gaze_angular_error(y_pred, y).item() * batch_size
                val_samples += batch_size
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "mae": f"{gaze_angular_error(y_pred, y).item():.2f}°"
                })
        
        # Average metrics
        train_loss /= train_samples
        train_mae /= train_samples
        val_loss /= val_samples
        val_mae /= val_samples
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch stats
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f}°")
        print(f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}°")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae
            }, BEST_MODEL_PATH)
            print(f"Best model saved to {BEST_MODEL_PATH}")

# --------------------------
# 7. Evaluation on Test Set
# --------------------------
def evaluate_test(model, test_loader):
    model.eval()
    test_mae = 0.0
    test_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating Test Set")
        for batch in pbar:
            if batch is None:
                continue
            
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            y_pred = model(x)
            batch_mae = gaze_angular_error(y_pred, y).item()
            
            batch_size = x.size(0)
            test_mae += batch_mae * batch_size
            test_samples += batch_size
            
            pbar.set_postfix({"test_mae": f"{batch_mae:.2f}°"})
    
    test_mae /= test_samples
    print(f"\nFinal Test MAE: {test_mae:.2f}°")
    return test_mae

# --------------------------
# 8. Main Execution
# --------------------------
if __name__ == "__main__":
    # Load data
    train_loader, val_loader, test_loader = load_data()
    
    # Initialize model, loss, optimizer, scheduler
    print(f"Initializing model on {DEVICE}...")
    model = DINOv2Gaze().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train model
    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # Load best model and evaluate test set
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate_test(model, test_loader)
