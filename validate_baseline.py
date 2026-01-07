import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel
import numpy as np
import random

# --------------------------
# 1. Configuration
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
BEST_MODEL_PATH = "./models/dinov2_gaze_baseline.pth"
PREPROCESSED_PATH = "./preprocessed_mpiigaze"
REPORT_PATH = "./reports/baseline_performance.txt"

# Create report directory
os.makedirs("./reports", exist_ok=True)

# RGB normalization
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

# Collate function
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --------------------------
# 3. Load Data
# --------------------------
def load_data():
    print(f"Loading preprocessed data from {PREPROCESSED_PATH}...")
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    
    # Define transforms
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create datasets
    train_dataset = MPIIGazeDataset(data['train_data'], transform=val_test_transform)
    val_dataset = MPIIGazeDataset(data['val_data'], transform=val_test_transform)
    test_dataset = MPIIGazeDataset(data['test_data'], transform=val_test_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, 
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
    
    return train_loader, val_loader, test_loader, data['test_data']

# --------------------------
# 4. DINOv2 + Gaze Regression Head
# --------------------------
class DINOv2Gaze(nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-base")
        hidden_size = self.dinov2.config.hidden_size
        self.gaze_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        outputs = self.dinov2(pixel_values=x)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        gaze_pred = self.gaze_head(cls_embedding)
        return gaze_pred

# --------------------------
# 5. Mean Angular Error (MAE) Calculation
# --------------------------
def gaze_angular_error(pred, target):
    """Compute angular error between predicted and ground-truth gaze vectors."""
    # Normalize vectors to unit length
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity (clamp to avoid numerical issues)
    cos_sim = torch.clamp(torch.sum(pred_norm * target_norm, dim=1), -1.0, 1.0)
    
    # Convert to radians, then degrees
    angle_rad = torch.acos(cos_sim)
    angle_deg = torch.rad2deg(angle_rad)
    
    return angle_deg

# --------------------------
# 6. Compute Metrics for All Sets
# --------------------------
def compute_metrics(model, loader, name):
    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Computing {name} metrics"):
            if batch is None:
                continue
            
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            
            loss = loss_fn(y_pred, y)
            mae = gaze_angular_error(y_pred, y)
            
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae.sum().item()
            total_samples += batch_size
            
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    return avg_loss, avg_mae, all_preds, all_targets

# --------------------------
# 7. Main Execution
# --------------------------
if __name__ == "__main__":
    # Load model
    print("Loading model...")
    model = DINOv2Gaze().to(DEVICE)
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load data
    train_loader, val_loader, test_loader, test_data = load_data()
    
    # Compute metrics
    print("\nComputing metrics...")
    train_loss, train_mae, train_preds, train_targets = compute_metrics(model, train_loader, "Train")
    val_loss, val_mae, val_preds, val_targets = compute_metrics(model, val_loader, "Val")
    test_loss, test_mae, test_preds, test_targets = compute_metrics(model, test_loader, "Test")
    
    # Generate sample predictions
    print("\nGenerating sample predictions...")
    sample_indices = random.sample(range(len(test_preds)), min(5, len(test_preds)))
    sample_report = []
    
    for idx in sample_indices:
        pred = np.array(test_preds[idx])
        target = np.array(test_targets[idx])
        
        # Compute angular error for sample
        pred_norm = pred / (np.linalg.norm(pred) + 1e-8)
        target_norm = target / (np.linalg.norm(target) + 1e-8)
        cos_sim = np.clip(np.dot(pred_norm, target_norm), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_sim))
        
        sample_report.append(
            f"Sample {idx}:\n"
            f"  Predicted Gaze: [{pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f}]\n"
            f"  Ground-Truth Gaze: [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]\n"
            f"  Angular Error: {angle_deg:.2f}°\n"
        )
    
    # Write report
    print(f"\nWriting report to {REPORT_PATH}...")
    with open(REPORT_PATH, "w") as f:
        f.write("=== RGB-DINO-Gaze Baseline Performance Report ===\n")
        f.write(f"Best Model Epoch: {checkpoint['epoch']}\n")
        f.write(f"Val Loss at Best Model: {checkpoint['val_loss']:.4f}\n")
        f.write(f"Val MAE at Best Model: {checkpoint['val_mae']:.2f}°\n\n")
        
        f.write("=== Full Metrics ===\n")
        f.write(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f}°\n")
        f.write(f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}°\n")
        f.write(f"Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.2f}°\n\n")
        
        f.write("=== Sample Test Predictions ===\n")
        f.write("\n".join(sample_report))
    
    print(f"\nReport saved to {REPORT_PATH}")
    print(f"\nKey Baseline Metrics:")
    print(f"Train MAE: {train_mae:.2f}°")
    print(f"Val MAE: {val_mae:.2f}°")
    print(f"Test MAE: {test_mae:.2f}° (Expected: 8–10° for MPIIGaze)")
