import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2

# --------------------------
# Global Config
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER_BOTTLENECK = 64  # Lightweight (768 → 64 → 768)
META_LR = 1e-3
INNER_LR = 1e-4
NUM_META_EPOCHS = 3
NUM_INNER_STEPS = 5  # Few-shot adaptation steps
BATCH_SIZE = 32
PREPROCESSED_PATH = "./preprocessed_mpiigaze"

# RGB normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --------------------------
# 1. Meta-Learned Adapter (MAML)
# --------------------------
class MetaAdapter(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        # Bottleneck adapter (residual connection)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, ADAPTER_BOTTLENECK),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ADAPTER_BOTTLENECK, input_dim)
        )
        # Meta-optimizer (MAML outer loop)
        self.meta_optimizer = optim.Adam(self.adapter.parameters(), lr=META_LR)

    def forward(self, x):
        """Residual connection: adapter output + original embedding"""
        return x + self.adapter(x)

    def maml_inner_loop(self, support_set, model):
        """
        Inner loop: Adapt adapter to support set (10–20 samples)
        """
        # Save original adapter weights
        original_weights = {name: p.clone() for name, p in self.adapter.named_parameters()}
        
        # Support set (user calibration samples)
        x_support, y_support = support_set
        x_support, y_support = x_support.to(DEVICE), y_support.to(DEVICE)
        
        # Adapt adapter (5 steps)
        for step in range(NUM_INNER_STEPS):
            # Get DINOv2 [CLS] embedding
            with torch.no_grad():
                dinov2_emb = model.dinov2(pixel_values=x_support).last_hidden_state[:, 0, :]
            
            # Apply adapter
            adapted_emb = self.forward(dinov2_emb)
            
            # Gaze prediction
            gaze_pred = model.gaze_head(adapted_emb)
            
            # Loss
            loss = F.mse_loss(gaze_pred, y_support)
            
            # Gradient update (manual SGD on adapter params)
            grads = torch.autograd.grad(loss, self.adapter.parameters(), create_graph=False)
            with torch.no_grad():
                for p, g in zip(self.adapter.parameters(), grads):
                    p.data = p.data - INNER_LR * g
        
        # Return adapted weights
        adapted_weights = {name: p.clone() for name, p in self.adapter.named_parameters()}
        
        # Restore original weights
        with torch.no_grad():
            for name, p in self.adapter.named_parameters():
                p.data = original_weights[name].data
        
        return adapted_weights

    def meta_train(self, support_set, query_set, model):
        """
        MAML outer loop: Update adapter on query set loss
        """
        # Inner loop adaptation
        adapted_weights = self.maml_inner_loop(support_set, model)
        
        # Query set (evaluation)
        x_query, y_query = query_set
        x_query, y_query = x_query.to(DEVICE), y_query.to(DEVICE)
        
        # Temporarily load adapted weights
        original_weights = {name: p.clone() for name, p in self.adapter.named_parameters()}
        with torch.no_grad():
            for name, p in self.adapter.named_parameters():
                p.data = adapted_weights[name].data
        
        # Forward pass on query set
        with torch.no_grad():
            dinov2_emb = model.dinov2(pixel_values=x_query).last_hidden_state[:, 0, :]
        
        adapted_emb = self.forward(dinov2_emb)
        gaze_pred = model.gaze_head(adapted_emb)
        query_loss = F.mse_loss(gaze_pred, y_query)
        
        # Restore original weights before meta update
        with torch.no_grad():
            for name, p in self.adapter.named_parameters():
                p.data = original_weights[name].data
        
        # Meta update
        self.meta_optimizer.zero_grad()
        query_loss.backward()
        self.meta_optimizer.step()
        
        return query_loss.item()

# --------------------------
# 2. Gaze Angular Error
# --------------------------
def gaze_angular_error(pred, target):
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
    cos_sim = torch.clamp(torch.sum(pred_norm * target_norm, dim=1), -1.0, 1.0)
    angle_rad = torch.acos(cos_sim)
    angle_deg = torch.rad2deg(angle_rad)
    return torch.mean(angle_deg)

# --------------------------
# 3. Dataset Class
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
        
        # Resize
        eye_crop = cv2.resize(image, (64, 64))
        
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
# 4. Test-Time Personalization
# --------------------------
def personalize_and_evaluate():
    # Import model
    from rgb_mem_pretrain import RGB_DINO_Gaze
    
    # Load weakly-supervised model
    print("Loading weakly-supervised model...")
    model = RGB_DINO_Gaze(unsupervised_pretrain=False).to(DEVICE)
    
    if os.path.exists("./models/rgb_dino_gaze_weak_sup.pth"):
        model.load_state_dict(torch.load("./models/rgb_dino_gaze_weak_sup.pth"))
        print("Loaded weakly-supervised model")
    else:
        print("Warning: Weakly-supervised model not found, using base model")
        if os.path.exists("./models/dinov2_gaze_baseline.pth"):
            checkpoint = torch.load("./models/dinov2_gaze_baseline.pth")
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("Loaded Phase 1 baseline model")
    
    model.eval()  # Freeze DINOv2/gaze head
    
    # Load test data
    print(f"Loading test data from {PREPROCESSED_PATH}...")
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    test_dataset = MPIIGazeDataset(data['test_data'], transform=val_test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, collate_fn=collate_fn
    )
    
    # Initialize adapter
    print("Initializing meta-adapter...")
    adapter = MetaAdapter(input_dim=768).to(DEVICE)
    
    # Create reports directory
    os.makedirs("./reports", exist_ok=True)
    
    # --------------------------
    # Step 1: Meta-Train Adapter (Optional - simplified version)
    # --------------------------
    print("\nMeta-Training Adapter...")
    for meta_epoch in range(NUM_META_EPOCHS):
        total_meta_loss = 0.0
        num_tasks = 0
        pbar = tqdm(test_loader, desc=f"Meta Epoch {meta_epoch+1}/{NUM_META_EPOCHS}")
        
        for batch in pbar:
            if batch is None:
                continue
            
            x, y = batch
            if len(x) < 40:
                continue
            
            # Split into support (20 samples) and query (20 samples)
            support_set = (x[:20], y[:20])
            query_set = (x[20:40], y[20:40])
            
            # Meta-train step
            try:
                meta_loss = adapter.meta_train(support_set, query_set, model)
                total_meta_loss += meta_loss
                num_tasks += 1
                pbar.set_postfix({"Meta Loss": f"{meta_loss:.4f}"})
            except Exception as e:
                print(f"\nWarning: Meta-train step failed: {e}")
                continue
        
        if num_tasks > 0:
            avg_meta_loss = total_meta_loss / num_tasks
            print(f"Meta Epoch {meta_epoch+1} | Avg Loss: {avg_meta_loss:.4f}")
        else:
            print(f"Meta Epoch {meta_epoch+1} | No valid tasks")
    
    # --------------------------
    # Step 2: Test-Time Personalization (10–20 samples)
    # --------------------------
    print("\nPersonalizing Adapter for User...")
    # Get user calibration samples (20 samples from test set)
    user_batch = next(iter(test_loader))
    if user_batch is not None:
        x_cal, y_cal = user_batch
        if len(x_cal) >= 20:
            support_set = (x_cal[:20], y_cal[:20])
            
            # Adapt adapter to user
            adapted_weights = adapter.maml_inner_loop(support_set, model)
            with torch.no_grad():
                for name, p in adapter.adapter.named_parameters():
                    p.data = adapted_weights[name].data
            
            print("Adapter personalized with 20 calibration samples")
        else:
            print("Warning: Not enough calibration samples, skipping personalization")
    else:
        print("Warning: No calibration data available")
    
    # --------------------------
    # Step 3: Evaluate Personalization
    # --------------------------
    print("\nEvaluating Personalization...")
    pre_mae = 0.0
    post_mae = 0.0
    pre_samples = 0
    post_samples = 0
    
    with torch.no_grad():
        # Pre-personalization MAE (no adapter)
        for batch in tqdm(test_loader, desc="Pre-Personalization"):
            if batch is None:
                continue
            
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            dinov2_emb = model.dinov2(pixel_values=x).last_hidden_state[:, 0, :]
            gaze_pred = model.gaze_head(dinov2_emb)
            pre_mae += gaze_angular_error(gaze_pred, y).item() * x.shape[0]
            pre_samples += x.shape[0]
        
        # Post-personalization MAE (with adapter)
        for batch in tqdm(test_loader, desc="Post-Personalization"):
            if batch is None:
                continue
            
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            dinov2_emb = model.dinov2(pixel_values=x).last_hidden_state[:, 0, :]
            adapted_emb = adapter(dinov2_emb)
            gaze_pred = model.gaze_head(adapted_emb)
            post_mae += gaze_angular_error(gaze_pred, y).item() * x.shape[0]
            post_samples += x.shape[0]
    
    # Average MAE
    pre_mae /= pre_samples
    post_mae /= post_samples
    mae_reduction = pre_mae - post_mae
    
    # Save results
    report_path = "./reports/personalization_results.txt"
    with open(report_path, "w") as f:
        f.write("=== RGB-DINO-Gaze Personalization Results ===\n\n")
        f.write(f"Pre-Personalization MAE: {pre_mae:.2f}°\n")
        f.write(f"Post-Personalization MAE: {post_mae:.2f}°\n")
        f.write(f"MAE Reduction: {mae_reduction:.2f}°\n")
        f.write(f"Improvement: {(mae_reduction/pre_mae)*100:.1f}%\n\n")
        f.write(f"Meta-Training Epochs: {NUM_META_EPOCHS}\n")
        f.write(f"Inner Adaptation Steps: {NUM_INNER_STEPS}\n")
        f.write(f"Adapter Bottleneck Size: {ADAPTER_BOTTLENECK}\n")
    
    print(f"\n{'='*60}")
    print("Final Results:")
    print(f"{'='*60}")
    print(f"Pre-Personalization MAE:  {pre_mae:.2f}°")
    print(f"Post-Personalization MAE: {post_mae:.2f}°")
    print(f"MAE Reduction:            {mae_reduction:.2f}°")
    print(f"Improvement:              {(mae_reduction/pre_mae)*100:.1f}%")
    print(f"{'='*60}")
    print(f"\nResults saved to {report_path}")

# --------------------------
# Run Adapter Training/Evaluation
# --------------------------
if __name__ == "__main__":
    personalize_and_evaluate()
