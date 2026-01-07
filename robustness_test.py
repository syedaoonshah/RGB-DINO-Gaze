"""
Phase 3 Module 4: Robustness Test
Validates performance under illumination variation, occlusion, and glasses wearing.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import cv2

# --------------------------
# Global Config
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
PREPROCESSED_PATH = "./preprocessed_mpiigaze"

# RGB normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --------------------------
# Dataset Class with Augmentations for Robustness Testing
# --------------------------
class MPIIGazeDataset(Dataset):
    def __init__(self, data, transform=None, challenge_type="normal"):
        self.data = data
        self.transform = transform
        self.challenge_type = challenge_type

    def __len__(self):
        return len(self.data)

    def apply_challenge(self, image):
        """Apply challenge-specific augmentations to simulate real-world conditions"""
        if self.challenge_type == "low_illumination":
            # Reduce brightness by 50%
            image = (image * 0.5).astype(np.uint8)
        elif self.challenge_type == "high_illumination":
            # Increase brightness (with clipping)
            image = np.clip(image * 1.5, 0, 255).astype(np.uint8)
        elif self.challenge_type == "mild_occlusion":
            # Simulate eyelid occlusion: black rectangle on top 20% of image
            h, w = image.shape[:2]
            image[:int(h*0.2), :] = 0
        elif self.challenge_type == "glasses":
            # Simulate glasses glare: add bright spots
            h, w = image.shape[:2]
            # Add elliptical bright spots
            cv2.ellipse(image, (w//4, h//2), (w//8, h//6), 0, 0, 360, (200, 200, 200), -1)
            cv2.ellipse(image, (3*w//4, h//2), (w//8, h//6), 0, 0, 360, (200, 200, 200), -1)
        elif self.challenge_type == "motion_blur":
            # Apply horizontal motion blur
            kernel_size = 5
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = np.ones(kernel_size) / kernel_size
            image = cv2.filter2D(image, -1, kernel)
        elif self.challenge_type == "noise":
            # Add Gaussian noise
            noise = np.random.normal(0, 25, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return image

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
        
        eye_crop = cv2.resize(image, (64, 64))
        
        # Apply challenge-specific augmentation
        eye_crop = self.apply_challenge(eye_crop)
        
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
# Meta-Adapter Class
# --------------------------
class MetaAdapter(torch.nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return x + self.adapter(x)

# --------------------------
# Gaze Angular Error
# --------------------------
def gaze_angular_error(pred, target):
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
    cos_sim = torch.clamp(torch.sum(pred_norm * target_norm, dim=1), -1.0, 1.0)
    angle_rad = torch.acos(cos_sim)
    angle_deg = torch.rad2deg(angle_rad)
    return torch.mean(angle_deg)

# --------------------------
# Load Model
# --------------------------
def load_model():
    """
    Load the best available model for robustness testing.
    
    CRITICAL FIX: For RGB-MEM, we load baseline gaze_head + RGB-MEM encoder.
    """
    from rgb_mem_pretrain import RGB_DINO_Gaze
    
    model = RGB_DINO_Gaze(unsupervised_pretrain=False).to(DEVICE)
    
    # First, ALWAYS load baseline gaze_head
    if os.path.exists("./models/dinov2_gaze_baseline.pth"):
        baseline_checkpoint = torch.load("./models/dinov2_gaze_baseline.pth", map_location=DEVICE)
        if isinstance(baseline_checkpoint, dict) and "model_state_dict" in baseline_checkpoint:
            model.load_state_dict(baseline_checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(baseline_checkpoint, strict=False)
        print("Loaded: Baseline gaze_head")
    
    # Then load best available encoder
    if os.path.exists("./models/rgb_dino_gaze_weak_sup.pth"):
        model.load_state_dict(torch.load("./models/rgb_dino_gaze_weak_sup.pth", map_location=DEVICE))
        print("Loaded: Weakly-supervised model (full)")
    elif os.path.exists("./models/pretrained_rgb_mem/rgb_mem_final.pth"):
        rgb_mem_weights = torch.load("./models/pretrained_rgb_mem/rgb_mem_final.pth", map_location=DEVICE)
        if isinstance(rgb_mem_weights, dict) and "model_state_dict" in rgb_mem_weights:
            rgb_mem_weights = rgb_mem_weights["model_state_dict"]
        # CRITICAL: Filter to only DINOv2 layers (not patch reconstruction head)
        filtered_weights = {k: v for k, v in rgb_mem_weights.items() 
                           if "dinov2" in k and "patch_reconstruction" not in k}
        model.load_state_dict(filtered_weights, strict=False)
        print(f"Loaded: RGB-MEM encoder ({len(filtered_weights)} layers) + Baseline gaze_head")
    else:
        print("Using: Phase 1 baseline model only")
    
    model.eval()
    return model

def load_adapter():
    """Load meta-adapter"""
    adapter = MetaAdapter(input_dim=768).to(DEVICE)
    if os.path.exists("./models/adapters/user_adapter.pth"):
        adapter.load_state_dict(torch.load("./models/adapters/user_adapter.pth", map_location=DEVICE))
        print("Loaded: User Adapter")
    adapter.eval()
    return adapter

# --------------------------
# Robustness Evaluation
# --------------------------
def evaluate_challenge(model, adapter, test_data, challenge_type, use_adapter=True):
    """Evaluate model under a specific challenge condition"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    dataset = MPIIGazeDataset(test_data, transform=transform, challenge_type=challenge_type)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=0, collate_fn=collate_fn)
    
    total_mae = 0.0
    total_samples = 0
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Testing {challenge_type}", leave=False):
            if batch is None:
                continue
            
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            dinov2_emb = model.dinov2(pixel_values=x).last_hidden_state[:, 0, :]
            if use_adapter:
                dinov2_emb = adapter(dinov2_emb)
            gaze_pred = model.gaze_head(dinov2_emb)
            
            # Compute MAE
            mae = gaze_angular_error(gaze_pred, y)
            total_mae += mae.item() * x.shape[0]
            total_samples += x.shape[0]
            
            for i in range(x.size(0)):
                err = gaze_angular_error(gaze_pred[i:i+1], y[i:i+1]).item()
                all_errors.append(err)
    
    avg_mae = total_mae / total_samples if total_samples > 0 else 0
    return avg_mae, all_errors

# --------------------------
# Main Robustness Test
# --------------------------
def run_robustness_test():
    print("="*70)
    print("Phase 3 Module 4: Robustness Test")
    print("="*70)
    
    # Create reports directory
    os.makedirs("./reports", exist_ok=True)
    
    # Load model and adapter
    print("\nLoading model...")
    model = load_model()
    adapter = load_adapter()
    
    # Load test data
    print("\nLoading MPIIGaze test data...")
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    test_data = data['test_data']
    
    # Use subset for faster testing (configurable)
    max_samples = min(len(test_data), 5000)  # Use up to 5000 samples
    test_data = test_data[:max_samples]
    print(f"Testing on {len(test_data)} samples")
    
    # Challenge types to test
    challenge_types = {
        "Normal Illumination": "normal",
        "Low Illumination": "low_illumination",
        "High Illumination": "high_illumination",
        "Mild Occlusion (Eyelids)": "mild_occlusion",
        "Glasses Reflection": "glasses",
        "Motion Blur": "motion_blur",
        "Gaussian Noise": "noise"
    }
    
    # Results storage
    robustness_results = []
    normal_mae = None
    
    # Evaluate each challenge
    for challenge_name, challenge_type in challenge_types.items():
        print(f"\n{'='*50}")
        print(f"Challenge: {challenge_name}")
        print(f"{'='*50}")
        
        mae, all_errors = evaluate_challenge(model, adapter, test_data, challenge_type)
        
        # Calculate performance retention
        if normal_mae is None:
            normal_mae = mae
            retention = 100.0
        else:
            retention = round((normal_mae / mae) * 100, 1) if mae > 0 else 100.0
        
        robustness_results.append({
            'Challenge Type': challenge_name,
            'MAE (°)': round(mae, 2),
            'Std (°)': round(np.std(all_errors), 2),
            'Performance Retention (%)': retention,
            'MAE Increase (°)': round(mae - normal_mae, 2) if normal_mae else 0.0
        })
        
        print(f"  MAE: {mae:.2f}°")
        print(f"  Std: {np.std(all_errors):.2f}°")
        print(f"  Performance Retention: {retention}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame(robustness_results)
    
    # Save results
    results_df.to_csv("./reports/robustness_results.csv", index=False)
    
    # Print summary table
    print("\n" + "="*70)
    print("ROBUSTNESS TEST RESULTS")
    print("="*70)
    print(results_df[['Challenge Type', 'MAE (°)', 'Performance Retention (%)']].to_string(index=False))
    print("="*70)
    
    # Analysis
    print("\nRobustness Analysis:")
    challenging_conditions = [r for r in robustness_results if r['Challenge Type'] != "Normal Illumination"]
    avg_retention = np.mean([r['Performance Retention (%)'] for r in challenging_conditions])
    print(f"  Average Performance Retention (across challenges): {avg_retention:.1f}%")
    
    most_robust = max(challenging_conditions, key=lambda x: x['Performance Retention (%)'])
    least_robust = min(challenging_conditions, key=lambda x: x['Performance Retention (%)'])
    print(f"  Most Robust Against: {most_robust['Challenge Type']} ({most_robust['Performance Retention (%)']}% retention)")
    print(f"  Least Robust Against: {least_robust['Challenge Type']} ({least_robust['Performance Retention (%)']}% retention)")
    
    print("\nResults saved to ./reports/robustness_results.csv")
    
    return results_df

if __name__ == "__main__":
    run_robustness_test()
