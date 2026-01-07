"""
Phase 3 Module 2: Ablation Study
Quantifies the contribution of RGB-MEM, Dual Self-Supervision, and Meta-Adapter.
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

# Ablation variants configuration
ABLATION_VARIANTS = {
    "Baseline (Vanilla DINOv2)": {
        "rgb_mem": False, 
        "dual_loss": False, 
        "adapter": False,
        "description": "DINOv2 backbone + gaze head (no domain-specific training)"
    },
    "Baseline + RGB-MEM": {
        "rgb_mem": True, 
        "dual_loss": False, 
        "adapter": False,
        "description": "RGB-MEM unsupervised pretraining on eye regions"
    },
    "Baseline + RGB-MEM + Dual Loss": {
        "rgb_mem": True, 
        "dual_loss": True, 
        "adapter": False,
        "description": "Weakly-supervised with temporal + geometry loss"
    },
    "Full Model (RGB-MEM + Dual Loss + Adapter)": {
        "rgb_mem": True, 
        "dual_loss": True, 
        "adapter": True,
        "description": "Complete model with MAML meta-adaptation"
    }
}

# --------------------------
# Dataset Class
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
        
        eye_crop = cv2.resize(image, (64, 64))
        
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
# Load Model for Ablation
# --------------------------
def load_ablation_model(variant_config):
    """
    Load model based on ablation configuration.
    
    CRITICAL FIX: For RGB-MEM variants, we load:
    1. Baseline gaze_head (trained for gaze prediction)
    2. RGB-MEM DINOv2 encoder weights (only dinov2 layers, NOT patch reconstruction head)
    """
    from rgb_mem_pretrain import RGB_DINO_Gaze
    
    # CRITICAL: Always set unsupervised_pretrain=False for GAZE PREDICTION mode
    model = RGB_DINO_Gaze(unsupervised_pretrain=False).to(DEVICE)
    
    # First, ALWAYS load baseline to get the trained gaze_head
    if os.path.exists("./models/dinov2_gaze_baseline.pth"):
        baseline_checkpoint = torch.load("./models/dinov2_gaze_baseline.pth", map_location=DEVICE)
        if isinstance(baseline_checkpoint, dict) and "model_state_dict" in baseline_checkpoint:
            model.load_state_dict(baseline_checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(baseline_checkpoint, strict=False)
        print("  Loaded: Baseline gaze_head")
    
    # Then, conditionally load additional weights based on variant config
    if variant_config["dual_loss"]:
        # Load weakly-supervised model if available
        if os.path.exists("./models/rgb_dino_gaze_weak_sup.pth"):
            model.load_state_dict(torch.load("./models/rgb_dino_gaze_weak_sup.pth", map_location=DEVICE))
            print("  Loaded: Weakly-supervised checkpoint (full)")
        elif variant_config["rgb_mem"] and os.path.exists("./models/pretrained_rgb_mem/rgb_mem_final.pth"):
            # Load RGB-MEM encoder on top of baseline gaze_head
            rgb_mem_weights = torch.load("./models/pretrained_rgb_mem/rgb_mem_final.pth", map_location=DEVICE)
            if isinstance(rgb_mem_weights, dict) and "model_state_dict" in rgb_mem_weights:
                rgb_mem_weights = rgb_mem_weights["model_state_dict"]
            # CRITICAL: Filter to only DINOv2 layers (not patch reconstruction head)
            filtered_weights = {k: v for k, v in rgb_mem_weights.items() 
                               if "dinov2" in k and "patch_reconstruction" not in k}
            model.load_state_dict(filtered_weights, strict=False)
            print(f"  Loaded: RGB-MEM encoder ({len(filtered_weights)} layers) + Baseline gaze_head")
    
    elif variant_config["rgb_mem"]:
        # Load RGB-MEM encoder on top of baseline gaze_head
        if os.path.exists("./models/pretrained_rgb_mem/rgb_mem_final.pth"):
            rgb_mem_weights = torch.load("./models/pretrained_rgb_mem/rgb_mem_final.pth", map_location=DEVICE)
            if isinstance(rgb_mem_weights, dict) and "model_state_dict" in rgb_mem_weights:
                rgb_mem_weights = rgb_mem_weights["model_state_dict"]
            # CRITICAL: Filter to only DINOv2 layers (not patch reconstruction head)
            filtered_weights = {k: v for k, v in rgb_mem_weights.items() 
                               if "dinov2" in k and "patch_reconstruction" not in k}
            model.load_state_dict(filtered_weights, strict=False)
            print(f"  Loaded: RGB-MEM encoder ({len(filtered_weights)} layers) + Baseline gaze_head")
        else:
            print("  Using: Baseline only (RGB-MEM not found)")
    else:
        # Pure baseline - already loaded above
        print("  Using: Phase 1 Baseline checkpoint")
    
    model.eval()
    return model

def load_adapter():
    """Load meta-adapter"""
    adapter = MetaAdapter(input_dim=768).to(DEVICE)
    if os.path.exists("./models/adapters/user_adapter.pth"):
        adapter.load_state_dict(torch.load("./models/adapters/user_adapter.pth", map_location=DEVICE))
        print("  Loaded: User Adapter")
    adapter.eval()
    return adapter

# --------------------------
# Ablation Evaluation
# --------------------------
def evaluate_ablation_variant(model, test_loader, adapter=None, use_adapter=False):
    """Evaluate a single ablation variant"""
    total_mae = 0.0
    total_samples = 0
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Evaluating", leave=False):
            if batch is None:
                continue
            
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            dinov2_emb = model.dinov2(pixel_values=x).last_hidden_state[:, 0, :]
            if use_adapter and adapter is not None:
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
# Main Ablation Study
# --------------------------
def run_ablation_study():
    print("="*70)
    print("Phase 3 Module 2: Ablation Study")
    print("="*70)
    
    # Create reports directory
    os.makedirs("./reports", exist_ok=True)
    
    # Load test data
    print("\nLoading MPIIGaze test data...")
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    test_dataset = MPIIGazeDataset(data['test_data'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=0, collate_fn=collate_fn)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Results storage
    ablation_results = []
    baseline_mae = None
    
    # Load adapter once
    adapter = load_adapter()
    
    # Evaluate each variant
    for variant_name, config in ABLATION_VARIANTS.items():
        print(f"\n{'='*50}")
        print(f"Variant: {variant_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*50}")
        
        model = load_ablation_model(config)
        use_adapter = config["adapter"]
        
        mae, all_errors = evaluate_ablation_variant(model, test_loader, adapter, use_adapter)
        
        # Calculate reduction vs baseline
        if baseline_mae is None:
            baseline_mae = mae
            reduction = 0.0
        else:
            reduction = round((baseline_mae - mae) / baseline_mae * 100, 1)
        
        ablation_results.append({
            'Model Variant': variant_name,
            'MAE (°)': round(mae, 2),
            'Std (°)': round(np.std(all_errors), 2),
            'MAE Reduction vs Baseline (%)': reduction,
            'Description': config['description']
        })
        
        print(f"  MAE: {mae:.2f}°")
        print(f"  Std: {np.std(all_errors):.2f}°")
        print(f"  Reduction vs Baseline: {reduction}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame(ablation_results)
    
    # Save results
    results_df.to_csv("./reports/ablation_results.csv", index=False)
    
    # Print summary table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(results_df[['Model Variant', 'MAE (°)', 'MAE Reduction vs Baseline (%)']].to_string(index=False))
    print("="*70)
    
    # Component contribution analysis
    print("\nComponent Contribution Analysis:")
    if len(ablation_results) >= 4:
        rgb_mem_contrib = ablation_results[0]['MAE (°)'] - ablation_results[1]['MAE (°)']
        dual_loss_contrib = ablation_results[1]['MAE (°)'] - ablation_results[2]['MAE (°)']
        adapter_contrib = ablation_results[2]['MAE (°)'] - ablation_results[3]['MAE (°)']
        
        print(f"  RGB-MEM Contribution: -{rgb_mem_contrib:.2f}° ({ablation_results[1]['MAE Reduction vs Baseline (%)']}% reduction)")
        print(f"  Dual Loss Contribution: -{dual_loss_contrib:.2f}°")
        print(f"  Meta-Adapter Contribution: -{adapter_contrib:.2f}°")
        print(f"  Total Improvement: -{ablation_results[0]['MAE (°)'] - ablation_results[3]['MAE (°)']:.2f}° ({ablation_results[3]['MAE Reduction vs Baseline (%)']}% reduction)")
    
    print("\nResults saved to ./reports/ablation_results.csv")
    
    return results_df

if __name__ == "__main__":
    run_ablation_study()
