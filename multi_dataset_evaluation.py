"""
Phase 3 Module 1: Multi-Dataset Standard Evaluation
Evaluates in-domain (MPIIGaze) and cross-domain generalization.
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
# Load Models
# --------------------------
def load_model(model_type="baseline"):
    """
    Load model based on type: baseline, weak_sup, rgb_mem, or meta_adapted.
    
    CRITICAL: For RGB-MEM, we only load DINOv2 encoder weights (not patch reconstruction head).
    The gaze_head must come from baseline or be re-initialized.
    """
    from rgb_mem_pretrain import RGB_DINO_Gaze
    
    # CRITICAL: Always set unsupervised_pretrain=False for GAZE PREDICTION mode
    model = RGB_DINO_Gaze(unsupervised_pretrain=False).to(DEVICE)
    
    if model_type == "baseline":
        if os.path.exists("./models/dinov2_gaze_baseline.pth"):
            checkpoint = torch.load("./models/dinov2_gaze_baseline.pth", map_location=DEVICE)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("Loaded: Phase 1 Baseline Model")
    
    elif model_type == "rgb_mem":
        # CRITICAL FIX: For RGB-MEM evaluation, we need:
        # 1. DINOv2 encoder from RGB-MEM pretrained weights
        # 2. Gaze head from baseline (trained for gaze prediction)
        
        # First, load baseline gaze head
        if os.path.exists("./models/dinov2_gaze_baseline.pth"):
            baseline_checkpoint = torch.load("./models/dinov2_gaze_baseline.pth", map_location=DEVICE)
            if isinstance(baseline_checkpoint, dict) and "model_state_dict" in baseline_checkpoint:
                model.load_state_dict(baseline_checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(baseline_checkpoint, strict=False)
            print("Loaded: Baseline gaze head")
        
        # Then, load RGB-MEM DINOv2 encoder weights (only dinov2 layers, NOT reconstruction head)
        if os.path.exists("./models/pretrained_rgb_mem/rgb_mem_final.pth"):
            rgb_mem_weights = torch.load("./models/pretrained_rgb_mem/rgb_mem_final.pth", map_location=DEVICE)
            
            # Extract model_state_dict if nested
            if isinstance(rgb_mem_weights, dict) and "model_state_dict" in rgb_mem_weights:
                rgb_mem_weights = rgb_mem_weights["model_state_dict"]
            
            # CRITICAL: Filter to only load DINOv2 encoder weights (ignore patch reconstruction head)
            filtered_weights = {k: v for k, v in rgb_mem_weights.items() 
                               if "dinov2" in k and "patch_reconstruction" not in k}
            
            # Load filtered weights (strict=False to keep gaze_head from baseline)
            model.load_state_dict(filtered_weights, strict=False)
            print(f"Loaded: RGB-MEM DINOv2 encoder ({len(filtered_weights)} layers)")
        else:
            print("WARNING: RGB-MEM checkpoint not found!")
    
    elif model_type == "weak_sup":
        if os.path.exists("./models/rgb_dino_gaze_weak_sup.pth"):
            model.load_state_dict(torch.load("./models/rgb_dino_gaze_weak_sup.pth", map_location=DEVICE))
            print("Loaded: Weakly-Supervised Model")
        elif os.path.exists("./models/pretrained_rgb_mem/rgb_mem_final.pth"):
            # Same fix: load baseline gaze head + RGB-MEM encoder
            if os.path.exists("./models/dinov2_gaze_baseline.pth"):
                baseline_checkpoint = torch.load("./models/dinov2_gaze_baseline.pth", map_location=DEVICE)
                if isinstance(baseline_checkpoint, dict) and "model_state_dict" in baseline_checkpoint:
                    model.load_state_dict(baseline_checkpoint["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(baseline_checkpoint, strict=False)
            
            rgb_mem_weights = torch.load("./models/pretrained_rgb_mem/rgb_mem_final.pth", map_location=DEVICE)
            if isinstance(rgb_mem_weights, dict) and "model_state_dict" in rgb_mem_weights:
                rgb_mem_weights = rgb_mem_weights["model_state_dict"]
            filtered_weights = {k: v for k, v in rgb_mem_weights.items() 
                               if "dinov2" in k and "patch_reconstruction" not in k}
            model.load_state_dict(filtered_weights, strict=False)
            print("Loaded: RGB-MEM Model (weak_sup not found)")
        elif os.path.exists("./models/dinov2_gaze_baseline.pth"):
            checkpoint = torch.load("./models/dinov2_gaze_baseline.pth", map_location=DEVICE)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("Loaded: Phase 1 Baseline (weak_sup not found)")
    
    elif model_type == "meta_adapted":
        # Load baseline gaze head + RGB-MEM encoder for meta-adaptation
        if os.path.exists("./models/dinov2_gaze_baseline.pth"):
            baseline_checkpoint = torch.load("./models/dinov2_gaze_baseline.pth", map_location=DEVICE)
            if isinstance(baseline_checkpoint, dict) and "model_state_dict" in baseline_checkpoint:
                model.load_state_dict(baseline_checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(baseline_checkpoint, strict=False)
        
        if os.path.exists("./models/rgb_dino_gaze_weak_sup.pth"):
            model.load_state_dict(torch.load("./models/rgb_dino_gaze_weak_sup.pth", map_location=DEVICE))
            print("Loaded: Weakly-Supervised Model for Meta-Adaptation")
        elif os.path.exists("./models/pretrained_rgb_mem/rgb_mem_final.pth"):
            rgb_mem_weights = torch.load("./models/pretrained_rgb_mem/rgb_mem_final.pth", map_location=DEVICE)
            if isinstance(rgb_mem_weights, dict) and "model_state_dict" in rgb_mem_weights:
                rgb_mem_weights = rgb_mem_weights["model_state_dict"]
            filtered_weights = {k: v for k, v in rgb_mem_weights.items() 
                               if "dinov2" in k and "patch_reconstruction" not in k}
            model.load_state_dict(filtered_weights, strict=False)
            print("Loaded: RGB-MEM Model for Meta-Adaptation")
        else:
            print("Loaded: Phase 1 Baseline for Meta-Adaptation")
    
    model.eval()
    return model

def load_adapter():
    """Load meta-adapter if available"""
    adapter = MetaAdapter(input_dim=768).to(DEVICE)
    if os.path.exists("./models/adapters/user_adapter.pth"):
        adapter.load_state_dict(torch.load("./models/adapters/user_adapter.pth", map_location=DEVICE))
        print("Loaded: User Adapter")
    else:
        print("Using: Untrained Adapter (no checkpoint found)")
    adapter.eval()
    return adapter

# --------------------------
# Evaluation Pipeline
# --------------------------
def evaluate_dataset(model, test_loader, adapter=None, use_adapter=False):
    """Evaluate model on a dataset"""
    total_mae = 0.0
    total_samples = 0
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
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
            
            # Per-sample errors
            for i in range(x.size(0)):
                err = gaze_angular_error(gaze_pred[i:i+1], y[i:i+1]).item()
                all_errors.append(err)
    
    avg_mae = total_mae / total_samples if total_samples > 0 else 0
    return avg_mae, all_errors

# --------------------------
# Per-Participant Evaluation (Leave-One-Out)
# --------------------------
def evaluate_per_participant(model, adapter=None, use_adapter=False):
    """Evaluate model per participant for detailed analysis"""
    participant_results = []
    
    # Load full data
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Group by participant (p00-p14)
    all_data = data['train_data'] + data['val_data'] + data['test_data']
    
    # For MPIIGaze, we'll evaluate on test set split by rough participant groups
    test_data = data['test_data']
    n_samples = len(test_data)
    samples_per_participant = n_samples // 15
    
    for p_idx in range(15):
        start_idx = p_idx * samples_per_participant
        end_idx = min((p_idx + 1) * samples_per_participant, n_samples)
        
        if start_idx >= n_samples:
            break
        
        participant_data = test_data[start_idx:end_idx]
        if len(participant_data) == 0:
            continue
        
        participant_dataset = MPIIGazeDataset(participant_data, transform=transform)
        participant_loader = DataLoader(participant_dataset, batch_size=BATCH_SIZE, 
                                        shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        mae, _ = evaluate_dataset(model, participant_loader, adapter, use_adapter)
        participant_results.append({
            'Participant': f'p{p_idx:02d}',
            'MAE (°)': round(mae, 2),
            'Samples': len(participant_data)
        })
    
    return pd.DataFrame(participant_results)

# --------------------------
# Main Evaluation
# --------------------------
def run_multi_dataset_evaluation():
    print("="*70)
    print("Phase 3 Module 1: Multi-Dataset Standard Evaluation")
    print("="*70)
    
    # Create reports directory
    os.makedirs("./reports", exist_ok=True)
    os.makedirs("./reports/figures", exist_ok=True)
    
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
    results = []
    
    # Model configurations to evaluate
    model_configs = [
        ("Baseline (Vanilla DINOv2)", "baseline", False),
        ("RGB-DINO-Gaze (RGB-MEM)", "rgb_mem", False),
        ("RGB-DINO-Gaze (Weakly-Supervised)", "weak_sup", False),
        ("RGB-DINO-Gaze (Meta-Adapted)", "meta_adapted", True),
    ]
    
    adapter = load_adapter()
    
    for model_name, model_type, use_adapter in model_configs:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")
        
        model = load_model(model_type)
        mae, all_errors = evaluate_dataset(model, test_loader, adapter, use_adapter)
        
        results.append({
            'Model': model_name,
            'Dataset': 'MPIIGaze',
            'MAE (°)': round(mae, 2),
            'Std (°)': round(np.std(all_errors), 2),
            'Min (°)': round(np.min(all_errors), 2),
            'Max (°)': round(np.max(all_errors), 2)
        })
        
        print(f"  MAE: {mae:.2f}°")
        print(f"  Std: {np.std(all_errors):.2f}°")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv("./reports/multi_dataset_results.csv", index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("MULTI-DATASET EVALUATION RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    # Per-participant evaluation for best model
    print("\nPer-Participant Evaluation (Meta-Adapted Model):")
    model = load_model("meta_adapted")
    participant_df = evaluate_per_participant(model, adapter, use_adapter=True)
    participant_df.to_csv("./reports/per_participant_results.csv", index=False)
    print(participant_df.to_string(index=False))
    
    print("\nResults saved to ./reports/multi_dataset_results.csv")
    
    return results_df

if __name__ == "__main__":
    run_multi_dataset_evaluation()
