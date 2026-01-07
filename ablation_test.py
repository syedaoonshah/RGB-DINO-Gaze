"""
Quick Model Evaluation Script
Evaluate all your existing checkpoints without retraining
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import cv2
from transformers import AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
PREPROCESSED_PATH = "./preprocessed_mpiigaze"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def compute_angular_error(pred, target):
    """Compute angular error in degrees"""
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
    cos_sim = torch.clamp(torch.sum(pred_norm * target_norm, dim=1), -1.0, 1.0)
    angle_rad = torch.acos(cos_sim)
    return torch.rad2deg(angle_rad)

class MPIIGazeDataset(Dataset):
    def __init__(self, data):
        self.data = data
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
        image = self.transform(image)
        return image, gaze

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class SimpleGazeModel(nn.Module):
    """Simple wrapper for evaluation"""
    def __init__(self):
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-base")
        self.gaze_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )
    
    def forward(self, x):
        outputs = self.dinov2(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        gaze = self.gaze_head(cls_token)
        return gaze, None

@torch.no_grad()
def evaluate_model(model, test_loader, model_name):
    """Evaluate a model and return MAE"""
    model.eval()
    all_errors = []
    
    print(f"\nEvaluating: {model_name}")
    for batch in tqdm(test_loader, desc="  Processing"):
        if batch is None:
            continue
        
        x, y = batch
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        
        gaze_pred, _ = model(x)
        
        errors = compute_angular_error(gaze_pred, y)
        all_errors.extend(errors.cpu().numpy())
    
    mae = np.mean(all_errors)
    std = np.std(all_errors)
    
    print(f"  MAE: {mae:.2f}°")
    print(f"  Std: {std:.2f}°")
    
    return mae, std

def main():
    print("="*70)
    print("QUICK MODEL EVALUATION (No Training)")
    print("="*70)
    
    # Load data
    print("\nLoading test data...")
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"), weights_only=False)
    
    # Normalize gaze
    for sample in data['test_data']:
        gaze = np.array(sample['gaze'])
        norm = np.linalg.norm(gaze)
        if norm > 1e-8:
            sample['gaze'] = (gaze / norm).tolist()
    
    test_dataset = MPIIGazeDataset(data['test_data'])
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Define models to evaluate
    models_to_eval = [
        {
            "name": "Baseline (Vanilla DINOv2, 100% labels)",
            "path": "./models/dinov2_gaze_baseline.pth",
            "key": "model_state_dict"  # Checkpoint format
        },
        {
            "name": "Semi-Supervised (RGB-MEM + Consistency, 50% labels)",
            "path": "./models/rgb_dino_gaze_simple.pth",
            "key": None  # Direct state dict
        },
        {
            "name": "Pure Self-Supervised (0% labels)",
            "path": "./models/rgb_dino_gaze_pure_selfsup.pth",
            "key": None
        }
    ]
    
    results = []
    
    for model_config in models_to_eval:
        model_path = model_config["path"]
        
        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_config['name']}: File not found")
            continue
        
        # Load model
        print(f"\nLoading {model_config['name']}...")
        model = SimpleGazeModel().to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if model_config["key"] and model_config["key"] in checkpoint:
            state_dict = checkpoint[model_config["key"]]
        else:
            state_dict = checkpoint
        
        # Load weights (strict=False to ignore missing keys)
        model.load_state_dict(state_dict, strict=False)
        
        # Evaluate
        mae, std = evaluate_model(model, test_loader, model_config["name"])
        
        results.append({
            "Model": model_config["name"],
            "MAE": mae,
            "Std": std
        })
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS SUMMARY")
    print("="*70)
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['Model']}")
        print(f"   MAE: {result['MAE']:.2f}° ± {result['Std']:.2f}°")
    
    # Calculate improvements
    if len(results) >= 2:
        baseline_mae = results[0]['MAE']
        print(f"\n{'='*70}")
        print("IMPROVEMENT ANALYSIS")
        print("="*70)
        
        for i in range(1, len(results)):
            improvement = baseline_mae - results[i]['MAE']
            percentage = (improvement / baseline_mae) * 100
            print(f"\n{results[i]['Model']}:")
            print(f"  Improvement over baseline: {improvement:.2f}° ({percentage:.1f}%)")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("./evaluation_results.csv", index=False)
    print(f"\n✓ Results saved to ./evaluation_results.csv")

if __name__ == "__main__":
    main()