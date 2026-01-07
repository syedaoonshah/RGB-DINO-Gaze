import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# --------------------------
# Global Config
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER_BOTTLENECK = 64
BATCH_SIZE = 32
PREPROCESSED_PATH = "./preprocessed_mpiigaze"

# RGB normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --------------------------
# 1. Meta-Adapter Class
# --------------------------
class MetaAdapter(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, ADAPTER_BOTTLENECK),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ADAPTER_BOTTLENECK, input_dim)
        )

    def forward(self, x):
        return x + self.adapter(x)

# --------------------------
# 2. Visualization Functions
# --------------------------
def visualize_gaze(image, pred_gaze, gt_gaze=None, figsize=(8, 8)):
    """
    Visualize gaze direction on an image:
    - Draws arrows for predicted (blue) and ground-truth (green) gaze.
    - Adds a black footer with coordinates (no text over the image).
    
    Args:
        image: RGB image (numpy array, shape [H, W, 3]).
        pred_gaze: Predicted gaze vector (x, y, z) or (pitch, yaw).
        gt_gaze: Optional ground-truth gaze vector (same format as pred_gaze).
        figsize: Size of the output figure.
    """
    # Create figure with image + black footer
    fig, (ax_img, ax_footer) = plt.subplots(
        2, 1, figsize=figsize, 
        gridspec_kw={'height_ratios': [10, 1]}
    )
    
    # Plot image (no text overlay)
    ax_img.imshow(image)
    ax_img.axis('off')
    
    # Get image center (start point for arrows)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Arrow length (scaled by image size)
    arrow_len = min(w, h) // 4
    
    # Draw predicted gaze arrow (blue)
    pred_x = center[0] + pred_gaze[0] * arrow_len
    pred_y = center[1] - pred_gaze[1] * arrow_len  # Y-axis inverted
    ax_img.arrow(
        center[0], center[1], pred_x - center[0], pred_y - center[1],
        color='blue', width=2, head_width=8, label='Predicted'
    )
    
    # Draw ground-truth gaze arrow (green) if provided
    if gt_gaze is not None:
        gt_x = center[0] + gt_gaze[0] * arrow_len
        gt_y = center[1] - gt_gaze[1] * arrow_len
        ax_img.arrow(
            center[0], center[1], gt_x - center[0], gt_y - center[1],
            color='green', width=2, head_width=8, label='Ground Truth'
        )
        ax_img.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    
    # Black footer for coordinates
    ax_footer.set_facecolor('black')
    ax_footer.axis('off')
    
    # Add coordinates to footer
    pred_text = f"Pred: (x={pred_gaze[0]:.2f}, y={pred_gaze[1]:.2f}, z={pred_gaze[2]:.2f})"
    if gt_gaze is not None:
        gt_text = f"GT:  (x={gt_gaze[0]:.2f}, y={gt_gaze[1]:.2f}, z={gt_gaze[2]:.2f})"
        footer_text = f"{gt_text} | {pred_text}"
    else:
        footer_text = pred_text
    
    ax_footer.text(
        0.01, 0.5, footer_text, 
        color='white', fontsize=10, 
        verticalalignment='center', horizontalalignment='left'
    )
    
    plt.tight_layout()
    return fig

def infer_and_visualize(image, model, adapter, gt_gaze=None):
    """
    Run inference on an image tensor and generate visualization.
    
    Args:
        image: Preprocessed image tensor (1, 3, 224, 224).
        model: Trained RGB-DINO-Gaze model.
        adapter: Personalized meta-adapter.
        gt_gaze: Optional ground-truth gaze vector.
    
    Returns:
        fig: Matplotlib figure with visualization.
        pred_gaze: Predicted gaze vector (numpy array).
    """
    model.eval()
    with torch.no_grad():
        img_tensor = image.unsqueeze(0).to(DEVICE) if image.dim() == 3 else image.to(DEVICE)
        dinov2_emb = model.dinov2(pixel_values=img_tensor).last_hidden_state[:, 0, :]
        adapted_emb = adapter(dinov2_emb)
        pred_gaze = model.gaze_head(adapted_emb).squeeze().cpu().numpy()
    
    # Convert tensor to numpy for visualization
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = image.cpu() * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)
    img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Generate visualization
    gt_gaze_np = gt_gaze.cpu().numpy() if isinstance(gt_gaze, torch.Tensor) else gt_gaze
    fig = visualize_gaze(img_np, pred_gaze, gt_gaze_np)
    
    return fig, pred_gaze

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
# 4. Gaze Angular Error
# --------------------------
def gaze_angular_error(pred, target):
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
    cos_sim = torch.clamp(torch.sum(pred_norm * target_norm, dim=1), -1.0, 1.0)
    angle_rad = torch.acos(cos_sim)
    angle_deg = torch.rad2deg(angle_rad)
    return torch.mean(angle_deg)

# --------------------------
# 5. Run Inference on New Images
# --------------------------
def run_new_image_inference(new_image_paths, gt_gazes=None):
    """
    Process a list of new images and save visualizations.
    
    Args:
        new_image_paths: List of paths to new images.
        gt_gazes: Optional list of ground-truth gaze vectors.
    """
    from rgb_mem_pretrain import RGB_DINO_Gaze
    
    # Initialize model
    print("Loading model...")
    model = RGB_DINO_Gaze(unsupervised_pretrain=False).to(DEVICE)
    
    if os.path.exists("./models/rgb_dino_gaze_weak_sup.pth"):
        model.load_state_dict(torch.load("./models/rgb_dino_gaze_weak_sup.pth"))
        print("Loaded weakly-supervised model")
    elif os.path.exists("./models/dinov2_gaze_baseline.pth"):
        checkpoint = torch.load("./models/dinov2_gaze_baseline.pth")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Loaded Phase 1 baseline model")
    else:
        print("Warning: No trained model found!")
    
    model.eval()
    
    # Initialize adapter
    adapter = MetaAdapter(input_dim=768).to(DEVICE)
    
    # Define preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create output directory
    os.makedirs("./gaze_visualizations", exist_ok=True)
    
    # Process each image
    for i, img_path in enumerate(new_image_paths):
        print(f"Processing: {img_path}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = transform(image_rgb)
        
        gt_gaze = gt_gazes[i] if (gt_gazes is not None and i < len(gt_gazes)) else None
        
        fig, pred_gaze = infer_and_visualize(img_tensor, model, adapter, gt_gaze)
        
        # Save visualization
        output_path = f"./gaze_visualizations/result_{i+1}.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {output_path}")
        print(f"Predicted gaze: {pred_gaze}")
        plt.close(fig)

# --------------------------
# 6. Visualize Test Set Samples
# --------------------------
def visualize_test_samples(num_samples=10):
    """
    Visualize predictions on random test set samples with ground truth comparison.
    """
    from rgb_mem_pretrain import RGB_DINO_Gaze
    
    # Load model
    print("Loading model...")
    model = RGB_DINO_Gaze(unsupervised_pretrain=False).to(DEVICE)
    
    if os.path.exists("./models/rgb_dino_gaze_weak_sup.pth"):
        model.load_state_dict(torch.load("./models/rgb_dino_gaze_weak_sup.pth"))
        print("Loaded weakly-supervised model")
    elif os.path.exists("./models/dinov2_gaze_baseline.pth"):
        checkpoint = torch.load("./models/dinov2_gaze_baseline.pth")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Loaded Phase 1 baseline model")
    
    model.eval()
    
    # Initialize adapter
    adapter = MetaAdapter(input_dim=768).to(DEVICE)
    
    # Load test data
    print(f"Loading test data...")
    data = torch.load(os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    test_dataset = MPIIGazeDataset(data['test_data'], transform=transform)
    
    # Create output directory
    os.makedirs("./gaze_visualizations", exist_ok=True)
    
    # Random sample indices
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    print(f"Visualizing {len(indices)} samples...")
    
    total_error = 0.0
    
    for i, idx in enumerate(indices):
        img_tensor, gt_gaze = test_dataset[idx]
        
        fig, pred_gaze = infer_and_visualize(img_tensor, model, adapter, gt_gaze)
        
        # Compute angular error
        pred_tensor = torch.tensor(pred_gaze).unsqueeze(0)
        gt_tensor = gt_gaze.unsqueeze(0)
        error = gaze_angular_error(pred_tensor, gt_tensor).item()
        total_error += error
        
        # Save visualization
        output_path = f"./gaze_visualizations/test_sample_{i+1}_error_{error:.2f}deg.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Sample {i+1}: Angular Error = {error:.2f}°")
        plt.close(fig)
    
    avg_error = total_error / len(indices)
    print(f"\nAverage Angular Error: {avg_error:.2f}°")
    
    return avg_error

# --------------------------
# 7. Full Evaluation with Visualizations
# --------------------------
def evaluate_with_visualizations(num_vis_samples=10):
    """
    Run full evaluation on test set and generate visualizations for subset.
    """
    from rgb_mem_pretrain import RGB_DINO_Gaze
    
    print("="*60)
    print("RGB-DINO-Gaze Evaluation with Visualizations")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = RGB_DINO_Gaze(unsupervised_pretrain=False).to(DEVICE)
    
    model_name = "No model"
    if os.path.exists("./models/rgb_dino_gaze_weak_sup.pth"):
        model.load_state_dict(torch.load("./models/rgb_dino_gaze_weak_sup.pth"))
        model_name = "Weakly-supervised model"
    elif os.path.exists("./models/dinov2_gaze_baseline.pth"):
        checkpoint = torch.load("./models/dinov2_gaze_baseline.pth")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model_name = "Phase 1 baseline model"
    
    print(f"Using: {model_name}")
    model.eval()
    
    # Initialize adapter
    adapter = MetaAdapter(input_dim=768).to(DEVICE)
    
    # Load test data
    print("Loading test data...")
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
    
    # Full evaluation
    print("\nRunning full evaluation...")
    total_error = 0.0
    total_samples = 0
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if batch is None:
                continue
            
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            dinov2_emb = model.dinov2(pixel_values=x).last_hidden_state[:, 0, :]
            adapted_emb = adapter(dinov2_emb)
            pred = model.gaze_head(adapted_emb)
            
            # Per-sample errors
            for i in range(x.size(0)):
                error = gaze_angular_error(pred[i:i+1], y[i:i+1]).item()
                all_errors.append(error)
            
            batch_error = gaze_angular_error(pred, y).item()
            total_error += batch_error * x.size(0)
            total_samples += x.size(0)
    
    avg_error = total_error / total_samples
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Test Samples: {total_samples}")
    print(f"Mean Angular Error: {avg_error:.2f}°")
    print(f"Std Angular Error: {np.std(all_errors):.2f}°")
    print(f"Min Angular Error: {np.min(all_errors):.2f}°")
    print(f"Max Angular Error: {np.max(all_errors):.2f}°")
    print(f"{'='*60}")
    
    # Generate visualizations
    print(f"\nGenerating {num_vis_samples} visualizations...")
    os.makedirs("./gaze_visualizations", exist_ok=True)
    
    indices = np.random.choice(len(test_dataset), min(num_vis_samples, len(test_dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        img_tensor, gt_gaze = test_dataset[idx]
        fig, pred_gaze = infer_and_visualize(img_tensor, model, adapter, gt_gaze)
        
        pred_tensor = torch.tensor(pred_gaze).unsqueeze(0)
        gt_tensor = gt_gaze.unsqueeze(0)
        error = gaze_angular_error(pred_tensor, gt_tensor).item()
        
        output_path = f"./gaze_visualizations/eval_sample_{i+1}_error_{error:.2f}deg.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    print(f"Visualizations saved to ./gaze_visualizations/")
    
    # Save results
    os.makedirs("./reports", exist_ok=True)
    with open("./reports/evaluation_results.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("RGB-DINO-Gaze Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Samples: {total_samples}\n")
        f.write(f"Mean Angular Error: {avg_error:.2f}°\n")
        f.write(f"Std Angular Error: {np.std(all_errors):.2f}°\n")
        f.write(f"Min Angular Error: {np.min(all_errors):.2f}°\n")
        f.write(f"Max Angular Error: {np.max(all_errors):.2f}°\n")
    
    print("Results saved to ./reports/evaluation_results.txt")
    
    return avg_error

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RGB-DINO-Gaze Visualization and Evaluation")
    parser.add_argument("--mode", type=str, default="evaluate", 
                        choices=["evaluate", "visualize", "infer"],
                        help="Mode: evaluate (full eval), visualize (test samples), infer (new images)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to visualize")
    parser.add_argument("--images", nargs="+", default=None,
                        help="Paths to new images for inference mode")
    
    args = parser.parse_args()
    
    if args.mode == "evaluate":
        evaluate_with_visualizations(num_vis_samples=args.num_samples)
    elif args.mode == "visualize":
        visualize_test_samples(num_samples=args.num_samples)
    elif args.mode == "infer":
        if args.images:
            run_new_image_inference(args.images)
        else:
            print("Please provide image paths with --images")
