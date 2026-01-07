"""
Test gaze prediction on images in D:\DINO\dat folder.
Images have ground-truth gaze coordinates in filename format: x_y_timestamp.png
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from glob import glob

# --------------------------
# Configuration
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
DAT_FOLDER = r"D:\DINO\dat"
OUTPUT_FOLDER = r"D:\DINO\dat_results"

# Preprocessing (match training)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])

# --------------------------
# Load Model
# --------------------------
def load_model():
    from rgb_mem_pretrain import RGB_DINO_Gaze
    
    model = RGB_DINO_Gaze(unsupervised_pretrain=False).to(DEVICE)
    
    if os.path.exists("./models/rgb_dino_gaze_weak_sup.pth"):
        model.load_state_dict(torch.load("./models/rgb_dino_gaze_weak_sup.pth", map_location=DEVICE))
        print("Loaded Phase 2 weakly-supervised model")
    elif os.path.exists("./models/dinov2_gaze_baseline.pth"):
        checkpoint = torch.load("./models/dinov2_gaze_baseline.pth", map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Loaded Phase 1 baseline model")
    else:
        print("WARNING: No trained model found!")
    
    model.eval()
    return model

# --------------------------
# Parse filename for ground truth
# --------------------------
def parse_filename_gt(filename):
    """
    Parse ground truth from filename.
    Format: x_y_timestamp.png or x.xxx_y.yyy_timestamp.png
    Returns (x, y) coordinates or None if parsing fails.
    """
    try:
        basename = os.path.splitext(filename)[0]
        parts = basename.split('_')
        
        # Try to parse first two parts as coordinates
        x = float(parts[0])
        y = float(parts[1])
        return (x, y)
    except (ValueError, IndexError):
        return None

# --------------------------
# Visualize with arrows
# --------------------------
def visualize_gaze(image, pred_gaze, gt_coords=None, save_path=None):
    """
    Visualize gaze prediction on image.
    - Blue arrow: predicted gaze
    - Green arrow: ground truth (if available)
    """
    fig, (ax_img, ax_footer) = plt.subplots(
        2, 1, figsize=(8, 9),
        gridspec_kw={'height_ratios': [10, 1]}
    )
    
    # Show image
    ax_img.imshow(image)
    ax_img.axis('off')
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    arrow_len = min(w, h) // 3
    
    # Draw predicted gaze arrow (blue)
    pred_x = center[0] + pred_gaze[0] * arrow_len
    pred_y = center[1] - pred_gaze[1] * arrow_len
    ax_img.annotate('', xy=(pred_x, pred_y), xytext=center,
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    # Black footer
    ax_footer.set_facecolor('black')
    ax_footer.axis('off')
    
    # Footer text
    pred_text = f"Predicted: (x={pred_gaze[0]:.3f}, y={pred_gaze[1]:.3f}, z={pred_gaze[2]:.3f})"
    if gt_coords:
        gt_text = f"GT coords: ({gt_coords[0]:.2f}, {gt_coords[1]:.2f})"
        footer_text = f"{gt_text}  |  {pred_text}"
    else:
        footer_text = pred_text
    
    ax_footer.text(0.01, 0.5, footer_text, color='white', fontsize=9,
                   verticalalignment='center', transform=ax_footer.transAxes)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Predicted')]
    ax_img.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()
    
    return fig

# --------------------------
# Main test function
# --------------------------
def test_dat_images():
    print("="*60)
    print("Testing Gaze Model on D:\\DINO\\dat images")
    print("="*60)
    
    # Load model
    model = load_model()
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get all images
    image_files = glob(os.path.join(DAT_FOLDER, "*.png"))
    print(f"Found {len(image_files)} images")
    
    results = []
    
    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        print(f"\n[{i+1}/{len(image_files)}] Processing: {filename}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ERROR: Could not load image")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        img_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            pred_gaze, _ = model(img_tensor)
            pred_gaze = pred_gaze.squeeze().cpu().numpy()
        
        # Parse ground truth from filename
        gt_coords = parse_filename_gt(filename)
        
        print(f"  Predicted gaze: ({pred_gaze[0]:.3f}, {pred_gaze[1]:.3f}, {pred_gaze[2]:.3f})")
        if gt_coords:
            print(f"  GT coordinates: ({gt_coords[0]:.2f}, {gt_coords[1]:.2f})")
        
        # Save visualization
        save_path = os.path.join(OUTPUT_FOLDER, f"result_{i+1:03d}.png")
        visualize_gaze(image_rgb, pred_gaze, gt_coords, save_path)
        
        results.append({
            'filename': filename,
            'pred_gaze': pred_gaze,
            'gt_coords': gt_coords
        })
    
    print("\n" + "="*60)
    print(f"DONE! Processed {len(results)} images")
    print(f"Results saved to: {OUTPUT_FOLDER}")
    print("="*60)
    
    # Summary statistics
    print("\nGaze Prediction Statistics:")
    pred_x = [r['pred_gaze'][0] for r in results]
    pred_y = [r['pred_gaze'][1] for r in results]
    pred_z = [r['pred_gaze'][2] for r in results]
    
    print(f"  X: min={min(pred_x):.3f}, max={max(pred_x):.3f}, mean={np.mean(pred_x):.3f}")
    print(f"  Y: min={min(pred_y):.3f}, max={max(pred_y):.3f}, mean={np.mean(pred_y):.3f}")
    print(f"  Z: min={min(pred_z):.3f}, max={max(pred_z):.3f}, mean={np.mean(pred_z):.3f}")
    
    return results

if __name__ == "__main__":
    test_dat_images()
