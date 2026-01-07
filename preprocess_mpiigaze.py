import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.io as sio
import glob

# --------------------------
# 1. Configuration
# --------------------------
DEVICE = "cpu"  # MediaPipe works on CPU (not needed for normalized MPIIGaze)
EYE_CROP_SIZE = (64, 64)
FINAL_INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
MPIIGAZE_PATH = "./MPIIGaze"
PREPROCESSED_PATH = "./preprocessed_mpiigaze"

# Note: MPIIGaze "Normalized" data already contains cropped eye regions,
# so we skip MediaPipe face detection for efficiency

# RGB normalization (ImageNet stats for DINOv2)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --------------------------
# 2. Load MPIIGaze Dataset
# --------------------------
def load_mpiigaze_data():
    """Load MPIIGaze normalized data from .mat files."""
    data_samples = []
    normalized_path = os.path.join(MPIIGAZE_PATH, "Data", "Normalized")
    
    # Get all participant folders (p00, p01, ...)
    participant_folders = sorted([f for f in os.listdir(normalized_path) 
                                  if os.path.isdir(os.path.join(normalized_path, f))])
    
    print(f"Found {len(participant_folders)} participants")
    
    for participant in tqdm(participant_folders, desc="Loading participants"):
        participant_path = os.path.join(normalized_path, participant)
        # Get all day files (day01.mat, day02.mat, ...)
        day_files = sorted(glob.glob(os.path.join(participant_path, "day*.mat")))
        
        for day_file in day_files:
            try:
                # Load .mat file
                mat_data = sio.loadmat(day_file)
                
                # MPIIGaze structure: mat_data['data'][0,0]['left'] or ['right']
                # Each contains: ['gaze'], ['image'], ['pose']
                if 'data' in mat_data:
                    data_struct = mat_data['data'][0, 0]
                    
                    # Extract left eye data (primary for gaze estimation)
                    left_data = data_struct['left'][0, 0]
                    
                    # Get arrays
                    images = left_data['image']  # Shape: (N, H, W, 3)
                    gazes = left_data['gaze']    # Shape: (N, 2) or (N, 3)
                    
                    # Add samples
                    for i in range(len(images)):
                        # Convert gaze to 3D if it's 2D (add z-component)
                        gaze = gazes[i]
                        if len(gaze) == 2:
                            # Convert 2D gaze angles to 3D vector
                            theta, phi = gaze
                            gaze = np.array([
                                -np.cos(phi) * np.sin(theta),
                                -np.sin(phi),
                                -np.cos(phi) * np.cos(theta)
                            ])
                        
                        data_samples.append({
                            'image': images[i],
                            'gaze': gaze,
                            'participant': participant
                        })
            except Exception as e:
                print(f"\nError loading {day_file}: {e}")
                continue
    
    print(f"\nLoaded {len(data_samples)} total samples")
    return data_samples

# --------------------------
# 3. Illumination Enhancement (CLAHE)
# --------------------------
def enhance_illumination(rgb_image):
    """Apply CLAHE to enhance illumination."""
    # Convert to LAB color space for CLAHE (preserve color)
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return rgb_clahe

# --------------------------
# 4. Custom Dataset Class
# --------------------------
class MPIIGazeDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get image (already normalized from MPIIGaze dataset)
        image = sample['image']
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            # Assume image is in [0, 1] or [-1, 1] range
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Ensure RGB format
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[0] == 3:  # (C, H, W) format
            image = image.transpose(1, 2, 0)  # Convert to (H, W, C)
        
        # MPIIGaze normalized images are already cropped eye regions
        # So we skip MediaPipe face detection and use the image directly
        eye_crop = cv2.resize(image, EYE_CROP_SIZE)
        
        # Enhance illumination
        eye_crop = enhance_illumination(eye_crop)
        
        # Apply transforms
        if self.transform:
            eye_crop = self.transform(eye_crop)
        
        # Gaze label (normalized 3D vector)
        gaze_label = torch.tensor(sample['gaze'], dtype=torch.float32)
        
        return eye_crop, gaze_label

# --------------------------
# 5. Collate function to filter None samples
# --------------------------
def collate_fn(batch):
    """Filter out None samples from batch."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --------------------------
# 6. Main Preprocessing Pipeline
# --------------------------
def main():
    # Step 1: Load data
    print("Loading MPIIGaze dataset...")
    data_samples = load_mpiigaze_data()
    
    if len(data_samples) == 0:
        print("ERROR: No data loaded. Please check the MPIIGaze dataset path.")
        return
    
    # Step 2: Split data (stratified by participant)
    print("Splitting data into train/val/test...")
    participants = [s['participant'] for s in data_samples]
    
    train_data, test_val_data = train_test_split(
        data_samples, test_size=0.2, stratify=participants, random_state=42
    )
    test_val_participants = [s['participant'] for s in test_val_data]
    val_data, test_data = train_test_split(
        test_val_data, test_size=0.5, stratify=test_val_participants, random_state=42
    )
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Step 3: Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(FINAL_INPUT_SIZE),
        transforms.ToTensor(),
        normalize
    ])
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(FINAL_INPUT_SIZE),
        transforms.ToTensor(),
        normalize
    ])
    
    # Step 4: Create datasets
    print("Creating PyTorch datasets...")
    train_dataset = MPIIGazeDataset(train_data, transform=train_transform)
    val_dataset = MPIIGazeDataset(val_data, transform=val_test_transform)
    test_dataset = MPIIGazeDataset(test_data, transform=val_test_transform)
    
    # Step 5: Create DataLoaders
    print("Creating DataLoaders...")
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
    
    # Step 6: Save DataLoaders for later use
    os.makedirs(PREPROCESSED_PATH, exist_ok=True)
    
    print("Saving preprocessed data...")
    torch.save({
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }, os.path.join(PREPROCESSED_PATH, "datasets.pt"))
    
    print(f"\nPreprocessing complete! Saved to {PREPROCESSED_PATH}")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

if __name__ == "__main__":
    main()
