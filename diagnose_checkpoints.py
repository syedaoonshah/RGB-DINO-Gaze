"""
Diagnostic Script: Check what's actually in your checkpoints
"""

import torch
import numpy as np

def check_checkpoint(path, name):
    """Analyze a checkpoint file"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"Path: {path}")
    print(f"{'='*70}")
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        # Check structure
        print(f"\nCheckpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Keys in checkpoint: {list(checkpoint.keys())}")
            
            # Check for metadata
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
            if 'val_mae' in checkpoint:
                print(f"  Val MAE: {checkpoint['val_mae']:.2f}°")
            
            # Get state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Analyze weights
        print(f"\nModel structure:")
        print(f"  Total parameters: {len(state_dict)}")
        
        # Check specific layers
        gaze_head_params = [k for k in state_dict.keys() if 'gaze_head' in k]
        dinov2_params = [k for k in state_dict.keys() if 'dinov2' in k]
        mem_head_params = [k for k in state_dict.keys() if 'mem_head' in k]
        
        print(f"  Gaze head params: {len(gaze_head_params)}")
        print(f"  DINOv2 params: {len(dinov2_params)}")
        print(f"  MEM head params: {len(mem_head_params)}")
        
        if gaze_head_params:
            print(f"\n  Gaze head layers:")
            for param in gaze_head_params[:5]:  # Show first 5
                print(f"    {param}")
        
        # Compute weight statistics to fingerprint the model
        print(f"\nWeight fingerprint (first gaze_head layer):")
        if gaze_head_params:
            first_param_name = gaze_head_params[0]
            first_param = state_dict[first_param_name].numpy()
            print(f"  Shape: {first_param.shape}")
            print(f"  Mean: {first_param.mean():.6f}")
            print(f"  Std: {first_param.std():.6f}")
            print(f"  Min: {first_param.min():.6f}")
            print(f"  Max: {first_param.max():.6f}")
            print(f"  Sum: {first_param.sum():.6f}")
        
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")

def compare_checkpoints(path1, path2, name1, name2):
    """Compare two checkpoints to see if they're the same"""
    print(f"\n{'='*70}")
    print(f"COMPARING: {name1} vs {name2}")
    print(f"{'='*70}")
    
    try:
        ckpt1 = torch.load(path1, map_location='cpu')
        ckpt2 = torch.load(path2, map_location='cpu')
        
        # Get state dicts
        if isinstance(ckpt1, dict) and 'model_state_dict' in ckpt1:
            state1 = ckpt1['model_state_dict']
        else:
            state1 = ckpt1
            
        if isinstance(ckpt2, dict) and 'model_state_dict' in ckpt2:
            state2 = ckpt2['model_state_dict']
        else:
            state2 = ckpt2
        
        # Compare keys
        keys1 = set(state1.keys())
        keys2 = set(state2.keys())
        
        print(f"\nKey comparison:")
        print(f"  Keys in both: {len(keys1 & keys2)}")
        print(f"  Only in {name1}: {len(keys1 - keys2)}")
        print(f"  Only in {name2}: {len(keys2 - keys1)}")
        
        # Compare weights for common keys
        common_keys = list(keys1 & keys2)
        if common_keys:
            # Check gaze_head weights
            gaze_keys = [k for k in common_keys if 'gaze_head' in k]
            if gaze_keys:
                first_key = gaze_keys[0]
                w1 = state1[first_key].numpy()
                w2 = state2[first_key].numpy()
                
                diff = np.abs(w1 - w2).mean()
                print(f"\nGaze head weight difference:")
                print(f"  Layer: {first_key}")
                print(f"  Mean absolute difference: {diff:.10f}")
                
                if diff < 1e-6:
                    print(f"  ⚠️ WARNING: Weights are IDENTICAL or nearly identical!")
                    print(f"  These might be the same model!")
                else:
                    print(f"  ✓ Weights are different (models are distinct)")
    
    except Exception as e:
        print(f"ERROR comparing: {e}")

if __name__ == "__main__":
    print("="*70)
    print("CHECKPOINT DIAGNOSTICS")
    print("="*70)
    
    # Check baseline
    check_checkpoint(
        "./models/dinov2_gaze_baseline.pth",
        "Baseline (Vanilla DINOv2)"
    )
    
    # Check your semi-supervised model
    check_checkpoint(
        "./models/rgb_dino_gaze_simple.pth",
        "Semi-Supervised (Yours)"
    )
    
    # Compare them
    compare_checkpoints(
        "./models/dinov2_gaze_baseline.pth",
        "./models/rgb_dino_gaze_simple.pth",
        "Baseline",
        "Semi-Supervised"
    )
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)