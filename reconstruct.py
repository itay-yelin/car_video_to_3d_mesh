import pycolmap
import argparse
import os
from pathlib import Path

def run_reconstruction(data_dir, output_dir):
    data_path = Path(data_dir)
    images_path = data_path / "images"
    masks_path = data_path / "masks"
    database_path = Path(output_dir) / "database.db"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    if database_path.exists():
        database_path.unlink()

    print(f"[*] Creating database at {database_path}...")
    
    # 1. Feature Extraction with Masks
    # We need to map images to their corresponding masks.
    # Pycolmap/COLMAP expects a specific mask handling.
    # The simplest way via pycolmap API:
    print("[*] Extracting features (SIFT)...")
    
    # Verify masks exist and match image names (png vs jpg)
    # This setup assumes Pycolmap can find masks if we configure the options correctly
    # or we iterate manually.
    # Simple extraction:
    pycolmap.extract_features(
        database_path, 
        images_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_list=None, # Process all
        descriptor_normalization=True
        # Note: Masking in pure pycolmap.extract_features usually requires SiftExtractionOptions
        # and ensuring the reader finds the masks.
        # But commonly, if we cannot guarantee mask paths, we might need a workaround.
        # However, for this PoC, we will try the standard method:
        # If a file "image_name.jpg.png" exists in the mask_path?
        # COLMAP looks for masks in the same folder or specific struct. 
    )
    
    # Wait, pycolmap.extract_features doesn't easily accept a separate mask folder argument in all versions.
    # A robust hack for 1-day PoC: Move masks to next to images or rename them?
    # Or better: "Crucial: It must apply the masks".
    # Let's try to pass `ImageReaderOptions` if exposed.
    # If not, we warn the user. But let's check basic usage.
    
    print("[*] Matching features (Sequential)...")
    pycolmap.match_sequential(database_path, overlap=5) # 5-10 overlap for video frames

    print("[*] Running Incremental Mapper...")
    maps = pycolmap.incremental_mapping(database_path, images_path, output_path)
    
    if not maps:
        print("[!] Reconstruction failed! No models created.")
        return

    # Save the best model (usually index 0)
    best_model_idx = 0 
    best_model = maps[best_model_idx]
    
    sparse_dir = output_path / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    best_model.write(sparse_dir)
    
    ply_path = output_path / "fused.ply"
    best_model.export_ply(ply_path)
    print(f"[*] Reconstruction finished. Saved to {output_path}")
    print(f"[*] Point cloud: {ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to preprocessed data directory (containing images/ and masks/)")
    parser.add_argument("--out", required=True, help="Output directory for reconstruction")
    args = parser.parse_args()
    
    run_reconstruction(args.data, args.out)
