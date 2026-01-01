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
    print("[*] Extracting features (SIFT) with Masks...")
    
    # Define options to use masks
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.mask_path = masks_path  # Point to the masks folder
    
    # Run extraction
    pycolmap.extract_features(
        database_path, 
        images_path, 
        reader_options=reader_options,
        verbose=True
    )
    
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

    # --- Dense Reconstruction (MVS) ---
    print("[*] Running Dense Reconstruction (MVS)...")
    dense_path = output_path / "dense"
    dense_path.mkdir(exist_ok=True)
    
    # Undistort images for dense stereo
    pycolmap.undistort_images(dense_path, sparse_dir, images_path)
    
    # Stereo matching (The heavy lifting)
    pycolmap.patch_match_stereo(dense_path) 
    
    # Fusion to dense point cloud
    dense_ply = dense_path / "fused.ply"
    pycolmap.stereo_fusion(dense_path, dense_ply)
    
    print(f"[*] Dense point cloud saved to {dense_ply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to preprocessed data directory (containing images/ and masks/)")
    parser.add_argument("--out", required=True, help="Output directory for reconstruction")
    args = parser.parse_args()
    
    run_reconstruction(args.data, args.out)
