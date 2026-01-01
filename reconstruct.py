import pycolmap
import argparse
import os
from pathlib import Path
import open3d as o3d
import numpy as np

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

    # --- Meshing ---
    mesh_output = output_path / "final_mesh.ply"
    create_mesh_from_dense_pcd(dense_ply, mesh_output)

def create_mesh_from_dense_pcd(pcd_path, output_mesh_path, depth=9):
    print(f"[*] Loading dense point cloud from {pcd_path}...")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    
    # 1. Estimate Normals (Crucial for Poisson)
    print("[*] Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # Align normals to point towards the camera locations (heuristic) 
    # or just orient consistently for a closed object
    pcd.orient_normals_consistent_tangent_plane(100)

    # 2. Poisson Surface Reconstruction
    print(f"[*] Running Poisson Reconstruction (depth={depth})...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )
    
    # 3. Clean the Mesh (Remove "Bubble" artifacts)
    # Poisson creates a "bubble" around the object. We remove low-density vertices.
    print("[*] Cleaning mesh artifacts...")
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print(f"[*] Saving final mesh to {output_mesh_path}")
    o3d.io.write_triangle_mesh(str(output_mesh_path), mesh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to preprocessed data directory (containing images/ and masks/)")
    parser.add_argument("--out", required=True, help="Output directory for reconstruction")
    args = parser.parse_args()
    
    run_reconstruction(args.data, args.out)
