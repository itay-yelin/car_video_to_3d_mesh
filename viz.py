import open3d as o3d
import numpy as np
import pycolmap
import argparse
from pathlib import Path

def draw_camera(start, end, color=[1, 0, 0], width=0.1):
    # Simple line
    return [start, end]

def visualize_results(model_path, ply_path):
    print(f"[*] Loading reconstruction from {model_path}")
    recon = pycolmap.Reconstruction(model_path)
    
    print(f"[*] Loading Point Cloud from {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    # Create visualizer elements for cameras
    cam_geometries = []
    
    # 0,0,0 coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    cam_geometries.append(axis)

    print(f"[*] Found {len(recon.images)} cameras.")
    
    # Frustum size
    scale = 0.2
    
    for image_id, image in recon.images.items():
        # Get camera position and rotation
        # COLMAP Common: T_world_to_cam (R, t)
        # We need T_cam_to_world (inverse)
        
        # rotation matrix
        R = image.cam_from_world.rotation.matrix()
        t = image.cam_from_world.translation
        
        # Camera center in world coordinates = -R^T * t
        center = -R.T @ t
        
        # Viewing direction (Z-axis of camera is usually forward in COLMAP? No, +Z is forward in Open3D/GL, COLMAP might be different)
        # COLMAP: +Z is optical axis?
        # We can just draw a small pyramid representing the camera.
        
        # Create a small mesh or lines for the camera
        # Simpler: Create a LineSet for the frustum
        
        # Points in camera coordinates
        # 0: Center
        # 1-4: Corners of the image plane at distance 'scale'
        w, h = 1.0, 0.75 # Aspect ratio approx
        points = [
            [0, 0, 0],
            [w, h, 1],
            [w, -h, 1],
            [-w, -h, 1],
            [-w, h, 1]
        ]
        points = np.array(points) * scale
        
        # Transform to world
        # X_world = R^T * (X_cam - t)  <-- Check math
        # Actually: X_cam = R * X_world + t
        # X_world = R^T * (X_cam - t) = R^T * X_cam - R^T * t = R^T * X_cam + Center
        
        points_world = (R.T @ points.T).T + center
        
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4], # Ray to corners
            [1, 2], [2, 3], [3, 4], [4, 1]  # Image plane rectangle
        ]
        
        colors = [[1, 0, 0] for _ in range(len(lines))]
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_world),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        cam_geometries.append(line_set)

    print("[*] Visualizing... (Close window to exit)")
    o3d.visualization.draw_geometries([pcd] + cam_geometries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to COLMAP sparse folder (e.g., output/sparse)")
    parser.add_argument("--ply", required=True, help="Path to fused.ply")
    args = parser.parse_args()
    
    visualize_results(args.model, args.ply)
