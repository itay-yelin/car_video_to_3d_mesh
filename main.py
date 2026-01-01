import argparse
import subprocess
import os
import sys

def run_step(command):
    print(f"\n[pipeline] Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"[!] Error executing: {command}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--project", default="./project_output", help="Project output folder")
    args = parser.parse_args()

    # Define paths
    data_dir = os.path.join(args.project, "data")
    recon_dir = os.path.join(args.project, "reconstruction")
    
    # 1. Preprocess
    run_step(f"python preprocess.py --video {args.video} --out {data_dir}")
    
    # 2. Reconstruct (Sparse + Dense + Mesh)
    run_step(f"python reconstruct.py --data {data_dir} --out {recon_dir}")
    
    # 3. Visualize (Dense Point Cloud)
    # Note: We point to the DENSE ply now, not the sparse one
    dense_ply = os.path.join(recon_dir, "dense", "fused.ply")
    sparse_model = os.path.join(recon_dir, "sparse")
    
    print("\n[*] Pipeline Finished. Launching visualizer...")
    run_step(f"python viz.py --model {sparse_model} --ply {dense_ply}")

if __name__ == "__main__":
    main()
