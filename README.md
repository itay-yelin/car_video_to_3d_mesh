# 1-Day Auto-Scanner PoC

This project is an automated photogrammetry pipeline compatible with UVeye's tech stack (Python, C++ bindings, Computer Vision).

**Environment**: Python 3.11 (Virtual Environment `venv`)

## Quick Start

1.  **Activate Environment**:
    ```powershell
    .\venv\Scripts\activate
    ```

2.  **Phase 1: Ingest & Preprocess**
    *Extracts frames, removes blur (Laplacian Variance), and masks background (AI).*
    ```powershell
    python preprocess.py --video your_video.mp4 --out ./project_data
    ```

3.  **Phase 2: Reconstruction**
    *Runs SfM (Structure from Motion) using `pycolmap`.*
    ```powershell
    python reconstruct.py --data ./project_data --out ./reconstruction
    ```

4.  **Phase 3: Visualization**
    *Visualizes the camera trajectory and sparse point cloud.*
    ```powershell
    python viz.py --model ./reconstruction/sparse --ply ./reconstruction/fused.ply
    ```

## Project Structure
*   `preprocess.py`: Smart frame extraction & Rembg masking.
*   `reconstruct.py`: Pycolmap SfM pipeline.
*   `viz.py`: Open3D visualization with camera frustums.
