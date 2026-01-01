import streamlit as st
import os
import cv2
import time
import subprocess
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AutoScanner 3D | UVeye PoC", layout="wide", page_icon="🚗")

# --- CSS TWEAKS (Professional Look) ---
st.markdown("""
    <style>
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def run_command(cmd):
    """Runs shell commands and logs to Streamlit."""
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        st.error(f"Command failed: {cmd}\n\nError:\n{e.stderr}")
        return None

def load_ply_as_plotly(ply_path, sample_rate=10):
    """Loads a PLY file and converts to Plotly 3D Scatter (Subsampled for speed)."""
    if not os.path.exists(ply_path):
        return None
    
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Subsample for web performance
    points = points[::sample_rate]
    colors = colors[::sample_rate]
    
    trace = go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers',
        marker=dict(size=2, color=colors, opacity=0.8)
    )
    return trace

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3089/3089851.png", width=50) # Placeholder Icon
    st.title("AutoScanner Pro")
    st.markdown("---")
    
    # 1. Input Source
    st.subheader("1. Input Data")
    upload_option = st.radio("Source:", ["Upload File", "Local Path"])
    
    video_path = None
    if upload_option == "Upload File":
        uploaded_file = st.file_uploader("Select MP4/MOV", type=["mp4", "mov", "avi"])
        if uploaded_file:
            # Save temp
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = "temp_video.mp4"
    else:
        video_path = st.text_input("Absolute Path to Video:", value="input.mp4")

    st.markdown("---")
    
    # 2. Pipeline Settings
    st.subheader("2. Pipeline Config")
    sample_rate = st.slider("Frame Sample Rate", 5, 60, 10, help="Process every Nth frame")
    blur_threshold = st.slider("Blur Threshold", 0.0, 500.0, 0.0, help="Lower = Keep more blurry frames") # Default 0 to ensure we get frames
    run_btn = st.button("🚀 START PIPELINE")

    st.markdown("---")
    st.caption("Powered by COLMAP & Open3D")

# --- MAIN DASHBOARD ---
if not video_path or not os.path.exists(video_path):
    st.info("👈 Please upload a video or provide a path to begin.")
    st.stop()

# TABS
tab_data, tab_3d, tab_logs = st.tabs(["🔍 Data Forensics", "🧊 3D Reconstruction", "📜 System Logs"])

# --- TAB 1: DATA FORENSICS ---
with tab_data:
    col_vid, col_frames = st.columns([1, 2])
    
    with col_vid:
        st.subheader("Original Footage")
        st.video(video_path)
        
    with col_frames:
        st.subheader("Frame Inspector")
        
        # Check if processing is done
        # Check if processing is done
        image_dir = "project_output/data/images"
        mask_dir = "project_output/data/masks"
        
        if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0:
            frames = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
            
            # Interactive Controls
            selected_idx = st.slider("Select Frame Index", 0, len(frames)-1, 0)
            show_mask = st.toggle("🛡️ Overlay Segmentation Mask", value=False)
            
            # Load Images
            frame_name = frames[selected_idx]
            img_path = os.path.join(image_dir, frame_name)
            mask_path = os.path.join(mask_dir, frame_name + ".png")
            
            img = Image.open(img_path)
            
            if show_mask and os.path.exists(mask_path):
                # Overlay Logic
                mask = Image.open(mask_path).convert("L")
                # Create red overlay
                overlay = Image.new("RGB", img.size, (255, 0, 0))
                # Composite
                img = Image.composite(overlay, img, mask)
                st.caption(f"Viewing: {frame_name} | **Mask Applied**")
            else:
                st.caption(f"Viewing: {frame_name} | **Raw RGB**")

            st.image(img, use_container_width=True)
            
        else:
            st.warning("No processed frames found. Run the pipeline first.")

# --- PIPELINE RUNNER ---
if run_btn:
    with tab_logs:
        st.subheader("Execution Logs")
        
        # Step 1: Preprocess
        st.info(f"[*] Starting Preprocessing on {video_path}...")
        os.makedirs("project_output/data", exist_ok=True)
        cmd_1 = f"python preprocess.py --video {video_path} --out project_output/data --sample_rate {sample_rate} --blur_threshold {blur_threshold}"
        log_1 = run_command(cmd_1)
        if log_1:
            st.code(log_1, language="bash")
            st.toast("Preprocessing Complete!", icon="✅")
        else:
            st.error("Preprocessing Failed! Check logs above.")
            st.stop()
        
        # Step 2: Reconstruct
        st.info(f"[*] Starting Reconstruction (this may take a while)...")
        os.makedirs("project_output/reconstruction", exist_ok=True)
        cmd_2 = f"python reconstruct.py --data project_output/data --out project_output/reconstruction"
        log_2 = run_command(cmd_2)
        if log_2:
            st.code(log_2, language="bash")
            st.toast("Reconstruction Complete!", icon="🎉")
        else:
            st.error("Reconstruction Failed! Check logs above.")
            st.stop()
        
        # Trigger Reload (with delay to let user see status)
        time.sleep(2)
        st.rerun()

# --- TAB 2: 3D RECONSTRUCTION ---
with tab_3d:
    st.subheader("Interactive 3D View")
    
    # Priority: Mesh -> Dense -> Sparse
    # We prefer the mesh if available, but Plotly is better at points. 
    # Let's stick to Dense Point Cloud for the "Interactive View" as requested, 
    # but check project_output folder.
    
    base_dir = "project_output/reconstruction"
    ply_file = os.path.join(base_dir, "dense", "fused.ply") 
    
    # Fallback to sparse if dense missing
    if not os.path.exists(ply_file):
        # Depending on colmap version/output
        candidates = [
            os.path.join(base_dir, "sparse", "0", "points3D.ply"),
            os.path.join(base_dir, "fused.ply")
        ]
        for c in candidates:
            if os.path.exists(c):
                ply_file = c
                break

    if os.path.exists(ply_file):
        st.success(f"loaded model: {ply_file}")
        
        # Metric Cards
        col1, col2 = st.columns(2)
        col1.metric("File Size", f"{os.path.getsize(ply_file)/1024/1024:.2f} MB")
        col2.metric("Pipeline Status", "Ready for Inspection")

        # Load & Visualize
        with st.spinner("Rendering 3D Point Cloud..."):
            trace = load_ply_as_plotly(ply_file)
            if trace:
                layout = go.Layout(
                    scene=dict(aspectmode='data'),
                    margin=dict(l=0, r=0, b=0, t=0),
                    height=600
                )
                fig = go.Figure(data=[trace], layout=layout)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to parse PLY file.")
    else:
        st.info("No 3D model found yet. Run the pipeline to generate.")
