import cv2
import numpy as np
import argparse
import os
from rembg import remove
from PIL import Image
from tqdm import tqdm

def variance_of_laplacian(image):
    """
    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def process_video(video_path, output_dir, sample_rate=10, blur_threshold=100.0):
    """
    Extracts frames, filters blur, and generates masks.
    """
    # Create directories
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[*] Processing {video_path}...")
    print(f"[*] Total frames: {total_frames}. Sampling every {sample_rate}th frame.")

    count = 0
    saved_count = 0
    
    pbar = tqdm(total=total_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Sampling Rate Check
        if count % sample_rate == 0:
            
            # 2. Blur Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)

            if fm > blur_threshold:
                filename = f"frame_{saved_count:05d}.jpg"
                img_path = os.path.join(img_dir, filename)
                # COLMAP SAFE NAMING: frame_00000.jpg -> frame_00000.jpg.png
                mask_path = os.path.join(mask_dir, filename + ".png")

                # Save original image
                cv2.imwrite(img_path, frame)

                # 3. AI Background Removal (Masking)
                # Convert to PIL for rembg
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # 'remove' returns the image with alpha channel (RGBA)
                # We need to extract just the alpha channel as a binary mask
                no_bg = remove(pil_img)
                mask = no_bg.split()[-1] # Get Alpha channel
                
                # Save mask (Binary: White=Car, Black=Background)
                mask.save(mask_path)

                saved_count += 1
            else:
                pass # Frame is too blurry, skip it

        count += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"[*] Done. Saved {saved_count} clean frames and masks to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--sample_rate", type=int, default=10, help="Frame sampling rate (default: 10)")
    parser.add_argument("--blur_threshold", type=float, default=100.0, help="Blur threshold (default: 100.0)")
    args = parser.parse_args()

    process_video(args.video, args.out, args.sample_rate, args.blur_threshold)
