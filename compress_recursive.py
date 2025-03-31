import os
import cv2
import datetime
import numpy as np
import subprocess
from tqdm import tqdm

# Create output directory if it doesn't exist
os.makedirs("output_compressed_yolo11x/test", exist_ok=True)

for i in range(1, 12):
    ffmpeg_input = f"output_yolo11x/test/Cam{i}/test.mp4"
    ffmpeg_out_fn = f"output_compressed_yolo11x/test/Cam{i}_out.mp4"
    # os.makedirs(f"output_compressed_yolo11x/test/Cam{i}", exist_ok=True)

    # Use subprocess.run instead of os.system for better error handling
    # Remove the .exe extension for Linux compatibility
    cmd = ["ffmpeg", "-i", ffmpeg_input, "-vcodec", "libx264", "-crf", "18", "-preset", "slow", ffmpeg_out_fn]

    try:
        print(f"Processing Cam{i}...")
        subprocess.run(cmd, check=True)
        print(f"Successfully compressed Cam{i}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing Cam{i}: {e}")
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please make sure ffmpeg is installed and in your PATH.")
        break

