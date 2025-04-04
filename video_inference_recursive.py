import os
import cv2
import datetime
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import subprocess  # Added for FFmpeg processing

# Define configuration constants
CONFIDENCE_THRESHOLD_LIMIT = 0  # originally 0.2, adjust as needed
MOSAIC_SIZE = 25
DEVICE = "cuda:0"

def process_video(input_path, output_path, model_path):
    """
    Process the video at input_path using the model in model_path.
    Draws a blur/mosaic over detections and writes the results to output_path.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Try initializing VideoWriter first with mp4v then fall back to XVID if needed.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not video_out.isOpened():
        print("H.264 codec not available, falling back to XVID")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Load the model checkpoint
    model = YOLO(model_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for j in tqdm(range(total_frames), desc=f"Processing {os.path.basename(input_path)} with {os.path.basename(model_path)}"):
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        # Resize the frame to 4 times larger while maintaining resolution and clarity
        # Using INTER_CUBIC interpolation for better quality when upscaling
        # Try increasing the scaling factor, e.g., *3 or *4
        scale_factor = 4 # Or 4, experiment with this value
        frame = cv2.resize(frame, (frame_width*scale_factor, frame_height*scale_factor), interpolation=cv2.INTER_CUBIC)

        # Run the model inference on the frame (device and verbosity set as needed)
        results = model(frame, device=DEVICE, verbose=False)
        result = results[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype=int)
        classes = np.array(result.boxes.cls.cpu(), dtype=int)
        confidence = np.array(result.boxes.conf.cpu(), dtype=float)

        # Apply mosaic effect to detections meeting the confidence threshold
        for bbox, cls, conf in zip(bboxes, classes, confidence):
            if conf < CONFIDENCE_THRESHOLD_LIMIT:
                continue
            x, y, x2, y2 = bbox
            roi = frame[y:y2, x:x2]
            blurred = cv2.GaussianBlur(roi, (MOSAIC_SIZE, MOSAIC_SIZE), 10)
            frame[y:y2, x:x2] = blurred

        # Downsample the frame to the original size while maintaining resolution and clarity
        # Using INTER_CUBIC for high-quality downsampling with good balance of quality and performance
        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

        video_out.write(frame)

    cap.release()
    video_out.release()
    return output_path

def recursive_process(input_path, model_paths, iteration=0):
    """
    Recursively process the video through each model in the list.
    The output of one iteration is used as the input for the next.
    """
    if iteration >= len(model_paths):
        return input_path  # All models have been processed.

    # Create a unique output name from the current input name.
    base, ext = os.path.splitext(input_path)
    output_path = base + f"_iter{iteration+1}" + ext
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nStarting iteration {iteration+1} using model: {model_paths[iteration]}")
    processed_video = process_video(input_path, output_path, model_paths[iteration])
    if processed_video is None:
        print("Processing failed at iteration ", iteration+1)
        return input_path

    # Recursively call the function with the new video as input.
    return recursive_process(processed_video, model_paths, iteration+1)

def main():
    # List of model checkpoint files (downloaded in get_ckpt.sh)
    model_paths = [
        "/root/ultralytics/yolo11x_best.pt",
        "/root/ultralytics/yolo11l_best.pt",
        "/root/ultralytics/yolo11m_best.pt",
        "/root/ultralytics/yolo11s_best.pt",
        "/root/ultralytics/yolo11n_best.pt"
    ]

    # Process videos for cameras 1 through 11
    for i in range(10, 12):
        original_video = f"NTU-MTMC/test/Cam{i}/Cam{i}.MP4"
        # Check if the video file exists
        if not os.path.exists(original_video):
            print(f"\nVideo file {original_video} does not exist. Skipping Camera {i}.")
            continue

        print(f"\nProcessing Camera {i}: {original_video}")
        # Run the recursive processing (each iteration uses the last iteration's output as input)
        final_video = recursive_process(original_video, model_paths)
        print(f"Final output for Camera {i}: {final_video}")

        # Check if the final processed video exists before attempting FFmpeg conversion.
        if not os.path.exists(final_video):
            print(f"Final video {final_video} does not exist. Skipping FFmpeg processing for Camera {i}.")
            continue

        # Post-process with FFmpeg for re-encoding using subprocess.run (referencing compress_recursive.py)
        ffmpeg_input = final_video
        ffmpeg_output = ffmpeg_input.replace(".MP4", "_final.MP4")
        cmd = ["ffmpeg", "-i", ffmpeg_input, "-vcodec", "libx264", "-crf", "18", "-preset", "slow", ffmpeg_output]
        try:
            print(f"Processing FFmpeg on Camera {i} video...")
            subprocess.run(cmd, check=True)
            print(f"FFmpeg processed video saved as {ffmpeg_output}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing FFmpeg on Camera {i}: {e}")
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please make sure ffmpeg is installed and in your PATH.")

if __name__ == "__main__":
    main()