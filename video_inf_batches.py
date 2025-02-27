import os
import cv2
import datetime
import numpy as np
import subprocess
from tqdm import tqdm
from ultralytics import YOLO

# Define configuration constants
CONFIDENCE_THRESHOLD_LIMIT = 0
BOX_COLOUR = (0, 255, 0)
MOSAIC_SIZE = 25
DEVICE = "cuda:0"
BATCH_SIZE = 64  # Adjust based on GPU memory

# Process camera videos one at a time, but frames in batches
for i in range(6, 12):
    cap = cv2.VideoCapture(f'NTU-MTMC/test/Cam{i}/Cam{i}.MP4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out_fn = f'output/test/Cam{i}/test.mp4'
    os.makedirs(os.path.dirname(video_out_fn), exist_ok=True)
    video_out = cv2.VideoWriter(video_out_fn, fourcc, fps, (frame_width, frame_height))

    # Check if VideoWriter was initialized properly
    if not video_out.isOpened():
        print(f"H.264 codec not available, falling back to XVID")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(video_out_fn, fourcc, fps, (frame_width, frame_height))

    # Load the YOLO model
    model = YOLO("train45/weights/best.pt")
    frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Process frames in batches
    batch_frames = []
    frame_indices = []

    for j in tqdm(range(int(frame_cnt)), desc=f"Processing Cam{i}"):
        ret, frame = cap.read()

        if not ret:
            print(f'Cannot read frame {j}')
            break

        # Add the frame to our batch
        batch_frames.append(frame)
        frame_indices.append(j)

        # When we have a full batch or it's the last frame, process the batch
        if len(batch_frames) == BATCH_SIZE or j == int(frame_cnt) - 1:
            # Process the entire batch at once - this is much more GPU efficient
            results = model(batch_frames, device=DEVICE, verbose=False)

            # Process each result individually
            for frame, result in zip(batch_frames, results):
                # Transform the results to numpy arrays
                bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
                classes = np.array(result.boxes.cls.cpu(), dtype="int")
                confidence = np.array(result.boxes.conf.cpu(), dtype="float")

                # Apply mosaic to each detection
                for cls, bbox, conf in zip(classes, bboxes, confidence):
                    if conf < CONFIDENCE_THRESHOLD_LIMIT:
                        continue

                    (x, y, x2, y2) = bbox
                    object_name = model.names[cls]

                    # Set box color based on confidence
                    if conf > 0.6:
                        BOX_COLOUR = (37, 245, 75)
                    elif conf < 0.6 and conf > 0.2:
                        BOX_COLOUR = (66, 224, 245)
                    else:
                        BOX_COLOUR = (78, 66, 245)

                    # Apply mosaic/blur
                    tmp = frame[y:y2, x:x2]
                    mosaic = cv2.GaussianBlur(tmp, (MOSAIC_SIZE, MOSAIC_SIZE), 10)
                    frame[y:y2, x:x2] = mosaic

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x2, y2), BOX_COLOUR, 2)
                    cv2.putText(frame, f"{object_name}: {conf:.2f}", (x, y - 5),
                               cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                # Write the processed frame
                video_out.write(frame)

            # Clear the batch
            batch_frames = []
            frame_indices = []

            # Progress indicator
            if j % 100 == 0:
                print(f"Processing frame {j}/{int(frame_cnt)} for camera {i}")

    # Clean up
    cap.release()
    video_out.release()

