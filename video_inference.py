import os
import cv2
import datetime
import numpy as np
import subprocess
from tqdm import tqdm
from ultralytics import YOLO

# Define configuration constants
CONFIDENCE_THRESHOLD_LIMIT = 0 # originally 0.2
BOX_COLOUR = (0, 255, 0)
MOSAIC_SIZE = 25

# Define the device type. Set to "mps" if you want to use M1 Mac GPU, otherwise use "cpu"
DEVICE = "cuda:0"

# Define video source. You can use a webcam, video file or a live stream
# VIDEO_SOURCE = cv2.VideoCapture(0)  # 0 for webcam
for i in range(1, 12):
    cap = cv2.VideoCapture(f'NTU-MTMC/test/Cam' + str(i) + '/Cam' + str(i) + '.MP4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or try 'H264', 'AVCL', 'AVC1'
    video_out_fn = 'output/test/Cam' + str(i) + '/test.mp4'
    os.makedirs(os.path.dirname(video_out_fn), exist_ok=True)
    video_out = cv2.VideoWriter(video_out_fn, fourcc, fps, (frame_width, frame_height))

    # Check if VideoWriter was initialized properly
    if not video_out.isOpened():
        print(f"H.264 codec not available, falling back to XVID")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(video_out_fn, fourcc, fps, (frame_width, frame_height))

    # Load the YOLO model
    model = YOLO("/root/ultralytics/runs/detect/train2/weights/best.pt")
    total_sec = 0
    frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #print(frame_cnt)
    #exit()
    #while True:

    for j in tqdm(range(int(frame_cnt))):
        start = datetime.datetime.now()
        ret, frame = cap.read()

        #frame = cv2.resize(frame, (1080, 1920, 3))
        # if there are no more frames to process, stop the loop
        if not ret:
            print(f'Cannot read frame')
            #cap = cv2.VideoCapture('../datasets/ntu_mtmc/test/cam1/cam1.MP4')
            break

        # Perform object detection, set MPS as the device type
        detections = model(frame, device=DEVICE, verbose=False)
        result = model(frame)[0]

        # Transform the results to numpy arrays and integers. Pixels are always integers
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        confidence = np.array(result.boxes.conf.cpu(), dtype="float")

        # Draw the bounding boxes and labels on the frame. The color of the bounding box depends on the confidence
        for cls, bbox, conf in zip(classes, bboxes, confidence):
            (x, y, x2, y2) = bbox
            object_name = model.names[cls]
            if conf < CONFIDENCE_THRESHOLD_LIMIT:
                continue
            if conf > 0.6:
                BOX_COLOUR = (37, 245, 75)
            elif conf < 0.6 and conf > 0.2:
                BOX_COLOUR = (66, 224, 245)
            else:
                BOX_COLOUR = (78, 66, 245)
            tmp = frame[y:y2, x:x2]
            #mosaic = cv2.resize(tmp, (MOSAIC_SIZE, MOSAIC_SIZE), interpolation=cv2.INTER_LINEAR)
            #mosaic = cv2.resize(mosaic, (x2 - x, y2 - y), interpolation=cv2.INTER_NEAREST)
            mosaic = cv2.GaussianBlur(tmp, (MOSAIC_SIZE, MOSAIC_SIZE), 10)
            frame[y:y2, x:x2] = mosaic
            cv2.rectangle(frame, (x, y), (x2, y2), BOX_COLOUR, 2)
            cv2.putText(frame, f"{object_name}: {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        # Measure time it took to process 1 frame and overlay fps on the frame
        end = datetime.datetime.now()

        # Calculate the frame per second and draw it on the frame
        # ffps = f"FPS: {1 / total:.2f}"
        # cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        # Display the output video
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.namedWindow("output", 0)
        # cv2.imshow("output", frame)

        # Stop processing when the "q" key is pressed
        # if cv2.waitKey(1) == ord("q"):
        #     break

        # Add a progress indicator instead of GUI display
        if j % 100 == 0:
            print(f"Processing frame {j}/{int(frame_cnt)} for camera {i}")

        # Write the frame to the output video?
        video_out.write(frame)
        #if j == 5000:
        #    break
    cap.release()
    # cv2.destroyAllWindows()
    # exit()

ffmpeg_input = "D:/research/NTU-MTMC/yolo-face/datasets/ntu_mtmc/train/Cam" + str(i) + ".MP4"
ffmpeg_out_fn = "D:/research/NTU-MTMC/yolo-face/datasets/ntu_mtmc/train/Cam" + str(i) + "_out.MP4"
os.system(f"ffmpeg.exe -i " + ffmpeg_input + " -vcodec libx264 -crf 18 -preset slow " + ffmpeg_out_fn)