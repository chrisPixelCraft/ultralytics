from ultralytics import YOLO
import yaml

with open("concat_cfg.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Load a model
model = YOLO(cfg["model"])  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(**cfg)