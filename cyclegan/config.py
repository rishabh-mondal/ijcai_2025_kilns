import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = f"/home/patel_zeel/kiln_compass_24/data"
TEST_DIR = f"../data"
VAL_DIR = f"../data/cyclegan_data/val"
BATCH_SIZE = 10
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 1.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 8
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = True
# home="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/cyclegan"
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=640, height=640),
        A.Normalize(max_pixel_value=255.0),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)