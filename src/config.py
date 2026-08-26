from pathlib import Path
import torch

# ========== 设备 ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 路径 ==========
TRAIN = Path("../data/train")
TEST = Path("../data/test")
FIGURES = Path("../results/figures")
LOGS = Path("../logs")
CHECKPOINTS = Path("../checkpoints")

# ========== 数据 ==========
TRAIN_VAL_SPLIT_RATIO = 0.95
BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1 
NUM_WORKERS = 12
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
SEED = 42

# ========== 训练 ==========
EPOCHS = 150
WARMUP_EPOCHS = 5
BASE_LR = 0.1 * (BATCH_SIZE * GRAD_ACCUM_STEPS) / 256
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
USE_AMP = True