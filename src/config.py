from pathlib import Path
import torch


# ========== 设备 ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 路径 ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = PROJECT_ROOT / "data"
TRAIN: Path = DATA_ROOT / "train"
TEST: Path = DATA_ROOT / "test"
LABELS: Path = DATA_ROOT / "Labels.json"
SPLIT: Path = DATA_ROOT / "split.json"

# ========= 实验划分 =========
RESULTS = PROJECT_ROOT / "results"
LOGS = PROJECT_ROOT / "logs"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"

# ========== 数据 ==========
NUM_CLASSES = 100
TRAIN_VAL_SPLIT_RATIO = 0.95
BATCH_SIZE = 64
NUM_WORKERS = 10
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
SEED = 42

# ========== 数据增强 ==========
NUM_OPS = 2
RAND_AUGMENT_MAGNITUDE = 12
RANDOM_ERASING_P = 0.25
PREFECTCH_FACTOR = 4

# ========== 训练 ==========
EPOCHS = 200
WARMUP_EPOCHS = 5
BASE_LR = 0.1 * BATCH_SIZE  / 256
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1