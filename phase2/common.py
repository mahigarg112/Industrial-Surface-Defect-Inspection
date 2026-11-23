import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("SDI_PHASE2_OUTPUT",str(ROOT / "phase2" / "artifacts"))).resolve()
if not OUT.is_relative_to(ROOT / "phase2"):
    raise ValueError("SDI_PHASE2_OUTPUT must stay inside this project's phase2 directory")
LEGACY = ROOT / "experiments" / "severstal_segmentation"
IMAGES = ROOT / "data" / "severstal-steel-defect-detection" / "train_images"
CLASSES = [f"class_{i}" for i in range(1, 5)]


def legacy_module(name):
    spec = importlib.util.spec_from_file_location(f"phase2_legacy_{name}", LEGACY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Standalone release: byte-identical copies of the preserved Phase I definitions.
# legacy_module remains only for historical audit/evaluation reproduction.
from .model import SmallUNet
from .dataset_utils import decode_rle


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(path, device="cuda"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != CLASSES:
        raise ValueError("Four-class mapping mismatch")
    model = SmallUNet(base_channels=checkpoint["config"]["base_channels"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
