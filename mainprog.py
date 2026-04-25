import os

# =========================
# SAFE ENVIRONMENT
# =========================
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download, login
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig
)

# =========================
# HUGGING FACE LOGIN
# =========================
HF_TOKEN = "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"
login(token=HF_TOKEN)

# =========================
# OPTIONAL: DRAW POINT HELPER
# =========================
def draw_point(image_input, point=None, radius=5):
    image = Image.open(image_input) if isinstance(image_input, str) else image_input.copy()

    if point is not None:
        x = point[0] * image.width
        y = point[1] * image.height
        ImageDraw.Draw(image).ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="red"
        )

    return image


# =========================
# STEP 1: DOWNLOAD MODEL
# =========================
repo_id = "zonghanHZH/ZonUI-3B"
local_dir = "ZonUI-3B"

print("Checking/downloading model...")

model_path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    force_download=False,
    resume_download=True,
    local_dir_use_symlinks=False
)

print("Model downloaded to:", model_path)


# =========================
# STEP 2: LOAD PROCESSOR
# =========================
print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)


# =========================
# STEP 3: LOAD MODEL ON CPU
# =========================
print("Loading model on CPU. This may take time...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
).eval()


# =========================
# STEP 4: GENERATION CONFIG
# =========================
generation_config = GenerationConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

generation_config.max_new_tokens = 256
generation_config.do_sample = False
generation_config.temperature = 0.0

model.generation_config = generation_config


# =========================
# IMAGE LIMITS
# =========================
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28


# =========================
# READY
# =========================
print("✅ ZonUI-3B loaded successfully on CPU")