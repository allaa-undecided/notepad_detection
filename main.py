import os
import ast
import time
import torch
import pyautogui
import requests
import pyperclip
import pygetwindow as gw

from PIL import Image, ImageDraw
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


model_path = r"C:\Users\areeg\OneDrive\Desktop\DON'T OPEN - ALLAA'S\Notepad Identifcation\ZonUI-3B"

SAVE_DIR = r"C:\Users\areeg\OneDrive\Desktop\DON'T OPEN - ALLAA'S\Notepad Identifcation"
os.makedirs(SAVE_DIR, exist_ok=True)


def fetch_first_post():
    urls = [
        "http://jsonplaceholder.typicode.com/posts",
        "https://jsonplaceholder.typicode.com/posts"
    ]

    for url in urls:
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20
            )
            response.raise_for_status()
            print("API worked:", url)
            return response.json()[0]

        except Exception as e:
            print("API failed:", url)
            print("Error:", repr(e))

    return {
        "id": 1,
        "title": "Fallback Post",
        "body": "The API request failed, so fallback text was used."
    }

def type_and_save_post(post):
    content = f"Title: {post['title']}\n\n{post['body']}"
    file_path = os.path.join(SAVE_DIR, f"post_{post['id']}_{timestamp}.txt")

    print("Typing post...")

    pyperclip.copy(content)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)

    print("Saving file...")

    pyautogui.hotkey("ctrl", "s")
    time.sleep(1)

    pyperclip.copy(file_path)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.3)

    pyautogui.press("enter")
    time.sleep(1)

    print("Saved:", file_path)


def wait_for_notepad(timeout=5):
    start = time.time()

    while time.time() - start < timeout:
        windows = gw.getWindowsWithTitle("Notepad")

        if windows:
            try:
                windows[0].activate()
            except Exception:
                pass

            return True

        time.sleep(0.5)

    return False


def draw_point(image_input, point=None, radius=10):
    image = Image.open(image_input) if isinstance(image_input, str) else image_input.copy()

    if point is not None:
        x = point[0] * image.width
        y = point[1] * image.height

        ImageDraw.Draw(image).ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="red"
        )

    return image


def verify_notepad_icon(model, processor, screenshot_path, normalized_x, normalized_y):
    image = Image.open(screenshot_path).convert("RGB")

    center_x = int(normalized_x * image.width)
    center_y = int(normalized_y * image.height)

    crop_size = 180

    left = max(0, center_x - crop_size)
    top = max(0, center_y - crop_size)
    right = min(image.width, center_x + crop_size)
    bottom = min(image.height, center_y + crop_size)

    cropped = image.crop((left, top, right, bottom))
    cropped_path = "verification_crop.png"
    cropped.save(cropped_path)

    verify_prompt = (
        "Look at this cropped screenshot. "
        "Is the Windows Notepad desktop shortcut icon clearly visible? "
        "Answer only YES or NO."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": cropped_path},
                {"type": "text", "text": verify_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[cropped],
        return_tensors="pt"
    ).to("cpu")

    with torch.no_grad():
        generated_ids = model.generate(**inputs)

    answer = processor.batch_decode(
        [generated_ids[0][inputs.input_ids.shape[1]:]],
        skip_special_tokens=True
    )[0].strip().lower()

    print("Verification answer:", answer)

    return answer.startswith("yes")


screen_width, screen_height = pyautogui.size()

pyautogui.moveTo(screen_width - 5, screen_height - 5, duration=0.2)
pyautogui.click()
time.sleep(1)


img_path = f"desktop_{timestamp}.png"
pyautogui.screenshot(img_path)

print("Screenshot saved:", img_path)


print("Loading model on CPU...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    trust_remote_code=True,
    use_safetensors=False,
    local_files_only=True,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
).eval()

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

generation_config = GenerationConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

generation_config.max_new_tokens = 32
generation_config.do_sample = False

model.generation_config = generation_config

print("Model loaded successfully.")


query = (
    "Find the Windows Notepad desktop shortcut icon. "
    "If there is no Notepad icon visible, return None. "
    "Do not guess."
)

system_prompt = (
    "Based on the screenshot, locate the requested desktop icon. "
    "Return ONLY [x, y] if clearly visible. "
    "If not visible, return None. Do NOT guess."
)


image = Image.open(img_path).convert("RGB")

min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28

resized_height, resized_width = smart_resize(
    image.height,
    image.width,
    factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)

resized_image = image.resize((resized_width, resized_height))

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": system_prompt},
            {"type": "image", "image": img_path},
            {"type": "text", "text": query},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[text],
    images=[resized_image],
    return_tensors="pt"
).to("cpu")


print("Finding Notepad icon...")

with torch.no_grad():
    generated_ids = model.generate(**inputs)

output_text = processor.batch_decode(
    [generated_ids[0][inputs.input_ids.shape[1]:]],
    skip_special_tokens=True
)[0].strip()

print("Raw output:", output_text)


if output_text.lower() in ["none", "null", "not found", "no"]:
    print("Notepad not found. Stopping.")
    exit()

try:
    coordinates = ast.literal_eval(output_text)
except Exception:
    print("Invalid output. Stopping.")
    exit()

if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
    print("Invalid coordinate format. Stopping.")
    exit()


normalized_x = float(coordinates[0]) / resized_width
normalized_y = float(coordinates[1]) / resized_height

absolute_x = normalized_x * image.width
absolute_y = normalized_y * image.height

screen_width, screen_height = pyautogui.size()

click_x = absolute_x * screen_width / image.width
click_y = absolute_y * screen_height / image.height

print("Predicted click:", click_x, click_y)

result_image = draw_point(img_path, [normalized_x, normalized_y], radius=12)
import datetime

ANNOTATED_DIR = r"C:\Users\areeg\OneDrive\Desktop\DON'T OPEN - ALLAA'S\Notepad Identifcation\annotated_screenshots"
os.makedirs(ANNOTATED_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = os.path.join(ANNOTATED_DIR, f"notepad_{timestamp}.png")

result_image.save(file_path)
print("Saved annotated image:", file_path)

is_verified = verify_notepad_icon(
    model=model,
    processor=processor,
    screenshot_path=img_path,
    normalized_x=normalized_x,
    normalized_y=normalized_y
)

if not is_verified:
    print("Predicted point was not verified as Notepad. Stopping safely.")
    exit()


print("Verified Notepad icon. Opening...")

pyautogui.moveTo(click_x, click_y, duration=0.3)
pyautogui.click()
pyautogui.press("enter")
time.sleep(2)


if not wait_for_notepad():
    print("Notepad did not open. Detection was probably wrong. Stopping safely.")
    exit()


post = fetch_first_post()
type_and_save_post(post)
