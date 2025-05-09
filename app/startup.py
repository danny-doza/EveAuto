import subprocess
import random
import pyautogui
import pytesseract
from PIL import ImageGrab, Image
import numpy as np
import cv2

from pathlib import Path

from ui_elements.overlays.overlays import Overlays
from input.emulator import emulator, mouse
from image_processing import ImageProcessor

def get_dir() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent

def launch_or_focus(app_name: str):
    # AppleScript to either focus or launch the app
    script = f'''
    tell application "{app_name}"
        activate
    end tell
    '''

    try:
        subprocess.run(["osascript", "-e", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch or focus {app_name}: {e}")


from Quartz import (
    CGWindowListCreateImage,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
    kCGWindowImageDefault,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowImageNominalResolution,
    CGRectInfinite,
    CGImageGetWidth,
    CGImageGetHeight,
    CGDataProviderCopyData,
    CGImageGetDataProvider,
)
from PIL import Image


def capture_full_resolution_screen() -> Image.Image:
    """
    Captures a full-resolution screenshot on macOS using Quartz.
    """
    image_ref = CGWindowListCreateImage(
        CGRectInfinite,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
        kCGWindowImageDefault | kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution
    )

    width = CGImageGetWidth(image_ref)
    height = CGImageGetHeight(image_ref)

    provider = CGImageGetDataProvider(image_ref)
    data = CGDataProviderCopyData(provider)

    image = Image.frombuffer(
        "RGBA",
        (width, height),
        data,
        "raw",
        "BGRA",
        0,
        1
    ).convert("RGB")

    return image

def preprocess_image_for_ocr(pil_image: Image.Image, upscale: float = 2.0) -> Image.Image:
    """
    Enhances a PIL image for better OCR accuracy using grayscale conversion, 
    resizing, and adaptive thresholding.
    Also saves intermediate steps for debugging.
    """
    debug_dir = get_dir() / "ocr_debug"
    debug_dir.mkdir(exist_ok=True)

    pil_image.save(debug_dir / "00_original.png")

    # Convert to OpenCV format
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(str(debug_dir / "01_grayscale.png"), gray)

    # Optional resize (helps pytesseract on small fonts)
    if upscale != 1.0:
        gray = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(debug_dir / "02_upscaled.png"), gray)

    # Apply adaptive thresholding
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cv2.imwrite(str(debug_dir / "03_threshold.png"), thresh)

    # Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(str(debug_dir / "04_morph.png"), morph)

    return Image.fromarray(morph)

def find_text_on_screen(text: str):
    screen = capture_full_resolution_screen()  # Or capture_full_resolution_screen() on macOS

    text_data = ImageProcessor.extract_text_data(screen)

    # Combine words into lines
    lines = {}
    for i in range(text_data):
        word = text_data["text"][i].strip()
        if not word:
            continue
        line_num = text_data["line_num"][i]
        if line_num not in lines:
            lines[line_num] = {
                "text": [],
                "lefts": [],
                "tops": [],
                "rights": [],
                "bottoms": []
            }
        x, y, w, h = (
            text_data["left"][i],
            text_data["top"][i],
            text_data["width"][i],
            text_data["height"][i]
        )
        lines[line_num]["text"].append(word)
        lines[line_num]["lefts"].append(x)
        lines[line_num]["tops"].append(y)
        lines[line_num]["rights"].append(x + w)
        lines[line_num]["bottoms"].append(y + h)

    for line in lines.values():
        line_text = " ".join(line["text"]).lower()
        if text.lower() in line_text:
            # Calculate bounding box for the line
            x1 = min(line["lefts"])
            y1 = min(line["tops"])
            x2 = max(line["rights"])
            y2 = max(line["bottoms"])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Scale for Retina (Quartz-to-screen coordinate fix)
            scale_factor = screen.size[0] / pyautogui.size().width
            cx = int(cx / scale_factor)
            cy = int(cy / scale_factor)

            print(f"Matched line: '{line_text}' at scaled ({cx}, {cy})")
            return cx, cy

    print("Play Now not found in combined lines.")
    return None


def start_eve():
    button_location = pyautogui.locateOnScreen(f"{get_dir()}/resources/play_now.png", confidence=0.8, grayscale=True)
    if button_location:
        show_debug_overlay(button_location[0], button_location[1], label="Play Now")
        #pyautogui.moveTo(button_location[0], button_location[1], duration=random.uniform(0.5, 0.9))
        mouse.click(pyautogui.center(button_location))
    else:
        print("Play Now button not found on screen. Ensure the game is running and the button is visible.")

def show_debug_overlay(x, y, label: str = "Point", duration: int = 3):
    size = 20
    half = size // 2
    Overlays.show_highlight_overlay(x - half, y - half, size, size, label=label, duration=duration, shape='circle')


def get_profile_position(n: int):
    if n not in [1, 2, 3]:
        raise ValueError("Profile number must be 1, 2, or 3")

    screen_width, screen_height = pyautogui.size()
    section_width = screen_width / 6
    x_padding = screen_width / 12  # 1/12 on each side
    y_center = screen_height / 2

    # x is offset by padding + full sections before it + half a section
    x = x_padding + (n - 1) * section_width + section_width / 2

    return int(x), int(y_center)

def select_profile(n: int):
    if n not in [1, 2, 3]:
        raise ValueError("Profile number must be 1, 2, or 3")
    
    x, y = get_profile_position(n)
    
    show_debug_overlay(x, y)
    #pyautogui.moveTo(x, y, duration=random.uniform(0.5, 0.9))
    mouse.click(x, y)  # Clicks the nth profile