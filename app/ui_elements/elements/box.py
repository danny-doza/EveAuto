import cv2
from dataclasses import dataclass
from difflib import SequenceMatcher
import Levenshtein
import numpy as np
from PIL import Image
import pyautogui
import pytesseract
from pytesseract import TesseractError
import random
import re
import time

from pynput.keyboard import Key
from pynput.mouse import Button

from image_processing import ImageProcessor

from ui_elements.elements.ui_element import UIBox
from ui_elements.elements.point import Point
from ui_elements.overlays.overlays import Overlays

from input.emulator import emulator, mouse, keyboard
from actions import Actions

from rich_log import get_logger

# Define logger
logger = get_logger("Box")

@dataclass
class Box(UIBox):

    p1: Point
    p2: Point
    scrollable: bool = False
    subelements: dict = None


    def __post_init__(self):
        if self.subelements is None:
            self.subelements = {}

    def _center(self) -> tuple[int, int]:
        x = self.p1.x + (self.p2.x - self.p1.x) // 2
        y = self.p1.y + (self.p2.y - self.p1.y) // 2


    def _to_region(self) -> tuple[int, int, int, int]:
        """Convert to pyautogui-style region: (left, top, width, height)"""
        width = int(self.p2.x - self.p1.x)
        height = int(self.p2.y - self.p1.y)
        return (int(self.p1.x), int(self.p1.y), width, height)


    def _region_contains_text(self, text: str, x: int, y: int, w: int = 40, h: int = 20,
                              scale_factor: float = 2.0,
                              threshold: int = 124, lower_hsv=None, upper_hsv=None,
                              confidence: float = 0.8) -> bool:
        region = (x, y, w, h)
        screenshot = pyautogui.screenshot(region=region)
        data = ImageProcessor.extract_text_data(screenshot, title=text, scale_factor=scale_factor,
                                                threshold=threshold, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                                                debug=True)
        for i in range(len(data['text'])):
            word = data['text'][i]
            logger.debug(f"{word}")

            match_confidence = Levenshtein.ratio(word.strip().lower(), text.strip().lower())
            # Try combining with previous word if applicable
            if i > 0:
                prev_word = data['text'][i - 1].strip().lower()
                combined_word = f"{prev_word} {word}"
                combined_confidence = Levenshtein.ratio(combined_word, text.strip().lower())

                # If the combined word is a better match, use that instead
                if combined_confidence > match_confidence:
                    word = combined_word
                    match_confidence = combined_confidence


            if match_confidence > confidence:
                return True
    
        return False


    def read_text(self,
                  scale_factor: float = 2.0,
                  threshold: int = 124, lower_hsv=None, upper_hsv=None,
                  confidence: float = 0.5) -> tuple[str, float]:
        """Extracts text and average confidence from the TextField region using OCR."""
        screenshot = self.screenshot()
        scale_factor = 2
        data = ImageProcessor.extract_text_data(screenshot, scale_factor=scale_factor,
                                                threshold=threshold, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                                                debug=True)

        if 'text' not in data or not data['text']:
            return None

        extracted_text = []
        confidences = []
        for i in range(1, len(data['text'])):
            word = data['text'][i]
            match_confidence = Levenshtein.ratio(word.strip().lower(), word.strip().lower())
            if match_confidence >= confidence:
                extracted_text.append(word)
                confidences.append(match_confidence)

        final_text = " ".join(extracted_text)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        return final_text, avg_conf


    def contains(self, text: str,
                 threshold: int = 124, lower_hsv=None, upper_hsv=None,
                 confidence: float = 0.5) -> bool:
        """
        Checks if the given text is present in the OCR-extracted text from the TextField region.
        Case-insensitive match.
        """
        x, y, w, h = self._to_region()
        if self._region_contains_text(text, x, y, w, h,
                                      threshold=threshold, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                                      confidence=confidence):
            return True
        return False


    def click(self, button: str = "LEFT", keys: str = None):
        """Clicks the center of the box."""

        if self.p1 is None or self.p2 is None:
            logger.warning("Point coords are missing, unable to click.")
            return False
        else:
            x = self.p1.x + (self.p2.x - self.p1.x) // 2
            y = self.p1.y + (self.p2.y - self.p1.y) // 2

        return Actions.click(x, y, button=button, keys=keys)


    def key_and_click(self, keys: str, button: str = "LEFT"):
        """Press a key or set of keys and click on the point."""

        if self.p1 is None or self.p2 is None:
            logger.warning("Point coords are missing, unable to click.")
            return False
        else:
            x = self.p1.x + (self.p2.x - self.p1.x) // 2
            y = self.p1.y + (self.p2.y - self.p1.y) // 2
        
        return Actions.key_and_click(x, y, keys, button=button)


    def scroll(self, clicks: int = -3, x: int = None, y: int = None):
        """
        Scrolls the mouse wheel within the center of the box.
        
        :param clicks: Number of scroll steps. Negative = up, positive = down.
        :param x: Optional override X position for scroll.
        :param y: Optional override Y position for scroll.
        """
        if x is None or y is None:
            x = self.p1.x + (self.p2.x - self.p1.x) // 2
            y = self.p1.y + (self.p2.y - self.p1.y) // 2

        return Actions.scroll(x, y, clicks)


    def show_debug_overlay(self, label: str = "Box", duration: int = 2):
        """Shows a visual overlay highlighting this box on screen."""
        region = self._to_region()
        Overlays.show_highlight_overlay(region[0], region[1], region[2], region[3], label=label, duration=duration, shape='rect')


    def screenshot(self):
        """Takes a screenshot of the region defined by this box."""
        return pyautogui.screenshot(region=self._to_region())


    def compute_color_confidence(self, mask_roi: np.ndarray) -> float:
        total_pixels = mask_roi.size
        matching_pixels = np.count_nonzero(mask_roi)
        return matching_pixels / total_pixels if total_pixels else 0


    def find_colored_region(self, lower_hsv, upper_hsv, scroll_if_missing=True):
        """
        Finds the region in the box matching the given HSV color range with the highest confidence.

        :param lower_hsv: Lower HSV bound (np.array).
        :param upper_hsv: Upper HSV bound (np.array).
        :param scroll_if_missing: Whether to scroll and retry if not found.
        :return: (x, y, w, h) of bounding rect in box-relative coords or None.
        """
        retry_attempts = 3
        best_region = None
        best_confidence = 0
        
        for _ in range(retry_attempts):
            screenshot = self.screenshot()
            hsv_blurred = ImageProcessor.preprocess_for_color_match(screenshot)
            
            mask = cv2.inRange(hsv_blurred, lower_hsv, upper_hsv)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi_mask = mask[y:y+h, x:x+w]
                    confidence = self.compute_color_confidence(roi_mask)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_region = (x, y, w, h)

                if best_region:
                    logger.info(f"Color region found with confidence: {best_confidence:.2f}")
                    return best_region, best_confidence

                if scroll_if_missing and self.scrollable:
                    logger.info("Color match not found. Scrolling...")
                    self.scroll()

        return None, 0


    def detect_halo(self, lower_hsv, upper_hsv, duration: float = 4.0, interval: float = 0.1, confidence: float = 0.5) -> bool:
        """
        Detects a fade-in and fade-out halo effect in the box area over a time window.

        :param duration: Total observation time (seconds).
        :param interval: Delay between screenshots (seconds).
        :return: True if halo appears and disappears, otherwise False.
        """
        start_time = time.time()
        seen_color = False
        color_disappeared = False

        while time.time() - start_time < duration:
            region, color_confidence = self.find_colored_region(lower_hsv, upper_hsv, scroll_if_missing=False)
            color_present = region is not None and color_confidence >= confidence

            if color_present and not seen_color:
                seen_color = True
                logger.info("Halo appeared.")
            elif seen_color and not color_present:
                color_disappeared = True
                logger.info("Halo disappeared.")
                break

            time.sleep(interval)

        return seen_color and color_disappeared


    def click_button_by_color(self, lower_hsv, upper_hsv, label="Target",
                                    button: str = "LEFT", keys: str = None) -> bool:
        
        region = self.find_colored_region(lower_hsv, upper_hsv)
        if not region:
            logger.warning("No matching color region found.")
            return None

        x, y, w, h = region
        abs_x = self.p1.x + x
        abs_y = self.p1.y + y
        center_x = abs_x + w // 2
        center_y = abs_y + h // 2

        Actions.click(center_x, center_y, keys=keys)
        
        return center_x, center_y


    def click_button_by_text(self, text: str, keys: str = None,
                             threshold: int = 124, lower_hsv=None, upper_hsv=None,
                             confidence: float = 0.5) -> bool:
        """Find and click a button by text with dual-pass OCR for white and gray text."""
        logger.info(f"Attempting to retrieve button with text: {text}")

        if self.subelements and text in self.subelements:
            logger.info(f"Cached value detected for text: {text}")
            cached = self.subelements[text]
            if self._region_contains_text(text, cached['x'], cached['y'], cached['w'], cached['h'],
                                          threshold=threshold, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                                          confidence=0.4):
                logger.info(f"Using cached location for '{text}': ({cached['center_x']}, {cached['center_y']})")
                Actions.click(cached['center_x'], cached['center_y'], keys=keys)
                return cached['x'], cached['y']
            else:
                logger.warning(f"Failed to find word in prev location, reattempting to find word.")

        screenshot = self.screenshot()
        scale_factor = 2
        data = ImageProcessor.extract_text_data(screenshot, title=text, scale_factor=scale_factor,
                                                threshold=threshold, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                                                debug=True)

        if 'text' not in data or not data['text']:
            logger.warning(f"No text found. Skipping check.")
            return None

        for i in range(len(data['text'])):
            word = data['text'][i]
            match_confidence = Levenshtein.ratio(word.strip().lower(), text.strip().lower())
            # Try combining with previous word if applicable
            if i > 0:
                prev_word = data['text'][i - 1].strip().lower()
                combined_word = f"{prev_word} {word}"
                combined_confidence = Levenshtein.ratio(combined_word, text.strip().lower())

                # If the combined word is a better match, use that instead
                if combined_confidence > match_confidence:
                    word = combined_word
                    match_confidence = combined_confidence

            if match_confidence >= confidence:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                w = int(w / scale_factor)
                h = int(h / scale_factor)

                abs_x = self.p1.x + x
                abs_y = self.p1.y + y
                center_x = self.p1.x + x + w // 2
                center_y = self.p1.y + y + h // 2

                logger.info(f"Match found at ({center_x}, {center_y})")
                Actions.click(center_x, center_y, keys=keys)
                self.subelements[text] = {
                    'x': abs_x,
                    'y': abs_y,
                    'w': w,
                    'h': h,
                    'center_x': center_x,
                    'center_y': center_y
                }
                return center_x, center_y

        logger.warning(f"Button with text '{text}' not found in box.")
        return None


    def get_text_location(self, text: str, keys: str = None,
                          threshold: int = 124, lower_hsv=None, upper_hsv=None,
                          confidence: float = 0.5) -> bool:
        """Find and click a button by text with dual-pass OCR for white and gray text."""
        logger.info(f"Attempting to retrieve button with text: {text}")

        if self.subelements and text in self.subelements:
            logger.info(f"Cached value detected for text: {text}")
            cached = self.subelements[text]
            if self._region_contains_text(text, cached['x'], cached['y'], cached['w'], cached['h'],
                                          threshold=threshold, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                                          confidence=0.4):
                logger.info(f"Using cached location for '{text}': ({cached['center_x']}, {cached['center_y']})")
                return cached['x'], cached['y']
            else:
                logger.warning(f"Failed to find word in prev location, reattempting to find word.")

        screenshot = self.screenshot()
        scale_factor = 2
        data = ImageProcessor.extract_text_data(screenshot, title=text, scale_factor=scale_factor,
                                                threshold=threshold, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                                                debug=True)

        if 'text' not in data or not data['text']:
            logger.warning(f"No text found. Skipping check.")
            return None

        for i in range(len(data['text'])):
            word = data['text'][i]
            match_confidence = Levenshtein.ratio(word.strip().lower(), text.strip().lower())
            # Try combining with previous word if applicable
            if i > 0:
                prev_word = data['text'][i - 1].strip().lower()
                combined_word = f"{prev_word} {word}"
                combined_confidence = Levenshtein.ratio(combined_word, text.strip().lower())

                # If the combined word is a better match, use that instead
                if combined_confidence > match_confidence:
                    word = combined_word
                    match_confidence = combined_confidence
            
            if match_confidence >= confidence:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                x = int(x / scale_factor)
                y = int(y / scale_factor)
                w = int(w / scale_factor)
                h = int(h / scale_factor)

                abs_x = self.p1.x + x
                abs_y = self.p1.y + y
                center_x = self.p1.x + x + w // 2
                center_y = self.p1.y + y + h // 2

                logger.info(f"Match found at ({center_x}, {center_y})")
                self.subelements[text] = {
                    'x': abs_x,
                    'y': abs_y,
                    'w': w,
                    'h': h,
                    'center_x': center_x,
                    'center_y': center_y
                }
                return center_x, center_y

        logger.warning(f"Button with text '{text}' not found in box.")
        return None


    def extract_numeric_with_units(self, confidence: float = 0.5):
        UNIT_CONVERSIONS = {
            'm': 1,
            'meter': 1,
            'meters': 1,
            'km': 1000,
            'kilometer': 1000,
            'kilometers': 1000,
            'mi': 1609.34,
            'mile': 1609.34,
            'miles': 1609.34,
            'ft': 0.3048,
            'feet': 0.3048,
            'foot': 0.3048,
            'yd': 0.9144,
            'yard': 0.9144,
            'yards': 0.9144,
        }
        unit_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(\w+)?")

        screenshot = self.screenshot()
        data = ImageProcessor.extract_text_data(screenshot, scale_factor=2.0)
        for i in range(len(data['text'])):
            val_str = data['text'][i].strip()
            unit_str = data['text'][i+1].strip() if i + 1 < len(data['text']) else ''
            try:
                value = float(val_str)
                best_unit = None
                best_ratio = 0.0
                for known_unit in UNIT_CONVERSIONS:
                    levenschtein_ratio = Levenshtein.ratio(unit_str.lower(), known_unit.lower())
                    if levenschtein_ratio > best_ratio and levenschtein_ratio > confidence:
                        best_ratio = levenschtein_ratio
                        best_unit = known_unit
                if best_unit:
                    return value * UNIT_CONVERSIONS[best_unit]
                return value  # Assume meters if no good match
            except ValueError:
                continue


    def has_significant_change(self, duration: float = 5, min_diff_meters: float = 500):
        """
        Detect significant change in numeric value over time, factoring in units.

        Supports values like '23 km', '400 m', '0.5 mi', etc.

        :param duration: How long to monitor (seconds).
        :param min_diff_meters: Minimum change in meters to return True.
        :return: True if significant change is detected.
        """
        start_time = time.time()
        values_in_meters = []


        while time.time() - start_time < duration:
            try:
                value_m = self.extract_numeric_with_units()
                logger.debug(value_m)
                if isinstance(value_m, (int, float)):
                    values_in_meters.append(value_m)
            except Exception as e:
                logger.error(f"Error reading value: {e}")
            time.sleep(0.1)

        if not values_in_meters:
            return False

        max_val = max(values_in_meters)
        min_val = min(values_in_meters)
        logger.info(f"Maximum value: {max_val}, Minimum value: {min_val}")
        return (max_val - min_val) >= min_diff_meters


    def extract_highlighted_text(self, threshold: int = 124) -> str:
        screenshot = self.screenshot()
        data = ImageProcessor.extract_text_data(screenshot,
                                                threshold=threshold)

        if 'text' not in data:
            return ""

        words = [word.strip() for word in data['text'] if word.strip()]
        return " ".join(words)


    # === Example Usage ===
    # # Capture a box region
    # box = UIElementCalibrator.get_box("Information Pane")

    # # Click in the center of the box
    # box.click()

    # # Show a live debug overlay (green rectangle)
    # box.show_debug_overlay("Information Pane", duration=3)

    # # Take a screenshot of the region
    # image = box.screenshot()
    # image.save("info_pane.png")

    # # Click a button within the box by its text
    # box.click_button_by_text("Start Game", confidence=0.8)
