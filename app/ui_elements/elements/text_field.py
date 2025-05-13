from dataclasses import dataclass
import cv2
import numpy as np

import pytesseract
from pytesseract import image_to_data, TesseractError

from ui_elements.elements.ui_element import UIElement, UIBox
from ui_elements.elements.box import Box
from image_processing import ImageProcessor

@dataclass
class TextField(UIBox):

    region: Box

    def _to_region(self) -> tuple[int, int, int, int]:
        """Convert to pyautogui-style region: (left, top, width, height)"""
        return self.region._to_region()


    def click(self):
        """Click on text field"""
        self.region.click()
    

    def key_and_click(self, key: str, x: int = None, y: int = None):
        """Press a key or set of keys and click on a text field."""
        self.region.key_and_click(key, x=x, y=y)


    def read_text(self) -> tuple[str, float]:
        """Extracts text and average confidence from the TextField region using OCR."""
        return self.region.read_text()


    def contains(self, text: str, confidence: float = 0.5) -> bool:
        """
        Checks if the given text is present in the OCR-extracted text from the TextField region.
        Case-insensitive match.
        """
        return self.region.contains(text, confidence=confidence)


    def show_debug_overlay(self, label: str = "TextField", duration: int = 2):
        self.region.show_debug_overlay(label=label, duration=duration)


    def screenshot(self):
        return self.region.screenshot()


    def click_button_by_color(self, lower_hsv, upper_hsv, label="Target",
                              button: str = "LEFT", keys: str = None):
        """Finds and clicks a colored icon within this box using OpenCV."""
        return self.region.click_button_by_color(lower_hsv, upper_hsv, label=label,
                                                 button=button, keys=keys)


    def click_button_by_text(self, text: str, keys: str = None, confidence: float = 0.3):
        """Finds and clicks a button within this box by its visible text using pytesseract."""
        return self.region.click_button_by_text(text, keys=keys, confidence=confidence)


    # === Example Usage ===
    # # Capture a text field region (inherits Box behavior)
    # textfield = UIElementCalibrator.get_text_field("Username Field")

    # # Click inside the field
    # textfield.click()

    # # Show a live debug overlay
    # textfield.show_debug_overlay("Username Field", duration=3)

    # # Save screenshot of field
    # img = textfield.screenshot()
    # img.save("username_field.png")