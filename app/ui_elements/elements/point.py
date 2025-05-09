from dataclasses import dataclass
import pyautogui
from pynput.keyboard import Key
from pynput.mouse import Button
import random
import time

from ui_elements.elements.ui_element import UIElement
from ui_elements.overlays.overlays import Overlays

from input.emulator import mouse, keyboard, emulator
from actions import Actions

@dataclass
class Point(UIElement):
    
    x: int
    y: int

    def _to_region(self, width: int = 20, height: int = 20) -> tuple[int, int, int, int]:
        """Convert to pyautogui-style region: (left, top, width, height)"""
        return (int(self.p1.x), int(self.p1.y), width, height)


    def click(self, button: str = "LEFT", keys: str = None):
        """Clicks on the point."""

        if self.x is None or self.y is None:
            print("Point x and y coords are missing, unable to click.")
            return False
        
        return Actions.click(self.x, self.y, button=button, keys=keys)


    def key_and_click(self, keys: str, button: str = "LEFT"):
        """Press a key or set of keys and click on the point."""

        if self.x is None or self.y is None:
            print("Point x and y coords are missing, unable to click.")
            return False

        return Actions.key_and_click(self.x, self.y, keys, button=button)


    def show_debug_overlay(self, label: str = "Point", duration: int = 2):
        """Highlights the point on screen with a small circle."""
        size = 20
        half = size // 2
        Overlays.show_highlight_overlay(self.x - half, self.y - half, size, size,
                                        label=label, duration=duration, shape='circle')


    def screenshot(self):
        """Capture a small square around the point."""
        size = 20
        half = size // 2
        return pyautogui.screenshot(region=(self.x - half, self.y - half, size, size))


    # === Example Usage ===
    # # Capture a single point
    # point = UIElementCalibrator.get_point("Login Button Center")

    # # Perform a click
    # point.click()

    # # Show a debug overlay (small green circle)
    # point.show_debug_overlay("Login Button Center", duration=9)

    # # Screenshot the region around the point
    # screenshot = point.screenshot()
    # screenshot.save("point_debug.png")