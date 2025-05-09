import json
import os
import pyautogui
import threading

from ui_elements.elements.point import Point
from ui_elements.elements.box import Box
from ui_elements.elements.text_field import TextField

from ui_elements.overlays.input_overlay import InputCaptureDialog
from ui_elements.overlays.overlays import Overlays

from pynput import keyboard, mouse

class UIElementCalibrator:

    _overlay_app = None
    _cache_path = "element_cache.json"
    _cache = {}

    @classmethod
    def _load_cache(cls):
        if os.path.exists(cls._cache_path):
            with open(cls._cache_path, "r") as f:
                raw = json.load(f)
                for k, v in raw.items():
                    if v["type"] == "Point":
                        cls._cache[k] = Point(*v["value"])
                    elif v["type"] == "Box":
                        cls._cache[k] = Box(Point(*v["p1"]), Point(*v["p2"]), v.get("scrollable", False))
                    elif v["type"] == "TextField":
                        b = v["box"]
                        box = Box(Point(*b["p1"]), Point(*b["p2"]), b.get("scrollable", False))
                        cls._cache[k] = TextField(box)
        else:
            cls._cache = {}


    @classmethod
    def _save_cache(cls):
        serializable = {}
        for k, v in cls._cache.items():
            if isinstance(v, Point):
                serializable[k] = {"type": "Point", "value": [v.x, v.y]}
            elif isinstance(v, Box):
                serializable[k] = {
                    "type": "Box",
                    "p1": [v.p1.x, v.p1.y],
                    "p2": [v.p2.x, v.p2.y],
                    "scrollable": v.scrollable
                }
            elif isinstance(v, TextField):
                b = v.region
                serializable[k] = {
                    "type": "TextField",
                    "box": {
                        "p1": [b.p1.x, b.p1.y],
                        "p2": [b.p2.x, b.p2.y],
                        "scrollable": b.scrollable
                    }
                }
        with open(cls._cache_path, "w") as f:
            json.dump(serializable, f, indent=2)


    @staticmethod
    def _start_esc_listener(esc_event: threading.Event):
        def on_press(key):
            if key == keyboard.Key.esc:
                esc_event.set()
                return False  # Stop listener

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener


    @staticmethod
    def _wait_for_input(prompt: str) -> Point | None:
        
        screen_center = pyautogui.size()
        center_x = screen_center.width // 2
        center_y = screen_center.height // 2 - 25
        Overlays.show_log_overlay(prompt, x=center_x, y=center_y, gravity="center")

        dialog = InputCaptureDialog()
        pos = dialog.get_click_position()
        
        print(f"[DEBUG] InputCaptureDialog returned: {pos}")
        return Point(pos.x(), pos.y()) if pos else None
    

    @staticmethod
    def _wait_for_enter():
        Overlays.show_continue_overlay("Press to Continue")


    @classmethod
    def get_point(cls, label: str = "a point") -> Point:
        point = cls._wait_for_input(f"Click to select {label}...")
        if point is None:
            return None
        return point  


    @classmethod
    def get_box(cls, label: str = "a box", scrollable: bool = False) -> Box:
        
        screen_center = pyautogui.size()
        center_x = screen_center.width // 2
        center_y = screen_center.height // 2 - 50
        instructions = f"Define {label} with two clicks on the top-left and bottom-right corners of the field"
        Overlays.show_log_overlay(instructions, x=center_x, y=center_y, gravity="center", wipe_existing=True)

        p1 = cls._wait_for_input("Click top-left corner...")
        if not p1:
            return None
        p2 = cls._wait_for_input("Click bottom-right corner...")
        if not p2:
            return None
        return Box(p1, p2, scrollable)


    @classmethod
    def get_text_field(cls, label: str = "a text field") -> TextField:
        box = cls.get_box(label)
        if box is None:
            return None
        return TextField(box)


    @classmethod
    def get_ui_elements(cls, elements: dict[str, object]) -> dict[str, object]:
        """
        Accepts a dict of {label: type | str}, where type is a UIElement class
        and str is an instruction prompt that waits for user ENTER.
        """
        cls._load_cache()

        results = {}

        for label, element_type in elements.items():
            if isinstance(element_type, str):
                # Instruction step
                screen_center = pyautogui.size()
                center_x = screen_center.width // 2
                center_y = screen_center.height // 2 - 50  # Adjusted for taskbar
                Overlays.show_log_overlay(f"\n=== Instruction: {element_type} ===\nPress ENTER when ready...", duration=5, x=center_x, y=center_y, gravity="center", wipe_existing=True)
                cls._wait_for_enter()
                continue

            if label in cls._cache:
                cached = cls._cache[label]
                cached.show_debug_overlay(label=f"(Cached) {label}", duration=3)
                prev_confirmed = Overlays.show_continue_overlay(
                    f"Is this still good for '{label}'?\nClick to confirm or close to redefine."
                )
                
                if prev_confirmed:
                    print(f"User accepted prev definition for ui-element: {label}")
                    
                    results[label] = cached
                    continue

            if element_type is Point:
                value = cls.get_point(label)
            elif element_type is Box:
                value = cls.get_box(label)
            elif element_type is TextField:
                value = cls.get_text_field(label)
            else:
                raise TypeError(f"Unsupported element type: {element_type}")

            if value is None:
                print(f"Skipping {label}...")
                continue

            results[label] = value
            value.show_debug_overlay(label=label, duration=5)

            cls._cache[label] = value
        
        cls._save_cache()
        return results
