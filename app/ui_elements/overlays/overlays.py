# mouse_overlay.py
import sys
import Quartz
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication

from ui_elements.overlays.continue_overlay import ContinueOverlay
from ui_elements.overlays.highlight_overlay import HighlightOverlay
from ui_elements.overlays.log_overlay import LogOverlay
from ui_elements.overlays.mouse_overlay import MouseCoordOverlay

from rich_log import get_logger
logger = get_logger("Overlays")

class Overlays:
    
    _app = None
    _overlays = []

    @classmethod
    def ensure_app(cls):
        if cls._app is None:
            cls._app = QApplication.instance()
        if cls._app is None:
            cls._app = QApplication(sys.argv)


    @classmethod
    def update_overlay(cls, overlay):
        cls.ensure_app()

        overlay.show()
        overlay.raise_()
        overlay.repaint()
        cls._app.processEvents()


    @classmethod
    def close_overlay(cls, overlay):
        logger.info(f"Attempting to close {overlay}.")

        overlay.close()
        if overlay in cls._overlays:
            cls._overlays.remove(overlay)

    
    @classmethod
    def close_all_overlays(cls):
        for overlay in cls._overlays:
            overlay.close()
        cls._overlays.clear()


    @classmethod
    def show_continue_overlay(cls, message="Click to Continue", timeout=None):
        cls.ensure_app()
        logger.info("Showing continue overlay")

        overlay = ContinueOverlay(message, timeout)
        cls._overlays.append(overlay)
        result = overlay.exec_()
        return overlay.result if overlay.result is not None else False


    @classmethod
    def show_highlight_overlay(cls, x, y, width, height, label=None, duration=9, shape='rect'):
        cls.ensure_app()

        logger.info(f"Highlighting {label} overlay at ({x}, {y}) with size ({width}, {height}) for {duration} seconds.")

        overlay = HighlightOverlay(int(x), int(y), int(width), int(height), label, duration, shape)
        cls._overlays.append(overlay)
        cls.update_overlay(overlay)


    @classmethod
    def show_log_overlay(cls, message, duration=7, x: int = 20, y: int = 20,
                         gravity="topleft", wipe_existing: bool = False):
        cls.ensure_app()
        logger.info("Showing log overlay")

        # Remove any existing log overlays if requested
        if wipe_existing:
            for ov in cls._overlays[:]:
                if isinstance(ov, LogOverlay):
                    ov.close()
                    cls._overlays.remove(ov)

        # If a log overlay already exists, append to it; otherwise create a new one
        overlay = next((o for o in cls._overlays if isinstance(o, LogOverlay)), None)
        if overlay:
            overlay.add_log(message)
        else:
            overlay = LogOverlay(x=x, y=y, gravity=gravity)
            overlay.add_log(message)
            cls._overlays.append(overlay)
            cls.update_overlay(overlay)


    @classmethod
    def show_mouse_overlay(cls):
        cls.ensure_app()

        logger.info("Showing mouse overlay")

        overlay = MouseCoordOverlay()
        cls._overlays.append(overlay)
        cls.update_overlay(overlay)

        sys.exit(cls._app.exec_())