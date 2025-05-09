# log_overlay.py
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtWidgets

from ui_elements.app_context import get_app

class LogOverlay(QLabel):
    def __init__(self, text="", x: int=20, y=20, gravity="topleft"):
        super().__init__()
        self.setText(text)
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool |
            Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                padding: 10px;
                border-radius: 8px;
                font-size: 14px;
            }
        """)
        self.adjustSize()
        overlay_width = self.sizeHint().width()
        overlay_height = self.sizeHint().height()

        if gravity == "center":
            self.move(x - overlay_width // 2, y - overlay_height // 2)
        elif gravity == "topright":
            self.move(x - overlay_width, y)
        elif gravity == "bottomleft":
            self.move(x, y - overlay_height)
        elif gravity == "bottomright":
            self.move(x - overlay_width, y - overlay_height)
        else:  # default to "topleft"
            self.move(x, y)
        self.show()


    def update_text(self, new_text):
        self.setText(new_text)
        self.adjustSize()
