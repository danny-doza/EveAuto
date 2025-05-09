# mouse_overlay.py
import sys
import Quartz
from PyQt5 import QtWidgets, QtCore

def get_mouse_position():
    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return int(loc.x), int(loc.y)

class MouseCoordOverlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self.label = QtWidgets.QLabel(self)
        self.label.setStyleSheet("""
            color: white;
            background-color: rgba(0, 0, 0, 180);
            padding: 5px 8px;
            border-radius: 6px;
            font: 12pt 'Courier';
        """)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(30)

        self.resize(120, 40)


    def update_position(self):
        x, y = get_mouse_position()
        self.label.setText(f"{x}, {y}")
        self.label.adjustSize()
        self.move(x + 15, y + 15)
