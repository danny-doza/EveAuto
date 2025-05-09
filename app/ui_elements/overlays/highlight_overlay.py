import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

from ui_elements.app_context import get_app

class HighlightOverlay(QtWidgets.QWidget):
    def __init__(self, x, y, width, height, label=None, duration=2, shape='rect'):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool |
            Qt.WindowTransparentForInput
        )
        #self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)

        self.setGeometry(x, y, width, height)
        self.label = label
        self.duration = duration
        self.shape = shape

        print(f"Highlighting {label} overlay at ({x}, {y}) with size ({width}, {height}) for {duration} seconds.")
        QtCore.QTimer.singleShot(duration * 1000, self.close)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 255), 2))
        painter.setBrush(QtGui.QColor(0, 255, 0, 80))  # semi-transparent green
        painter.fillRect(self.rect(), QtGui.QColor(255, 0, 0, 255))  # solid red

        if self.shape == 'circle':
            diameter = min(self.width(), self.height())
            painter.drawEllipse(0, 0, diameter - 1, diameter - 1)
        else:
            painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        if self.label:
            painter.setPen(QtGui.QColor(255, 255, 255, 255))
            painter.drawText(10, 20, self.label)
