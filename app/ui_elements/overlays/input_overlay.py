from PyQt5.QtWidgets import QDialog, QApplication, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QPoint

class InputCaptureDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setWindowModality(Qt.ApplicationModal)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.setStyleSheet("background-color: rgba(0, 0, 0, 48);")

        # Add debug label
        layout = QVBoxLayout()
        label = QLabel("Overlay Active — Click to continue")
        label.setStyleSheet("color: white; font-size: 24px;")
        layout.addWidget(label, alignment=Qt.AlignCenter)
        self.setLayout(layout)

        self._click_pos = None


    def showEvent(self, event):
        print("[DEBUG] Overlay shown")
        self.grabMouse()
        self.activateWindow()
        self.raise_()


    def closeEvent(self, event):
        print("[DEBUG] Overlay closed")
        self.releaseMouse()


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            print("[DEBUG] ESC key pressed, closing dialog")
            self._click_pos = None
            self.reject()
        else:
            super().keyPressEvent(event)


    def mousePressEvent(self, event):
        print(f"[DEBUG] Mouse click at {event.globalPos()}")
        self._click_pos = event.globalPos()
        self.accept()


    def get_click_position(self):
        self.exec_()
        # If ESC was pressed, _click_pos is None. If mouse was clicked, _click_pos is set.
        return self._click_pos