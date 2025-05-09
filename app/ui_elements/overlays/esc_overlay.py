import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication
from PyQt5.QtCore import Qt, QTimer, QEventLoop

class EscOverlay(QWidget):
    def __init__(self, message="Press ESC to Continue", timeout=None):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Dialog
        )
        self.setWindowModality(Qt.ApplicationModal)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 160);
            }
            QLabel {
                font-size: 18px;
                color: white;
            }
        """)

        from PyQt5.QtWidgets import QLabel
        label = QLabel(message)
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

        self.resize(QApplication.primaryScreen().geometry().size())
        self.move(0, 0)
        self.installEventFilter(self)

        self.esc_pressed = False

        if timeout:
            QTimer.singleShot(timeout * 1000, self.close)

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self.esc_pressed = True
            self.close()
            return True
        return super().eventFilter(obj, event)

def show_esc_overlay(message="Press ESC to Continue", timeout=None):
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    overlay = EscOverlay(message, timeout)
    overlay.setFocusPolicy(Qt.StrongFocus)
    overlay.setFocus()

    overlay.show()
    app.processEvents()

    loop = QEventLoop()
    overlay.destroyed.connect(loop.quit)
    loop.exec_()

    if created_app:
        app.quit()

    return overlay.esc_pressed