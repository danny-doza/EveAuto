from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QApplication, QLabel

class ContinueOverlay(QDialog):
    def __init__(self, message="Click to Continue", timeout=None):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setWindowModality(Qt.ApplicationModal)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.setStyleSheet("background-color: rgba(0, 0, 0, 48);")

        self.result = None

        self.button = QPushButton(message)
        self.button.setFixedSize(200, 50)
        self.button.setStyleSheet("""...""")
        self.button.clicked.connect(self.confirm)

        layout = QVBoxLayout()
        label = QLabel("Click to continue")
        label.setStyleSheet("color: white; font-size: 24px;")
        layout.addWidget(label, alignment=Qt.AlignCenter)
        layout.addWidget(self.button)
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

        #self.resize(300, 100)
        self.center()

        if timeout:
            QTimer.singleShot(timeout * 1000, self.accept)

    def showEvent(self, event):
        self.activateWindow()
        self.raise_()
        super().showEvent(event)

    def closeEvent(self, event):
        super().closeEvent(event)

    def confirm(self):
        self.result = True
        self.accept()

    def mousePressEvent(self, event):
        button_rect = self.button.geometry()
        button_rect.moveTopLeft(self.button.mapToGlobal(button_rect.topLeft()))
        if not button_rect.contains(event.globalPos()):
            self.result = False
            self.reject()

    def center(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)