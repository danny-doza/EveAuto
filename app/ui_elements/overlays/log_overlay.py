from PyQt5.QtWidgets import QWidget, QListWidget, QListWidgetItem, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFontMetrics, QPalette

class LogOverlay(QWidget):
    def __init__(self, x=20, y=20, width=300, max_entries=5, gravity='topleft', timeout=7):
        super().__init__()
        self._x = x
        self._y = y
        self._gravity = gravity
        self.max_entries = max_entries

        # Create a single-shot timer on the overlay
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.close)
        self._close_timer.start(timeout * 1000)

        # Window settings
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        # --- Set up the list widget ---
        self.listwidget = QListWidget()
        # Hide scrollbars, we'll manage scrolling manually
        self.listwidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.listwidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Black background, white text
        palette = self.listwidget.palette()
        palette.setColor(QPalette.Base, Qt.black)
        palette.setColor(QPalette.Text, Qt.white)
        self.listwidget.setPalette(palette)
        # Rounded corners + font sizing
        self.listwidget.setStyleSheet("""
            border-radius: 8px;
            font-size: 14px;
        """)

        # --- Layout and sizing based on font metrics ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self.listwidget)
        self.setLayout(layout)

        fm = QFontMetrics(self.listwidget.font())
        entry_h = fm.height()
        total_h = entry_h * self.max_entries + layout.contentsMargins().top() + layout.contentsMargins().bottom()
        self.setFixedSize(width, total_h)

        # Initial positioning and show
        self.move_to_gravity(x, y, gravity)
        self.show()


    def move_to_gravity(self, x, y, gravity):
        w, h = self.width(), self.height()
        if gravity == 'center':
            self.move(x - w // 2, y - h // 2)
        elif gravity == 'topright':
            self.move(x - w, y)
        elif gravity == 'bottomleft':
            self.move(x, y - h)
        elif gravity == 'bottomright':
            self.move(x - w, y - h)
        else:  # 'topleft'
            self.move(x, y)


    def add_log(self, text: str):
        """Add a new log entry; drop oldest if over limit, and scroll."""
        item = QListWidgetItem(text)
        self.listwidget.addItem(item)
        if self.listwidget.count() > self.max_entries:
            self.listwidget.takeItem(0)
        self.listwidget.scrollToBottom()

        self._close_timer.start(self._close_timer.interval())
