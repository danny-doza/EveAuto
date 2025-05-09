import sys
from PyQt5.QtWidgets import QApplication

_app_instance = None

def get_app():
    global _app_instance
    app = QApplication.instance()
    if app is None:
        _app_instance = QApplication(sys.argv)
        return _app_instance
    return app