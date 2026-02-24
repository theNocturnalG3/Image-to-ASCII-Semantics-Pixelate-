import sys
from PySide6 import QtWidgets

from app.ui import MainWindow
from app.utils.theme import apply_dark_palette
from app.utils.resources import set_windows_app_user_model_id, load_app_icon

def main():
    set_windows_app_user_model_id("com.nocturnal.ascii_mosaic")

    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)

    icon = load_app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    win = MainWindow()
    if not icon.isNull():
        win.setWindowIcon(icon)

    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()