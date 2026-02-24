import os
import sys
from PySide6 import QtGui

def project_root() -> str:
    # app/utils/resources.py -> app/utils -> app -> project_root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def resource_path(rel_path: str) -> str:
    """
    PyInstaller-safe:
    - In packaged build: sys._MEIPASS is the extraction root and contains assets/
    - In dev: use project root
    """
    base = getattr(sys, "_MEIPASS", project_root())
    return os.path.join(base, rel_path)

def set_windows_app_user_model_id(app_id: str):
    """
    Helps Windows taskbar icon + grouping. Call BEFORE QApplication.
    """
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass

def load_app_icon() -> QtGui.QIcon:
    ico = resource_path(os.path.join("assets", "app2_white.ico"))
    if os.path.exists(ico):
        return QtGui.QIcon(ico)

    png = resource_path(os.path.join("assets", "app2_white.png"))
    if os.path.exists(png):
        return QtGui.QIcon(png)

    return QtGui.QIcon()