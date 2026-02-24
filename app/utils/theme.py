from PySide6 import QtGui, QtWidgets

def apply_dark_palette(app: QtWidgets.QApplication):
    palette = QtGui.QPalette()
    base = QtGui.QColor(20, 20, 22)
    alt = QtGui.QColor(28, 28, 32)
    text = QtGui.QColor(230, 230, 235)
    disabled = QtGui.QColor(150, 150, 160)

    palette.setColor(QtGui.QPalette.Window, base)
    palette.setColor(QtGui.QPalette.WindowText, text)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(16, 16, 18))
    palette.setColor(QtGui.QPalette.AlternateBase, alt)
    palette.setColor(QtGui.QPalette.Text, text)
    palette.setColor(QtGui.QPalette.Button, alt)
    palette.setColor(QtGui.QPalette.ButtonText, text)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(72, 122, 255))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, disabled)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled)
    app.setPalette(palette)

    app.setStyleSheet("""
        QMainWindow { background: #141416; }
        QGroupBox {
            border: 1px solid #2B2B30;
            border-radius: 10px;
            margin-top: 10px;
            padding: 10px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
        QTabWidget::pane { border: 1px solid #2B2B30; border-radius: 10px; }
        QTabBar::tab {
            background: #1C1C20; padding: 8px 12px; margin-right: 4px;
            border-top-left-radius: 8px; border-top-right-radius: 8px;
        }
        QTabBar::tab:selected { background: #2A2A31; }
        QPushButton {
            background: #2A2A31; border: 1px solid #3A3A44;
            padding: 8px 10px; border-radius: 10px;
        }
        QPushButton:hover { background: #343440; }
        QPushButton:pressed { background: #202026; }
        QComboBox, QSpinBox, QLineEdit {
            background: #1A1A1E; border: 1px solid #3A3A44;
            padding: 6px; border-radius: 10px;
        }
        QTextEdit {
            background: #101012; border: 1px solid #2B2B30;
            border-radius: 10px;
        }
        QSlider::groove:horizontal { height: 6px; background: #2B2B30; border-radius: 3px; }
        QSlider::handle:horizontal {
            width: 16px; margin: -6px 0; border-radius: 8px; background: #4B8BFF;
        }
        QScrollArea { border: none; }
        QTableWidget {
            background: #101012;
            border: 1px solid #2B2B30;
            border-radius: 10px;
            gridline-color: #2B2B30;
        }
        QHeaderView::section {
            background: #1C1C20;
            border: 1px solid #2B2B30;
            padding: 6px;
        }
    """)