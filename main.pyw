from PySide6.QtWidgets import QApplication, QMessageBox
from windows.mainwindow import MainWindow
import sys
import traceback


def excepthook(exc_type, exc_value, exc_traceback):
    # Print to console
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    # Optional: show a message box
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Error")
    msg.setText(f"{exc_type.__name__}: {exc_value}")
    msg.exec()

sys.excepthook = excepthook


def main():
    app = QApplication()
    app.setApplicationName("ipsf-simulator")
    app.setApplicationDisplayName("iPSF Simulation")
    app.setStyle("fusion")

    w = MainWindow()
    w.show()
    

    app.exec()
    

if __name__ == "__main__":
    main()