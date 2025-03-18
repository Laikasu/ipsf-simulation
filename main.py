from PySide6.QtWidgets import QApplication


from windows.mainwindow import MainWindow

def main():
    app = QApplication()
    app.setApplicationName("psf simulator")
    app.setApplicationDisplayName("PSF Simulator")
    app.setStyle("fusion")

    w = MainWindow()
    w.show()
    

    app.exec()
    

if __name__ == "__main__":
    main()