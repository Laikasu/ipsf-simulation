from PySide6.QtWidgets import QApplication


from windows.mainwindow import MainWindow

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