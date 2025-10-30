from PySide6.QtCore import QStandardPaths, QDir, QTimer, QEvent, QFileInfo, Qt, Signal, QThread, QSettings
from PySide6.QtGui import QAction, QKeySequence, QCloseEvent, QIcon, QImage
from PySide6.QtWidgets import QMainWindow, QMessageBox, QLabel, QApplication, QFileDialog, QToolBar, QDockWidget, QWidget, QVBoxLayout

import json
import os

import numpy as np

from windows.mplcanvas import MplCanvas
from windows.mplplot import MplPlot
from windows.mplplot import PlotWindow

import model
from windows.parameterwindow import ParameterWindow

class MainWindow(QMainWindow):
    def __init__(self):
        application_path = os.path.abspath(os.path.dirname(__file__)) + os.sep
        QMainWindow.__init__(self)
        #self.setWindowIcon(QIcon(application_path + "/images/psf.ico"))
        

        # Make sure the %appdata%/demoapp directory exists
        appdata_directory = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        picture_directory = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        QDir(appdata_directory).mkpath(".")
        

        self.psf_directory = picture_directory + "/PSF"
        QDir(self.psf_directory).mkpath(".")

        #self.display = ImageView(self)
        self.display = MplCanvas(self)

        self.intensity = None
        self.mode = "scat"

        self.parameters_file = appdata_directory + "/parameters.json"


        self.parameter_window = ParameterWindow("Parameters", self)
        self.parameter_window.update_psf.connect(self.update_psf)
        self.update_psf(self.parameter_window.params)
        self.parameter_window.sweep_params.connect(self.sweep)
        self.parameter_window.fps_changed.connect(self.display.set_fps)

        
        
        self.plot_window = PlotWindow()
        self.plot = self.plot_window.plot
        

        self.plot_window.hide()
        
        self.addDockWidget(Qt.RightDockWidgetArea, self.parameter_window)
        

        self.createUI()
            
    
    def closeEvent(self, event):
        # with open(self.parameters_file, 'w') as file:
        #     json.dump(asdict(self.parameter_window.params), file)
        QApplication.quit()

    def createUI(self):
        self.resize(1024, 600)

        #=========#
        # Actions #
        #=========#
        application_path = os.path.abspath(os.path.dirname(__file__)) + os.sep
        
        self.show_parameters_act = QAction("&Parameters", self)
        self.show_parameters_act.setStatusTip("Show parameter panel")
        self.show_parameters_act.triggered.connect(lambda: self.parameter_window.setVisible(not self.parameter_window.isVisible()))
        self.show_parameters_act.setShortcut(Qt.CTRL | Qt.SHIFT | Qt.Key_P)

        self.show_plot_act = QAction("&Show Plot", self)
        self.show_plot_act.setStatusTip("Show plot panel")
        self.show_plot_act.triggered.connect(lambda: self.plot_window.setVisible(not self.plot_window.isVisible()))
        self.show_plot_act.setShortcut(Qt.CTRL | Qt.SHIFT | Qt.Key_S)

        self.set_scat_act = QAction("&Scattering", self)
        self.set_scat_act.setStatusTip("Show scattering psf")
        self.set_scat_act.triggered.connect(lambda: self.display.set_mode("scattering"))

        self.set_if_act = QAction("&Interference", self)
        self.set_if_act.setStatusTip("Show interference psf")
        self.set_if_act.triggered.connect(lambda: self.display.set_mode("interference"))
        
        self.set_sig_act = QAction("&Signal", self)
        self.set_sig_act.setStatusTip("Show total signal")
        self.set_sig_act.triggered.connect(lambda: self.display.set_mode("signal"))

        self.set_viridis_act = QAction("viridis cmap", self)
        self.set_viridis_act.triggered.connect(lambda: self.display.set_cmap("viridis"))
        
        self.set_gray_act = QAction("grayscale cmap", self)
        self.set_gray_act.triggered.connect(lambda: self.display.set_cmap("gray"))

        self.save_figure_act = QAction("Save", self)
        self.save_figure_act.setStatusTip("Save figure")
        self.save_figure_act.triggered.connect(self.display.save)
        self.save_figure_act.setShortcut(QKeySequence.Save)
        
        self.exit_act = QAction("E&xit", self)
        self.exit_act.setShortcut(QKeySequence.Quit)
        self.exit_act.setStatusTip("Exit program")
        self.exit_act.triggered.connect(self.close)

        #=========#
        # Menubar #
        #=========#

        # edit_menu = self.menuBar().addMenu("&Edit")
        # edit_menu.addAction(self.edit_parameters_act)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.save_figure_act)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_act)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.show_parameters_act)
        view_menu.addAction(self.show_plot_act)
        view_menu.addSeparator()
        view_menu.addAction(self.set_scat_act)
        view_menu.addAction(self.set_if_act)
        view_menu.addAction(self.set_sig_act)
        view_menu.addAction(self.set_gray_act)
        view_menu.addAction(self.set_viridis_act)

        
        



        #=========#
        # Toolbar #
        #=========#

        # toolbar = QToolBar(self)
        # self.addToolBar(Qt.TopToolBarArea, toolbar)
        # toolbar.addAction(self.device_select_act)
        # toolbar.addAction(self.device_properties_act)
        # toolbar.addSeparator()
        # toolbar.addAction(self.trigger_mode_act)
        # toolbar.addSeparator()
        # toolbar.addAction(self.start_live_act)
        # toolbar.addSeparator()
        # toolbar.addAction(self.subtract_background_act)
        # toolbar.addAction(self.set_roi_act)
        # toolbar.addAction(self.move_act)
        # toolbar.addSeparator()
        # toolbar.addAction(self.snap_background_act)
        # toolbar.addAction(self.snap_raw_photo_act)
        # toolbar.addAction(self.snap_processed_photo_act)
        # toolbar.addAction(self.z_sweep_act)



        self.setCentralWidget(self.display)
        

        # self.statusBar().showMessage("Ready")
        # self.aquisition_label = QLabel("", self.statusBar())
        # self.statusBar().addPermanentWidget(self.aquisition_label)
        # self.statistics_label = QLabel("", self.statusBar())
        # self.statusBar().addPermanentWidget(self.statistics_label)
        # self.statusBar().addPermanentWidget(QLabel("  "))
        # self.camera_label = QLabel(self.statusBar())
        # self.statusBar().addPermanentWidget(self.camera_label)

        # self.update_statistics_timer = QTimer()
        # self.update_statistics_timer.timeout.connect(self.onUpdateStatisticsTimer)
        # self.update_statistics_timer.start()
    
    def update_psf(self, params: dict):
        # pxsize is necessary for scalebar
        pxsize = params['pxsize'] if 'pxsize' in params else model.pxsize
        magnification = params['magnification'] if 'magnification' in params else model.magnification
        pxsize_obj = pxsize/magnification
        signal=['scattering', 'interference', 'signal']

        intensity = model.simulate_camera(signal=signal, **params)
        self.intensity = {s:I for s, I in zip(signal, intensity)}
        self.display.update_image(self.intensity, pxsize_obj)
    
    def sweep(self, params: dict):
        pxsize = params['pxsize'] if 'pxsize' in params else model.pxsize
        magnification = params['magnification'] if 'magnification' in params else model.magnification

        signal = ['scattering', 'interference', 'signal']
        intensity = model.simulate_camera(signal=signal,**params)
        self.intensity = {s:I for s, I in zip(signal, np.moveaxis(intensity, 1, 0))}
        pxsizes = (pxsize/magnification)

        param_name, param = [(k,v) for (k,v) in params.items() if isinstance(v, np.ndarray)][0]
        
        self.display.update_animation(self.intensity, pxsizes, param_name, param)
        self.plot.plot(self.intensity, param)