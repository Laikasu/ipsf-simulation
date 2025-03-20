from PySide6.QtCore import QStandardPaths, QDir, QTimer, QEvent, QFileInfo, Qt, Signal, QThread, QSettings
from PySide6.QtGui import QAction, QKeySequence, QCloseEvent, QIcon, QImage
from PySide6.QtWidgets import QMainWindow, QMessageBox, QLabel, QApplication, QFileDialog, QToolBar

import json
import os
from dataclasses import asdict, replace


from windows.imageview import ImageView
from windows.mplcanvas import MplCanvas
from model import *

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
        
        self.addDockWidget(Qt.RightDockWidgetArea, self.parameter_window)

        self.createUI()
            
    
    def closeEvent(self, event):
        # with open(self.parameters_file, 'w') as file:
        #     json.dump(asdict(self.parameter_window.params), file)
        QApplication.quit()

    def createUI(self):
        self.resize(1024, 768)

        #=========#
        # Actions #
        #=========#
        application_path = os.path.abspath(os.path.dirname(__file__)) + os.sep
        
        self.show_parameters_act = QAction("&Parameters", self)
        self.show_parameters_act.setStatusTip("Select a video capture device")
        self.show_parameters_act.triggered.connect(self.show_parameter_window)

        self.set_scat_act = QAction("&Scatter", self)
        self.set_scat_act.setStatusTip("Show scatter psf")
        self.set_scat_act.triggered.connect(lambda: self.set_mode("scat"))

        self.set_if_act = QAction("&Interference", self)
        self.set_if_act.setStatusTip("Show interference psf")
        self.set_if_act.triggered.connect(lambda: self.set_mode("if"))
        
        self.set_tot_act = QAction("&Total", self)
        self.set_tot_act.setStatusTip("Show total signal")
        self.set_tot_act.triggered.connect(lambda: self.set_mode("tot"))

        self.save_figure_act = QAction("Save", self)
        self.save_figure_act.setStatusTip("Save figure")
        self.save_figure_act.triggered.connect(self.display.save)
        
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

        

        # device_menu = self.menuBar().addMenu("&Device")
        # device_menu.addAction(self.device_select_act)
        # device_menu.addAction(self.device_properties_act)
        # device_menu.addAction(self.device_driver_properties_act)
        # device_menu.addAction(self.set_roi_act)
        # device_menu.addAction(self.move_act)
        # device_menu.addAction(self.trigger_mode_act)
        # device_menu.addAction(self.start_live_act)
        # device_menu.addSeparator()
        # device_menu.addAction(self.close_device_act)

        psf_menu = self.menuBar().addMenu("&PSF")
        psf_menu.addAction(self.set_scat_act)
        psf_menu.addAction(self.set_if_act)
        psf_menu.addAction(self.set_tot_act)

        
        



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

    def set_mode(self, mode):
        self.mode = mode
        if self.intensity is not None:
            if isinstance(self.intensity, list):
                self.display.update_animation([intensity[self.mode] for intensity in self.intensity])
            else:
                self.display.update_image(self.intensity[self.mode])
    
    def show_parameter_window(self):
        self.parameter_window.show()
    
    def update_psf(self, params: DesignParams):
        camera = Camera(params)
        pxsize_obj = params.pxsize/params.magnification
        scatter_field = calculate_scatter_field(params)
        self.intensity = calculate_intensities(scatter_field, params, camera, r_resolution=self.parameter_window.rresolution)
        self.display.update_image(self.intensity[self.mode], pxsize_obj)
    
    def sweep(self, params: DesignParams, param_name: str, param: np.ndarray):
        self.intensity = []

        params = replace(params)
        pxsizes = []
        for p in param:
            setattr(params, param_name, p)
            camera = Camera(params)
            scatter_field = calculate_scatter_field(params)
            intensity = calculate_intensities(scatter_field, params, camera, r_resolution=self.parameter_window.rresolution)
            self.intensity.append(intensity)
            pxsizes.append(params.pxsize/params.magnification)
        
        self.display.update_animation([intensity[self.mode] for intensity in self.intensity], pxsizes)