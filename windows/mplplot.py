import numpy as np
import matplotlib
matplotlib.use("Agg")

from typing import List

from scipy.signal import savgol_filter

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QMenuBar
from PySide6.QtCore import QStandardPaths, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class PlotWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Plot")
        self.resize(1024, 768)
        self.plot = MplPlot(self)

        self.save_act = QAction("Save")
        self.save_act.triggered.connect(self.plot.save)
        self.save_act.setShortcut(QKeySequence.Save)

        self.close_act = QAction("Close")
        self.close_act.triggered.connect(self.close)
        self.close_act.setShortcuts([QKeySequence.Quit, QKeySequence.Cancel])

        self.show_interference_act = QAction("Show Interference")
        self.show_interference_act.setCheckable(True)
        self.show_interference_act.setChecked(True)
        self.show_interference_act.toggled.connect(self.plot.set_show_if)

        self.show_scatter_act = QAction("Show Scattering")
        self.show_scatter_act.setCheckable(True)
        self.show_scatter_act.setChecked(True)
        self.show_scatter_act.toggled.connect(self.plot.set_show_scat)

        self.show_sig_act = QAction("Show Signal")
        self.show_sig_act.setCheckable(True)
        self.show_sig_act.setChecked(True)
        self.show_sig_act.toggled.connect(self.plot.set_show_sig)
        
        self.show_derivatives_act = QAction("Show Derivatives")
        self.show_derivatives_act.setCheckable(True)
        self.show_derivatives_act.toggled.connect(self.plot.set_show_derivatives)

        self.menu_bar = QMenuBar()
        file_menu = self.menu_bar.addMenu("File")
        file_menu.addAction(self.save_act)
        file_menu.addSeparator()
        file_menu.addAction(self.close_act)

        view_menu = self.menu_bar.addMenu("View")
        view_menu.addAction(self.show_scatter_act)
        view_menu.addAction(self.show_interference_act)
        view_menu.addAction(self.show_sig_act)
        view_menu.addAction(self.show_derivatives_act)

        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(0,0,0,0)
        plot_layout.setAlignment(Qt.AlignTop)
        plot_layout.addWidget(self.menu_bar,0)
        plot_layout.addWidget(self.plot)
        self.setLayout(plot_layout)


#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


class MplPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=10, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.tight_layout(pad=4)
        super().__init__(fig)
        
        self.axes.set_title('Contrasts change during sweep')
        self.contrast = {}
        self.show_scat = True
        self.show_if = True
        self.show_sig = True
        self.show_derivatives = False
        self.plot_scat: List[Line2D] = None
        self.plot_if: List[Line2D] = None
        self.plot_sig: List[Line2D] = None
        self.plot_scat_deriv: List[Line2D] = None
        self.plot_if_deriv: List[Line2D] = None
        self.plot_sig_deriv: List[Line2D] = None
    
    def set_show_if(self, value):
        self.show_if = value
        if self.plot_if is not None:
            self.update_visibility()
    
    def set_show_scat(self, value):
        self.show_scat = value
        if self.plot_scat is not None:
            self.update_visibility()
    
    def set_show_sig(self, value):
        self.show_sig = value
        if self.plot_sig is not None:
            self.update_visibility()
    
    def set_show_derivatives(self, show):
        if show != self.show_derivatives:
            self.figure.clf()
        self.show_derivatives = show
        self.update_plot()
    
    def update_plot(self):
        if self.show_derivatives:
            self.axes.cla()
            self.axes_deriv = self.figure.add_subplot(122)
            self.axes_deriv.set_title('Derivative of intensity')
            self.axes = self.figure.add_subplot(121)
            self.axes.set_title('Intensity')
            self.plot_scat, = self.axes.plot(self.param, self.contrast['scattering'], label='scattering')
            self.plot_if, = self.axes.plot(self.param, self.contrast['interference'], label='interference')
            self.plot_sig, = self.axes.plot(self.param, self.contrast['signal'], label='signal')
            self.axes.legend()
            smooth_contrast_scat = savgol_filter(self.contrast['scattering'], window_length=5, polyorder=2)
            smooth_contrast_if = savgol_filter(self.contrast['interference'], window_length=5, polyorder=2)
            smooth_contrast_sig = savgol_filter(self.contrast['signal'], window_length=5, polyorder=2)
            self.plot_scat_deriv, = self.axes_deriv.plot(self.param, np.gradient(smooth_contrast_scat, self.param))
            self.plot_if_deriv, = self.axes_deriv.plot(self.param, np.gradient(smooth_contrast_if, self.param))
            self.plot_sig_deriv, = self.axes_deriv.plot(self.param, np.gradient(smooth_contrast_sig, self.param))
        else:
            self.axes.cla()
            self.axes = self.figure.add_subplot(111)
            self.axes.set_title('Intensity')
            self.plot_scat_deriv = None
            self.plot_if_deriv = None
            self.plot_sig_deriv = None
            self.plot_scat, = self.axes.plot(self.param, self.contrast['scattering'], label='scattering')
            self.plot_if, = self.axes.plot(self.param, self.contrast['interference'], label='interference')
            self.plot_sig, = self.axes.plot(self.param, self.contrast['signal'], label='signal')
            self.axes.legend()
        self.update_visibility()



    def update_visibility(self):
        if self.plot_scat_deriv is not None:
            self.plot_scat_deriv.set_visible(self.show_scat)
        if self.plot_if_deriv is not None:
            self.plot_if_deriv.set_visible(self.show_if)
        if self.plot_sig_deriv is not None:
            self.plot_sig_deriv.set_visible(self.show_sig)
        
        self.plot_scat.set_visible(self.show_scat)
        self.plot_if.set_visible(self.show_if)
        self.plot_sig.set_visible(self.show_sig)

        if self.show_derivatives:
            self.axes_deriv.relim(visible_only=True)
            self.axes_deriv.autoscale_view()
        self.axes.relim(visible_only=True)
        self.axes.autoscale_view()
        self.draw_idle()


    def plot(self, intensities, param):
        self.figure.clf()
        self.param = param
        intensity = intensities['scattering']
        N, height, width = np.shape(intensity)
        self.contrast['scattering'] = intensity[:,height//2, width//2]
        
        intensity = intensities['interference']
        N, height, width = np.shape(intensity)
        self.contrast['interference'] = intensity[:,height//2, width//2]

        intensity = intensities['signal']
        N, height, width = np.shape(intensity)
        self.contrast['signal'] = intensity[:,height//2, width//2]
        
        self.update_plot()


    def save(self):
        loc = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Figure", loc,"Image Files(*.png *.jpg *.bmp)")
        if filepath:
            self.figure.savefig(filepath)