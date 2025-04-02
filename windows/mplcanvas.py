import numpy as np
import matplotlib
matplotlib.use("Agg")

from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QStandardPaths, QFileInfo
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.animation import FuncAnimation

import tifffile as tiff

import processing as pc

from typing import List

#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, pxsize=3.45):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_axis_off()
        fig.tight_layout(pad=2)
        super().__init__(fig)
        
        self.image: AxesImage = None
        self.anim: FuncAnimation = None
        self.fps = 10
        self.cmap = 'viridis'
        self.mode = 'scat'

        self.data_directory = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)

    def update_image(self, intensities: np.ndarray = None, pxsize: float = None):
        if intensities is not None:
            self.intensities = intensities
        frame = self.intensities[self.mode]
        width, height = frame.shape
        if self.anim is not None:
            self.anim.pause()
            self.anim = None
        if self.image is None:
            self.image = self.axes.imshow(frame, cmap=self.cmap)
            if pxsize is not None:
                self.pxsize = pxsize
                self.image.set_extent((0, pxsize*width, 0, pxsize*height))
            self.colorbar = self.figure.colorbar(self.image)
            scalebar = AnchoredSizeBar(self.axes.transData, 0.2, '200 nm', 'lower center', pad=0.1, frameon=False, color='white', sep=5)
            self.axes.add_artist(scalebar)
        else:
            self.image.set_data(frame)

            # Update pxsize
            if pxsize:
                self.pxsize = pxsize

            self.image.set_extent((0, self.pxsize*width, 0, self.pxsize*height))
            # Double bc it doesnt set them simultaneously
            self.image.set_clim(np.min(frame), np.max(frame))
            self.image.set_clim(np.min(frame), np.max(frame))
            self.draw_idle()

    

    def animate(self, frame: int):
        intensity = self.intensities[frame][self.mode]
        self.image.set_data(intensity)
        if self.pxsizes is not None:
            self.pxsize = self.pxsizes[frame]
        if self.param is not None:
            self.axes.set_title(self.param_name + f' = {self.param[frame]:.0f}')
        width, height = intensity.shape
        self.image.set_extent((0, self.pxsize*width, 0, self.pxsize*height))
        return self.image,

    def save(self):
        if self.anim:
            dialog = QFileDialog()
            dialog.setNameFilters(("GIF Files (*.gif)", "TIFF file (*.tif)"))
            dialog.setFileMode(QFileDialog.FileMode.AnyFile)
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dialog.setDirectory(self.data_directory)
            if dialog.exec():
                filepath = dialog.selectedFiles()[0]
                selected_filter = dialog.selectedNameFilter()
                # Determine selected format from the filter
                if "tif" in selected_filter:
                    format_extension = ".tif"
                elif "gif" in selected_filter:
                    format_extension = ".gif"
                if not filepath.lower().endswith(format_extension):
                    filepath += format_extension

                if "tif" in selected_filter:
                    intensities = np.array([intensity[self.mode] for intensity in self.intensities])
                    
                    minimum = np.min(intensities)
                    maximum = np.max(intensities)
                    intensities /= max(minimum, maximum)
                    tiff.imwrite(filepath, pc.float_to_mono(intensities))
                elif "gif" in selected_filter:
                    self.anim.save(filepath, fps=self.fps)
            self.data_directory = dialog.directory()
        elif self.image:
            dialog = QFileDialog(self)
            dialog.setNameFilter("Image Files(*.png *.jpg *.bmp)")
            dialog.setFileMode(QFileDialog.FileMode.AnyFile)
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dialog.setDirectory(self.data_directory)
            if dialog.exec():
                filepath = dialog.selectedFiles()[0]
                self.figure.savefig(filepath)
            self.data_directory = dialog.directory()

    def update_animation(self, intensities: List[np.ndarray] = None, pxsizes: List[float] = None, param_name: str = None, param: np.ndarray = None):
        if self.anim is not None:
            self.anim.pause()
        if intensities is not None:
            self.intensities = intensities
        if pxsizes is not None:
            self.pxsizes = pxsizes
        if param_name is not None:
            self.param_name = param_name
        if param is not None:
            self.param = param

        intensities = [intensity[self.mode] for intensity in self.intensities]
        minimum = np.min(intensities)
        maximum = np.max(intensities)
        self.image.set_clim(minimum, maximum)
        self.image.set_clim(minimum, maximum)
        self.anim = FuncAnimation(self.figure, self.animate, frames=len(intensities), interval=int(1000/self.fps), blit=False)
        self.draw_idle()

    def set_fps(self, fps):
        self.fps = fps
        if self.anim is not None:
            self.update_animation()
    
    def set_cmap(self, cmap):
        self.cmap = cmap
        if self.image is not None:
            self.image.set_cmap(cmap)
            if self.anim is None:
                self.update_image()
    
    def set_mode(self, mode):
        self.mode = mode
        if self.anim is not None:
            self.update_animation()
        elif self.image is not None:
            self.update_image()


