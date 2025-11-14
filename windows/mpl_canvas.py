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

from collections.abc import Sequence
from numpy.typing import NDArray

#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, pxsize=3.45):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_axis_off()
        fig.tight_layout(pad=2)
        super().__init__(fig)


        self.fps = 10
        self.cmap = 'viridis'
        self.mode = 'signal'
        
        self.axes_image: AxesImage = self.axes.imshow(np.zeros((35, 35)), cmap=self.cmap)
        self.anim: FuncAnimation | None = None
        self.colorbar = self.figure.colorbar(self.axes_image)
        self.scalebar = AnchoredSizeBar(self.axes.transData, 0.4, '400 nm', 'lower center', pad=0.1, frameon=False, color='white', sep=5)
        self.axes.add_artist(self.scalebar)
        

        self.data_directory = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)

    def update_image(self, intensities: NDArray | None = None, pxsize: float | None = None):
        if intensities is not None:
            self.intensities = intensities
        frame = self.intensities[self.mode]
        
        # Remove anim
        if self.anim is not None:
            self.anim.pause()
            self.anim = None
        
        self.axes_image.set_data(frame)

        # Update pxsize
        if pxsize is not None:
            rows, cols = frame.shape
            self.pxsize = pxsize
            self.axes_image.set_extent((0, self.pxsize*rows, 0, self.pxsize*cols))

        # Limits (double bc it doesnt set them simultaneously)
        self.axes_image.set_clim(np.min(frame), np.max(frame))
        self.axes_image.set_clim(np.min(frame), np.max(frame))

        self.draw_idle()

    

    def animate(self, frame: int):
        intensity = self.intensities[self.mode][frame]
        self.axes_image.set_data(intensity)

        
        if self.param is not None and self.param_name is not None:
            self.axes.set_title(self.param_name + f' = {self.param[frame]:.2f}')
        
        rows, cols = intensity.shape
        self.axes_image.set_extent((0, self.pxsize*rows, 0, self.pxsize*cols))
        return self.axes_image,

    def save(self):
        if self.anim:
            # Save animation as multipage tif or gif
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
                    self.anim.pause()
                    intensities = self.intensities[self.mode].copy()
                    
                    minimum = np.min(intensities)
                    maximum = np.max(intensities)
                    intensities /= max(minimum, maximum)
                    tiff.imwrite(filepath, pc.float_to_mono(intensities))
                    self.anim.resume()
                elif "gif" in selected_filter:
                    self.anim.save(filepath, fps=self.fps)
            self.data_directory = dialog.directory()
        else:
            # Save animation as animated gif
            dialog = QFileDialog(self)
            dialog.setNameFilter("Image Files(*.png *.jpg *.bmp)")
            dialog.setFileMode(QFileDialog.FileMode.AnyFile)
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dialog.setDirectory(self.data_directory)
            if dialog.exec():
                filepath = dialog.selectedFiles()[0]
                self.figure.savefig(filepath)
            self.data_directory = dialog.directory()
        
    
    def update_animation(self, 
                         intensities: dict[str, NDArray],
                         param_name: str | None = None,
                         param: np.ndarray | None = None):
        """Set new intensity data as animation"""
        
        if self.anim is not None:
            self.anim.pause()
        
        self.intensities = intensities
        self.param_name = param_name
        self.param = param

        video: NDArray = self.intensities[self.mode]
        minimum = np.min(video)
        maximum = np.max(video)
        self.axes_image.set_clim(minimum, maximum)
        self.axes_image.set_clim(minimum, maximum)
        self.anim = FuncAnimation(self.figure, self.animate, frames=len(video), interval=int(1000/self.fps), blit=False)
        self.draw_idle()

    
    def refresh_animation(self):
        """Refresh animation using new fps, cmap, mode"""
        video: NDArray = self.intensities[self.mode]
        minimum = np.min(video)
        maximum = np.max(video)
        self.axes_image.set_clim(minimum, maximum)
        self.axes_image.set_clim(minimum, maximum)
        self.anim = FuncAnimation(self.figure, self.animate, frames=len(video), interval=int(1000/self.fps), blit=False)
        self.draw_idle()


    def set_fps(self, fps):
        self.fps = fps
        if self.anim is not None:
            self.refresh_animation()
    
    def set_cmap(self, cmap):
        self.cmap = cmap
        self.axes_image.set_cmap(cmap)
        if self.anim is None:
            self.update_image()
    
    def set_mode(self, mode: str):
        self.mode = mode
        if self.anim is not None:
            self.refresh_animation()
            self.update_image()