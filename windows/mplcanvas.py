from PySide6.QtCore import QRect, QMargins, Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QShortcut, QKeySequence, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QWidget

import numpy as np
import time

AnchorCenter = QGraphicsView.ViewportAnchor.AnchorViewCenter
AnchorMouse = QGraphicsView.ViewportAnchor.AnchorUnderMouse
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, pxsize=3.45):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_axis_off()
        
        fig.tight_layout(pad=0)
        super().__init__(fig)
        self.image: AxesImage = None

    def update_image(self, frame: np.ndarray, pxsize=3.45):
        width, height = frame.shape
        if self.image is None:
            self.image = self.axes.imshow(frame, cmap='viridis', extent=(0, pxsize*width, 0, pxsize*height))
            self.colorbar = self.figure.colorbar(self.image)
            scalebar = AnchoredSizeBar(self.axes.transData, 0.2, '200 nm', 'lower center', pad=0.1, frameon=False, color='white', sep=5)
            self.axes.add_artist(scalebar)
        else:
            self.image.set_data(frame)
            
            # Double bc it doesnt set them simultaneously
            self.image.set_clim(np.min(frame), np.max(frame))
            self.image.set_clim(np.min(frame), np.max(frame))
            self.image.set_extent((0, pxsize*width, 0, pxsize*height))
            self.draw_idle()