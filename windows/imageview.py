from PySide6.QtCore import QRect, QMargins, Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QShortcut, QKeySequence, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QWidget

import numpy as np
import time

AnchorCenter = QGraphicsView.ViewportAnchor.AnchorViewCenter
AnchorMouse = QGraphicsView.ViewportAnchor.AnchorUnderMouse
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.cm as cm

class ImageView(QGraphicsView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        #self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        self.display = QGraphicsPixmapItem()
        #self.display.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self.display)

        self.setMinimumSize(640, 480)
        self.setScene(self._scene)
        

    def update_image(self, frame):
        t = time.time()
        # Normalization
        frame = frame - np.min(frame)
        if np.max(frame) > 0:
            frame = frame/np.max(frame)
        frame = (cm.viridis(frame)*65535).astype(np.uint16)
        width, height, channels = frame.shape
        #print(frame.dtype)
        self.display.setPixmap(QPixmap.fromImage(QImage(frame.data, width, height, QImage.Format_RGBA64)).scaledToWidth(400))
        
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        size = rect.size()
        window_width = size.width()
        window_height = size.height()
        scale = min(window_width, window_height)
        
        self.scale(scale / self._scene.width(), scale / self._scene.height())