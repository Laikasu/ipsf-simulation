from PySide6.QtCore import QObject, Signal
import model
import numpy as np

class MainController(QObject):

    display_update = Signal(dict, float)
    display_anim_update = Signal(dict, str, np.ndarray)
    plot_update = Signal(dict, str, np.ndarray)

    
    def update_psf(self, params: dict):
        # pxsize is necessary for scalebar
        pxsize = params.get('pxsize', model.defaults['pxsize'])
        magnification = params.get('magnification', model.defaults['magnification'])
        pxsize_obj = pxsize/magnification

        interference_contrast, scattering_contrast = model.simulate_camera(**params)
        self.intensity = {'scattering': scattering_contrast,
                          'interference': interference_contrast,
                          'signal': scattering_contrast + interference_contrast}
        self.display_update.emit(self.intensity, pxsize_obj)

    
    
    def sweep(self, params: dict):
        interference_contrast, scattering_contrast = model.simulate_camera(**params)
        self.intensity = {'scattering': scattering_contrast,
                          'interference': interference_contrast,
                          'signal': scattering_contrast + interference_contrast}

        param_name, param = [(k,v) for (k,v) in params.items() if isinstance(v, np.ndarray)][0]
        
        self.display_anim_update.emit(self.intensity, param_name, param)
        self.plot_update.emit(self.intensity, param_name, param)