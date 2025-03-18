from PySide6.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox, QCheckBox, QSpinBox, QGroupBox, QTabWidget, QHBoxLayout, QVBoxLayout, QComboBox
from PySide6.QtCore import Signal, Qt, QFileInfo

from dataclasses import asdict
from model import DesignParams
import json
from functools import partial

class ParameterWindow(QTabWidget):
    """Window where you set the parameters."""

    changed_parameters = Signal()
    
    def __init__(self, parent):
        super().__init__()
        self.setWindowTitle("Parameters")
        self.resize(600, 400)
        self.params = DesignParams()
        self.rresolution = 40

        # Load from file
        if QFileInfo.exists(parent.parameters_file):
            # Load info from parameters file
            try:
                with open(parent.parameters_file, 'r') as file:
                    self.params = DesignParams(**json.load(file))
            except:
                print("failed loading params")

        
        # All parameters
        self.magnification = QSpinBox(value=self.params.magnification, minimum=10, maximum=200, singleStep=10, suffix="x")
        self.magnification.setValue(self.params.magnification)
        self.magnification.valueChanged.connect(partial(self.set_parameter, self.params, "magnification"))

        self.roi_size = QDoubleSpinBox(minimum=0.2, maximum=10, singleStep=0.4, suffix=" micron", decimals=1)
        self.roi_size.setValue(self.params.roi_size)
        self.roi_size.valueChanged.connect(partial(self.set_parameter, self.params, "roi_size"))

        self.pxsize = QDoubleSpinBox(minimum=1, maximum=10, singleStep=0.1, suffix=" micron", decimals=2)
        self.pxsize.setValue(self.params.pxsize)
        self.pxsize.valueChanged.connect(partial(self.set_parameter, self.params, "pxsize"))

        self.wavelen = QSpinBox(minimum=300, maximum=900, singleStep=50, suffix=" nm")
        self.wavelen.setValue(self.params.wavelen)
        self.wavelen.valueChanged.connect(partial(self.set_parameter, self.params, "wavelen"))

        self.azimuth = QSpinBox(minimum=0, maximum=360, singleStep=10, suffix="°")
        self.azimuth.setValue(self.params.azimuth)
        self.azimuth.valueChanged.connect(partial(self.set_parameter, self.params, "azimuth"))
    
        self.inclination = QSpinBox(minimum=0, maximum=90, singleStep=10, suffix="°")
        self.inclination.setValue(self.params.inclination)
        self.inclination.valueChanged.connect(partial(self.set_parameter, self.params, "inclination"))
        
        self.resolution = QSpinBox(minimum=10, maximum=200, singleStep=10)
        self.resolution.setValue(self.rresolution)
        self.resolution.valueChanged.connect(partial(self.set_parameter, self, "rresolution"))
        
        self.unpolarized = QCheckBox()
        self.unpolarized.setChecked(self.params.unpolarized)
        self.unpolarized.stateChanged.connect(partial(self.set_parameter, self.params, "unpolarized"))

        self.misc_value = QDoubleSpinBox(minimum=0, maximum=1000, singleStep=1)
        self.misc = QComboBox()
        self.misc.addItems(asdict(self.params).keys())
        self.misc.currentTextChanged.connect(lambda name: self.misc_value.setValue(float(getattr(self.params, name))))
        self.misc_value.setValue(self.params.n_oil)
        

        # Group into groups and tabs

        self.setup_group = QGroupBox("Setup")
        setup_layout = QFormLayout()
        setup_layout.addRow("Magnification", self.magnification)
        setup_layout.addRow("ROI Size", self.roi_size)
        setup_layout.addRow("Pixel Size", self.pxsize)
        setup_layout.addRow("Wavelength", self.wavelen)
        self.setup_group.setLayout(setup_layout)

        self.particle_group = QGroupBox("Particle")
        particle_layout = QFormLayout()
        particle_layout.addRow("Azimuth", self.azimuth)
        particle_layout.addRow("Inclination", self.inclination)
        particle_layout.addRow("Unpolarized", self.unpolarized)
        self.particle_group.setLayout(particle_layout)

        self.model_group = QGroupBox("Model")
        model_layout = QFormLayout()
        model_layout.addRow("Radial Resolution", self.resolution)
        self.model_group.setLayout(model_layout)

        misc_layout = QHBoxLayout()
        misc_layout.addWidget(self.misc)
        misc_layout.addWidget(self.misc_value)


        # Setup tab
        self.setup_tab = QWidget(self)
        self.addTab(self.setup_tab, "Setup")
        setuptab_layout = QHBoxLayout()
        # Column one
        setuptab_layout.addWidget(self.setup_group)

        # Column two
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.particle_group)
        vlayout.addWidget(self.model_group)
        vlayout.addLayout(misc_layout)
        setuptab_layout.addLayout(vlayout)

        self.setup_tab.setLayout(setuptab_layout)
        
    
    def set_parameter(self, obj, name, value):
        setattr(obj, name, value)
        self.changed_parameters.emit()