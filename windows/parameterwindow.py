from PySide6.QtWidgets import QWidget, QLabel, QFormLayout, QDoubleSpinBox, QCheckBox, QSpinBox, QGroupBox, QTabWidget, QHBoxLayout, QVBoxLayout, QComboBox, QPushButton, QGridLayout, QDockWidget
from PySide6.QtCore import Signal, Qt, QFileInfo, QTimer


from dataclasses import asdict
from model import DesignParams
import json
from functools import partial

from model import *
import numpy as np


# TO DO: Parameter overhaul

class ParameterWindow(QDockWidget):
    """Window where you set the parameters."""

    update_psf = Signal(DesignParams)
    sweep_params = Signal(DesignParams, str, np.ndarray)
    fps_changed = Signal(int)


    def __init__(self, name, parent=None):
        super().__init__(name, parent)
        # should probably have associated units
        self.params = DesignParams()
        self.tabwidget = QTabWidget(self)
        self.setWidget(self.tabwidget)

        # Should probably go in params?
        self.rresolution: int = 40
        self.scatterer: str = "gold"

        # Load from file
        # if QFileInfo.exists(parent.parameters_file):
        #     # Load info from parameters file
        #     try:
        #         with open(parent.parameters_file, 'r') as file:
        #             self.params = DesignParams(**json.load(file))
        #     except:
        #         print("failed loading params")

        
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
        self.wavelen.valueChanged.connect(self.set_wavelen)

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
        self.unpolarized.stateChanged.connect(lambda: self.set_parameter(self.params, "unpolarized", self.unpolarized.isChecked()))

        # Layers

        self.n_oil = QDoubleSpinBox(minimum=1, maximum=10, singleStep=0.1, decimals=4)
        self.n_oil.setValue(self.params.n_oil)
        self.n_oil.valueChanged.connect(partial(self.set_parameter, self.params, "n_oil"))

        self.n_oil0 = QDoubleSpinBox(minimum=1, maximum=10, singleStep=0.1, decimals=4)
        self.n_oil0.setValue(self.params.n_oil0)
        self.n_oil0.valueChanged.connect(partial(self.set_parameter, self.params, "n_oil0"))

        self.n_glass = QDoubleSpinBox(minimum=1, maximum=10, singleStep=0.1, decimals=4)
        self.n_glass.setValue(self.params.n_glass)
        self.n_glass.valueChanged.connect(partial(self.set_parameter, self.params, "n_glass"))

        self.n_glass0 = QDoubleSpinBox(minimum=1, maximum=10, singleStep=0.1, decimals=4)
        self.n_glass0.setValue(self.params.n_glass0)
        self.n_glass0.valueChanged.connect(partial(self.set_parameter, self.params, "n_glass0"))

        self.t_oil = QDoubleSpinBox(minimum=1, maximum=1000, singleStep=1, suffix=" micron", decimals=0)
        self.t_oil.setValue(self.params.t_oil)
        self.t_oil.valueChanged.connect(partial(self.set_parameter, self.params, "t_oil"))

        self.t_oil0 = QDoubleSpinBox(minimum=1, maximum=1000, singleStep=1, suffix=" micron", decimals=0)
        self.t_oil0.setValue(self.params.t_oil0)
        self.t_oil0.valueChanged.connect(partial(self.set_parameter, self.params, "t_oil0"))

        self.t_glass = QDoubleSpinBox(minimum=1, maximum=1000, singleStep=1, suffix=" micron", decimals=0)
        self.t_glass.setValue(self.params.t_glass)
        self.t_glass.valueChanged.connect(partial(self.set_parameter, self.params, "t_glass"))

        self.t_glass0 = QDoubleSpinBox(minimum=1, maximum=1000, singleStep=1, suffix=" micron", decimals=0)
        self.t_glass0.setValue(self.params.t_glass0)
        self.t_glass0.valueChanged.connect(partial(self.set_parameter, self.params, "t_glass0"))

        self.z_particle = QDoubleSpinBox(minimum=0, maximum=10, singleStep=0.01, decimals=2, suffix=" micron")
        self.z_particle.setValue(self.params.z_p)
        self.z_particle.valueChanged.connect(partial(self.set_parameter, self.params, "z_p"))

        self.z_focus = QDoubleSpinBox(minimum=-5, maximum=15, singleStep=0.01, decimals=2, suffix=" micron")
        self.z_focus.setValue(self.params.z_focus)
        self.z_focus.valueChanged.connect(partial(self.set_parameter, self.params, "z_focus"))
        
        self.n_scat = QComboBox()
        self.n_scat.addItems(("gold", "polystyrene"))
        self.n_scat.currentTextChanged.connect(self.set_n)
        

        self.n_medium = QDoubleSpinBox(minimum=1, maximum=10, singleStep=0.1, decimals=4)
        self.n_medium.setValue(self.params.n_medium)
        self.n_medium.valueChanged.connect(partial(self.set_parameter, self.params, "n_medium"))

        
        self.misc = QComboBox()
        sweepable_params = [k for k, v in asdict(self.params).items() if type(v) in (int, float)]
        self.misc.addItems(sweepable_params)

        self.start = QDoubleSpinBox(minimum=0, maximum=1000)
        self.stop = QDoubleSpinBox(minimum=0, maximum=1000)
        self.start.valueChanged.connect(lambda value: self.stop.setValue(max(value, self.stop.value())))
        self.stop.valueChanged.connect(lambda value: self.start.setValue(min(value, self.start.value())))
        self.num = QSpinBox(minimum=1, value=10)
        self.fps = QSpinBox(minimum=1, value=10)
        self.fps.valueChanged.connect(self.fps_changed.emit)
        self.start_sweep = QPushButton("Start sweep")
        self.start_sweep.clicked.connect(self.sweep)
        

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

        self.layers_group = QGroupBox("Layers")
        layers_layout = QGridLayout()
        layers_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layers_layout.addWidget(QLabel("Parameter"),0,0)
        layers_layout.addWidget(QLabel("n_oil"),1,0)
        layers_layout.addWidget(QLabel("n_glass"),2,0)
        layers_layout.addWidget(QLabel("t_oil"),3,0)
        layers_layout.addWidget(QLabel("t_glass"),4,0)

        layers_layout.addWidget(QLabel("Design"),0,1)
        layers_layout.addWidget(self.n_oil0, 1, 1)
        layers_layout.addWidget(self.n_glass0, 2, 1)
        layers_layout.addWidget(self.t_oil0, 3, 1)
        layers_layout.addWidget(self.t_glass0, 4, 1)

        layers_layout.addWidget(QLabel("Real"),0,2)
        layers_layout.addWidget(self.n_oil, 1, 2)
        layers_layout.addWidget(self.n_glass, 2, 2)
        layers_layout.addWidget(self.t_oil, 3, 2)
        layers_layout.addWidget(self.t_glass, 4, 2)

        self.layers_group.setLayout(layers_layout)

        self.variable_group = QGroupBox("Variables")
        variable_layout = QFormLayout()
        variable_layout.addRow("Particle Position", self.z_particle)
        variable_layout.addRow("Focal Position", self.z_focus)
        variable_layout.addRow("n_medium", self.n_medium)
        variable_layout.addRow("n_scatterer", self.n_scat)
        self.variable_group.setLayout(variable_layout)

        animtab_layout = QFormLayout()
        animtab_layout.addRow("Parameter", self.misc)
        animtab_layout.addRow("Start", self.start)
        animtab_layout.addRow("Stop", self.stop)
        animtab_layout.addRow("Number", self.num)
        animtab_layout.addRow("fps", self.fps)
        animtab_layout.addWidget(self.start_sweep)


        # Setup tab
        self.setup_tab = QWidget(self)
        self.tabwidget.addTab(self.setup_tab, "Setup")
        setuptab_layout = QVBoxLayout()
        setuptab_layout.addWidget(self.setup_group)
        setuptab_layout.addWidget(self.particle_group)
        setuptab_layout.addWidget(self.model_group)
        setuptab_layout.addStretch(1)

        self.setup_tab.setLayout(setuptab_layout)

        self.aberration_tab = QWidget(self)
        self.tabwidget.addTab(self.aberration_tab, "Aberration")
        aberrationtab_layout = QVBoxLayout()
        aberrationtab_layout.addWidget(self.layers_group)
        aberrationtab_layout.addWidget(self.variable_group)
        aberrationtab_layout.addStretch(1)
        self.aberration_tab.setLayout(aberrationtab_layout)

        # Sweep animation tab
        self.anim_tab = QWidget(self)
        self.tabwidget.addTab(self.anim_tab, "Animation")

        self.anim_tab.setLayout(animtab_layout)
    
    def set_n(self, text):
        self.scatterer = text
        self.update_n()
        self.update_psf.emit(self.params)
    
    def set_wavelen(self, value):
        self.params.wavelen = value
        self.update_n()
        self.update_psf.emit(self.params)

    def update_n(self):
        if self.scatterer == "gold":
            self.params.n_scat = n_gold(self.params.wavelen)
        elif self.scatterer == "polystyrene":
            self.params.n_scat = n_ps
        print(self.params.n_scat)
    
    def set_parameter(self, obj, name, value):
        setattr(obj, name, value)
        self.update_psf.emit(self.params)
    
    def sweep(self):
        param = np.linspace(self.start.value(), self.stop.value(), self.num.value())
        param_name = self.misc.currentText()
        self.sweep_params.emit(self.params, param_name, param)