from PySide6.QtWidgets import QWidget, QLabel, QFormLayout, QDoubleSpinBox, QCheckBox, QSpinBox, QGroupBox, QTabWidget, QHBoxLayout, QVBoxLayout, QComboBox, QPushButton, QGridLayout, QDockWidget
from PySide6.QtCore import Signal, Qt, QFileInfo, QTimer

import numpy as np
import model

from functools import partial

# TO DO: Parameter overhaul

class ParameterWindow(QDockWidget):
    '''Window where you set the parameters.'''

    update_psf = Signal(dict)
    sweep_params = Signal(dict)
    fps_changed = Signal(int)

    params_info = {
            'magnification': {'minimum': 10, 'maximum': 200, 'singleStep': 10, 'suffix': 'x'},
            'roi_size': {'minimum': 0.2, 'maximum': 10, 'singleStep': 0.4, 'suffix': ' um', 'decimals': 1},
            'pxsize': {'minimum': 1, 'maximum': 10, 'singleStep': 0.1, 'suffix': ' um', 'decimals': 2},
            'wavelen': {'minimum': 300, 'maximum': 900, 'singleStep': 20, 'decimals': 1, 'suffix': ' nm'},
            'azimuth': {'minimum': 0, 'maximum': 360, 'singleStep': 10, 'suffix': '°'},
            'inclination': {'minimum': 0, 'maximum': 90, 'singleStep': 10, 'suffix': '°'},
            'RI': {'minimum': 1, 'maximum': 10, 'singleStep': 0.01, 'decimals': 4},
            'n_medium': {'minimum': 1, 'maximum': 10, 'singleStep': 0.01, 'decimals': 4, 'suffix': ''},
            'thickness': {'minimum': 1, 'maximum': 1000, 'singleStep':1, 'decimals': 0, 'suffix': ' um'},
            'z_p': {'minimum': 0, 'maximum': 10, 'singleStep': 0.01, 'decimals': 2, 'suffix': ' um'},
            'defocus': {'minimum': -5, 'maximum': 5, 'singleStep': 0.01, 'decimals': 2, 'suffix': ' um'},
            'xy_position': {'minimum': -2, 'maximum': 2, 'singleStep': 0.1, 'decimals': 2, 'suffix': ' um'},
            'diameter' : {'minimum': 0.1, 'maximum': 1000, 'singleStep': 10, 'decimals': 1, 'suffix': ' nm'},
            'r_resolution': {'minimum': 10, 'maximum': 100, 'singleStep': 10},
            'efficiency': {'minimum': 0.1, 'maximum': 10, 'singleStep': 0.1, 'decimals': 1},
            'aspect_ratio': {'minimum': 1, 'maximum': 10, 'singleStep': 0.1, 'decimals': 2},
        }

    def __init__(self, name, parent=None):
        super().__init__(name, parent)
        # should probably have associated units
        self.params = {}
        self.tabwidget = QTabWidget(self)
        self.setWidget(self.tabwidget)
        

        self.efficiency = QDoubleSpinBox(**self.params_info['efficiency'])
        self.efficiency.setValue(model.defaults['efficiency'])
        self.efficiency.valueChanged.connect(partial(self.changed_value, 'efficiency'))

        self.aspect_ratio = QDoubleSpinBox(**self.params_info['aspect_ratio'])
        self.aspect_ratio.setValue(model.defaults['aspect_ratio'])
        self.aspect_ratio.valueChanged.connect(partial(self.changed_value, 'aspect_ratio'))
        
        # All parameters
        self.magnification = QSpinBox(**self.params_info['magnification'])
        self.magnification.setValue(model.defaults['magnification'])
        self.magnification.valueChanged.connect(partial(self.changed_value, 'magnification'))

        self.roi_size = QDoubleSpinBox(**self.params_info['roi_size'])
        self.roi_size.setValue(model.defaults['roi_size'])
        self.roi_size.valueChanged.connect(partial(self.changed_value, 'roi_size'))

        self.pxsize = QDoubleSpinBox(**self.params_info['roi_size'])
        self.pxsize.setValue(model.defaults['pxsize'])
        self.pxsize.valueChanged.connect(partial(self.changed_value, 'pxsize'))

        self.wavelen = QDoubleSpinBox(**self.params_info['wavelen'])
        self.wavelen.setValue(model.defaults['wavelen'])
        self.wavelen.valueChanged.connect(partial(self.changed_value, 'wavelen'))

        # Angles
        self.azimuth = QSpinBox(**self.params_info['azimuth'])
        self.azimuth.setValue(model.defaults['azimuth'])
        self.azimuth.valueChanged.connect(partial(self.changed_value, 'azimuth'))
    
        self.inclination = QSpinBox(**self.params_info['inclination'])
        self.inclination.setValue(model.defaults['inclination'])
        self.inclination.valueChanged.connect(partial(self.changed_value, 'inclination'))

        self.polarization_angle = QSpinBox(**self.params_info['azimuth'])
        self.polarization_angle.setValue(model.defaults['polarization_angle'])
        self.polarization_angle.valueChanged.connect(partial(self.changed_value, 'polarization_angle'))

        # Model

        self.resolution = QSpinBox(**self.params_info['r_resolution'])
        self.resolution.setValue(model.defaults['r_resolution'])
        self.resolution.valueChanged.connect(partial(self.changed_value, 'r_resolution'))

        self.polarized = QCheckBox()
        self.polarized.setChecked(model.defaults['polarized'])
        self.polarized.stateChanged.connect(partial(self.changed_value, 'polarized'))
        self.polarized.stateChanged.connect(self.update_controls)

        self.dipole = QCheckBox()
        self.dipole.setChecked(model.defaults['dipole'])
        self.dipole.stateChanged.connect(partial(self.changed_value, 'dipole'))
        self.dipole.stateChanged.connect(self.update_controls)

        self.aberrations = QCheckBox()
        self.aberrations.setChecked(model.defaults['aberrations'])
        self.aberrations.stateChanged.connect(partial(self.changed_value, 'aberrations'))
        self.aberrations.stateChanged.connect(self.update_controls)

        self.multipolar_toggle = QCheckBox()
        self.multipolar_toggle.setChecked(model.defaults['multipolar'])
        self.multipolar_toggle.stateChanged.connect(partial(self.changed_value, 'multipolar'))



        # Layers

        self.n_oil = QDoubleSpinBox(**self.params_info['RI'])
        self.n_oil.setValue(model.n_oil)
        self.n_oil.valueChanged.connect(partial(self.changed_value, 'n_oil'))

        self.n_oil0 = QDoubleSpinBox(**self.params_info['RI'])
        self.n_oil0.setValue(model.n_oil)
        self.n_oil0.valueChanged.connect(partial(self.changed_value, 'n_oil0'))

        self.n_glass = QDoubleSpinBox(**self.params_info['RI'])
        self.n_glass.setValue(model.n_glass)
        self.n_glass.valueChanged.connect(partial(self.changed_value, 'n_glass'))

        self.n_glass0 = QDoubleSpinBox(**self.params_info['RI'])
        self.n_glass0.setValue(model.n_glass)
        self.n_glass0.valueChanged.connect(partial(self.changed_value, 'n_glass0'))

        self.t_oil0 = QDoubleSpinBox(**self.params_info['thickness'])
        self.t_oil0.setValue(model.defaults['t_oil'])
        self.t_oil0.valueChanged.connect(partial(self.changed_value, 't_oil0'))

        self.t_glass = QDoubleSpinBox(**self.params_info['thickness'])
        self.t_glass.setValue(model.defaults['t_glass'])
        self.t_glass.valueChanged.connect(partial(self.changed_value, 't_glass'))

        self.t_glass0 = QDoubleSpinBox(**self.params_info['thickness'])
        self.t_glass0.setValue(model.defaults['t_glass'])
        self.t_glass0.valueChanged.connect(partial(self.changed_value, 't_glass0'))

        self.x0 = QDoubleSpinBox(**self.params_info['xy_position'])
        self.x0.setValue(model.defaults['x0'])
        self.x0.valueChanged.connect(partial(self.changed_value, 'x0'))

        self.y0 = QDoubleSpinBox(**self.params_info['xy_position'])
        self.y0.setValue(model.defaults['y0'])
        self.y0.valueChanged.connect(partial(self.changed_value, 'y0'))

        self.z_p = QDoubleSpinBox(**self.params_info['z_p'])
        self.z_p.setValue(model.defaults['z_p'])
        self.z_p.valueChanged.connect(partial(self.changed_value, 'z_p'))

        self.defocus = QDoubleSpinBox(**self.params_info['defocus'])
        self.defocus.setValue(model.defaults['defocus'])
        self.defocus.valueChanged.connect(partial(self.changed_value, 'defocus'))
        
        self.n_scat = QComboBox()
        self.n_scat.addItems(('gold', 'polystyrene', 'custom'))
        self.n_scat.setCurrentText(model.defaults['scat_mat'])
        self.n_scat.currentTextChanged.connect(partial(self.changed_value, 'scat_mat'))

        self.n_custom = QDoubleSpinBox(**self.params_info['RI'])
        self.n_custom.setValue(model.defaults['n_custom'])
        self.n_custom.valueChanged.connect(partial(self.changed_value, 'n_custom'))

        self.n_medium = QDoubleSpinBox(**self.params_info['RI'])
        self.n_medium.setValue(model.defaults['n_medium'])
        self.n_medium.valueChanged.connect(partial(self.changed_value, 'n_medium'))

        self.diameter = QDoubleSpinBox(**self.params_info['diameter'])
        self.diameter.setValue(model.defaults['diameter'])
        self.diameter.valueChanged.connect(partial(self.changed_value, 'diameter'))

        # Animation
        
        

        self.start = QDoubleSpinBox(minimum=-10, maximum=1000)
        self.start.setValue(500)

        self.stop = QDoubleSpinBox(minimum=-10, maximum=1000)
        self.stop.setValue(600)

        self.num = QSpinBox(minimum=1, maximum=200, value=20, singleStep=10)

        self.misc = QComboBox()
        sweepable_params = ('defocus', 'z_p', 'wavelen', 'n_medium', 'diameter')
        self.misc.addItems(sweepable_params)
        self.misc.currentTextChanged.connect(self.change_anim_bounds)
        self.misc.setCurrentText('wavelen')

        self.fps = QSpinBox(minimum=1, maximum=200, value=10, singleStep=10)
        self.fps.valueChanged.connect(self.fps_changed.emit)

        self.live_mode = QCheckBox('Live Mode')

        self.start_sweep = QPushButton('Start sweep')
        self.start_sweep.clicked.connect(self.sweep)
        
        

        # Group into groups and tabs

        # Setup group
        self.setup_group = QGroupBox('Setup')
        setup_layout = QFormLayout()
        setup_layout.addRow('Magnification', self.magnification)
        setup_layout.addRow('ROI Size', self.roi_size)
        setup_layout.addRow('Pixel Size', self.pxsize)
        self.setup_group.setLayout(setup_layout)

        # Orientation group
        self.orientation_group = QGroupBox('Orientation and Polarization')
        orientation_layout = QFormLayout()
        orientation_layout.addWidget(QLabel('Scatterer'))
        orientation_layout.addRow('Multipolar', self.multipolar_toggle)
        orientation_layout.addRow('Azimuth', self.azimuth)
        orientation_layout.addRow('Inclination', self.inclination)
        orientation_layout.addRow('Dipole', self.dipole)
        orientation_layout.addRow('Aspect ratio', self.aspect_ratio)
        orientation_layout.addWidget(QLabel('Excitation'))
        orientation_layout.addRow('Polarized', self.polarized)
        orientation_layout.addRow('Polarization', self.polarization_angle)
        self.orientation_group.setLayout(orientation_layout)

        # Model group
        self.model_group = QGroupBox('Model')
        model_layout = QFormLayout()
        model_layout.addRow('Radial Resolution', self.resolution)
        model_layout.addRow('Efficiency', self.efficiency)
        self.model_group.setLayout(model_layout)

        # Aberrations group
        self.abertations_group = QGroupBox('Abberations')
        abertations_group_layout = QVBoxLayout()
        check_layout = QFormLayout()
        check_layout.addRow('Abberations', self.aberrations)
        abertations_layout = QGridLayout()
        abertations_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        abertations_layout.addWidget(QLabel('Parameter'),0,0)
        abertations_layout.addWidget(QLabel('n_oil'),1,0)
        abertations_layout.addWidget(QLabel('n_glass'),2,0)
        abertations_layout.addWidget(QLabel('t_oil'),3,0)
        abertations_layout.addWidget(QLabel('t_glass'),4,0)

        abertations_layout.addWidget(QLabel('Design'),0,1)
        abertations_layout.addWidget(self.n_oil0, 1, 1)
        abertations_layout.addWidget(self.n_glass0, 2, 1)
        abertations_layout.addWidget(self.t_oil0, 3, 1)
        abertations_layout.addWidget(self.t_glass0, 4, 1)

        abertations_layout.addWidget(QLabel('Real'),0,2)
        abertations_layout.addWidget(self.n_oil, 1, 2)
        abertations_layout.addWidget(self.n_glass, 2, 2)
        abertations_layout.addWidget(self.t_glass, 4, 2)

        abertations_group_layout.addLayout(check_layout)
        abertations_group_layout.addLayout(abertations_layout)
        self.abertations_group.setLayout(abertations_group_layout)

        # Particle Group
        self.particle_group = QGroupBox('Scattering')
        particle_layout = QFormLayout()
        particle_layout.addRow('Wavelength', self.wavelen)
        particle_layout.addRow('Defocus', self.defocus)

        particle_layout.addRow('X Position', self.x0)
        particle_layout.addRow('Y Position', self.y0)
        particle_layout.addRow('Z Position', self.z_p)
        particle_layout.addRow('n_scat', self.n_scat)
        particle_layout.addRow('custom RI', self.n_custom)
        particle_layout.addRow('n_medium', self.n_medium)
        particle_layout.addRow('diameter', self.diameter)
        self.particle_group.setLayout(particle_layout)


        # Scattering tab
        self.scatterer_tab = QWidget(self)
        scatterertab_layout = QVBoxLayout()
        scatterertab_layout.addWidget(self.particle_group)
        scatterertab_layout.addWidget(self.orientation_group)
        scatterertab_layout.addWidget(self.model_group)
        scatterertab_layout.addStretch(1)
        self.scatterer_tab.setLayout(scatterertab_layout)

        # Setup tab
        self.setup_tab = QWidget(self)
        setuptab_layout = QVBoxLayout()
        setuptab_layout.addWidget(self.setup_group)
        setuptab_layout.addWidget(self.model_group)
        setuptab_layout.addStretch(1)
        self.setup_tab.setLayout(setuptab_layout)

        # Sweep animation tab
        self.anim_tab = QWidget(self)
        animtab_layout = QFormLayout()
        animtab_layout.addRow('Parameter', self.misc)
        animtab_layout.addRow('Start', self.start)
        animtab_layout.addRow('Stop', self.stop)
        animtab_layout.addRow('Number', self.num)
        animtab_layout.addRow('fps', self.fps)
        animtab_layout.addRow('Live mode', self.live_mode)
        animtab_layout.addWidget(self.start_sweep)
        self.anim_tab.setLayout(animtab_layout)


        self.tabwidget.addTab(self.scatterer_tab, 'Scattering')
        self.tabwidget.addTab(self.setup_tab, 'Setup')
        self.tabwidget.addTab(self.anim_tab, 'Sweep')

        self.update_controls()

    
    def changed_value(self, name, value):
        self.params[name] = value
        self.update_controls()
        if self.live_mode.isChecked():
            self.sweep()
        else:
            self.update_psf.emit(self.params)

    def update_controls(self):
        a = not self.multipolar_toggle.isChecked()
        self.azimuth.setEnabled(a)
        self.inclination.setEnabled(a)
        self.aspect_ratio.setEnabled(a)
        self.dipole.setEnabled(a)

        self.polarization_angle.setEnabled(self.polarized.isChecked())
        self.n_custom.setEnabled(self.n_scat.currentText() == 'custom')

        checked = self.aberrations.isChecked()
        for param in [self.n_glass, self.n_glass0, self.n_oil, self.n_oil0, self.t_oil0, self.t_glass, self.t_glass0]:
            param.setEnabled(checked)
    
    def change_anim_bounds(self, param):
        info = self.params_info[param]
        for k, v in info.items():
            self.start.setProperty(k, v)
            self.start.setValue(model.defaults[param])
            self.stop.setProperty(k, v)
            self.stop.setValue(model.defaults[param]+(self.num.value()*info['singleStep']))
    
    def sweep(self):
        param = np.linspace(self.start.value(), self.stop.value(), self.num.value())
        param_name = self.misc.currentText()
        params = self.params.copy()
        params[param_name] = param
        self.sweep_params.emit(params)
