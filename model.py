import numpy as np
from scipy.special import jv
from scipy.integrate import quad
from scipy.interpolate import interp1d
import os

# TO DO: other data
# TO DO: name all formulae

script_dir = os.path.dirname(os.path.abspath(__file__))
# Magnozzi data for gold
# https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Au/Magnozzi-25C.yml
# Johnsonn and Christy

gold = np.genfromtxt(os.path.join(script_dir, 'Magnozzi-25C.csv'), delimiter=',', skip_header=1).T
_gold_wavelen = gold[0]
_n_gold = gold[1] - 1j*gold[2]
n_gold = interp1d(_gold_wavelen*1000, _n_gold)

import scipy.constants as const


res_p=8.45
def drude_lorenz_gold(wavelen):
    # Constants from Improved Drude-Lorentz dielectric function for gold nanospheres
    # Anthony Centeno
    f = np.array([0.98, 0.1222, 0.2492, 2.7845, 0.1082, 7.8214])[:]
    res_0 = np.array([0, 3.5684, 4.2132, 9.1499, 2.9144, 38.9633])[:]
    damping = np.array([0.044113, 0.97329, 1.1139, 0.4637, 0.70308, 0.48978])[:]
    # Rakic et al.
    # f = np.array([0.760, 0.024, 0.010, 0.071, 0.601, 4.384])
    # res_0 = np.array([0, 0.415, 0.830, 2.969, 4.304, 13.32])
    # damping = np.array([0.053, 0.241, 0.345, 0.870, 2.494, 2.214])


    freq_eV = const.h*const.c/wavelen/10**-9/const.e
    return np.sqrt(1 + np.sum(f*res_p**2/(res_0**2-freq_eV**2-1j*freq_eV*damping)))


n_ps = 1.5537
n_air = 1
n_water = 1.333
n_glass = 1.499
n_oil = 1.518
n_glyc = 1.461

class DesignParams():
    scat_materials = ("polystyrene", "gold")
    def __init__(self,
                n_oil: float = n_oil,
                n_oil0: float = n_oil,
                n_glass: float = n_glass,
                n_glass0: float = n_glass,
                n_medium: float = n_water,
                t_oil0: float = 100, # micron
                t_oil: float = 100, # micron
                t_glass: float = 170, # micron
                t_glass0: float = 170, # micron
                diameter: float = 30, # nm
                z_p: float = 0, # micron
                z_focus: float = 0,
                wavelen: float = 525, # nm
                NA: float = 1.4,
                azimuth: float = 0, # degrees
                inclination: float = 0, # degrees
                unpolarized: bool = False,
                roi_size: float = 2, # micron
                pxsize: float = 3.45, # micron
                magnification: float = 60,
                scat_mat: str = "gold",
                x0: float = 0,
                y0: float = 0,
                gold_model: str = 'experiment'):
        self.n_oil: float = n_oil
        self.n_oil0: float = n_oil0
        self.n_glass: float = n_glass
        self.n_glass0: float = n_glass0
        self.n_medium: float = n_medium
        self.t_oil0: float = t_oil0 # micron
        self.t_oil: float = t_oil # micron
        self.t_glass: float = t_glass # micron
        self.t_glass0: float = t_glass0 # micron
        self._wavelen: float = wavelen # nm
        self._scat_mat = scat_mat
        self.diameter: float = diameter # nm
        self.z_p: float = z_p # micron
        self.z_focus: float = z_focus
        self.NA: float = NA
        self.azimuth: float = azimuth # degrees
        self.inclination: float = inclination # degrees
        self.unpolarized: bool = unpolarized
        self.roi_size: float = roi_size # micron
        self.pxsize: float = pxsize # micron
        self.magnification: float = magnification
        self.x0: float = x0
        self.y0: float = y0
        self._gold_model: str = gold_model
        self.update_n()
    
    def update_n(self):
        if self.scat_mat == "gold":
            if self._gold_model == 'experiment':
                self.n_scat = n_gold(self._wavelen)
            elif self._gold_model == 'drude-lorenz':
                self.n_scat = drude_lorenz_gold(self._wavelen)
    
    @property
    def scat_mat(self):
        return self._scat_mat
        
    @scat_mat.setter
    def scat_mat(self, value: str):
        if value == "gold":
            self._scat_mat = value
            self.update_n()
        elif value == "polystyrene":
            self.n_scat = n_ps
            self._scat_mat = value
        else:
            raise ValueError(f"Material {value} not found")
    
    @property
    def gold_model(self):
        return self._gold_model
    
    @gold_model.setter
    def gold_model(self, value: str):
        if value not in ['experiment', 'drude-lorenz']:
            raise ValueError('This model is not implemented')
        else:
            self._gold_model = value
            self.update_n()
        
        
    @property
    def wavelen(self):
        return self._wavelen
    
    @wavelen.setter
    def wavelen(self, value):
        self._wavelen = value
        self.update_n()
    
    @property
    def a(self):
        return self.diameter/2


    def to_dict(self):
        return self.__dict__
    

class Camera():
    def __init__(self, params: DesignParams):
        x0 = params.x0*10**-6
        y0 = params.y0*10**-6
        self.roi_size = params.roi_size*10**-6
        self.pxsize = params.pxsize*10**-6
        self.magnification = params.magnification
        self.pxsize_obj = self.pxsize/self.magnification
        pixels = int(self.roi_size//self.pxsize_obj)
        self.pixels = pixels + 1 if pixels%2 == 0 else pixels
        self.xs = np.linspace(-self.roi_size/2, self.roi_size/2, pixels)
        self.ys = np.linspace(-self.roi_size/2, self.roi_size/2, pixels)
        x, y = np.meshgrid(self.xs, self.ys)
        self.r = np.sqrt((x-x0)**2 + (y-y0)**2)
        self.phi = np.arctan2(y-y0, x-x0)


def opd(angle_oil, p):
    n_oil = p.n_oil
    n_oil0 = p.n_oil0
    t_oil = p.t_oil*10**-6
    t_oil0 = p.t_oil0*10**-6
    t_glass = p.t_glass*10**-6
    t_glass0 = p.t_glass0*10**-6
    n_glass = p.n_glass
    n_glass0 = p.n_glass0
    n_medium = p.n_medium
    z_p = p.z_p*10**-6
    z_focus = p.z_focus*10**-6

    #return 0
    # opd = (n_oil*np.cos(angle_oil)*(z_p - z_focus + n_oil*(-z_p/n_medium - t_glass/n_glass + t_glass0/n_glass0 + t_oil0/n_oil0)) +
    #     z_p*np.sqrt(n_medium**2 - n_oil**2 *np.sin(angle_oil)**2) + t_glass*np.sqrt(n_glass**2 - n_oil**2 *np.sin(angle_oil)**2) -
    #     t_glass0*np.sqrt(n_glass0**2 - n_oil**2 *np.sin(angle_oil)**2) - t_oil0*np.sqrt(n_oil0**2 - n_oil**2 *np.sin(angle_oil)**2))
    opd = n_oil*np.cos(angle_oil)*(z_p - z_focus)
    return opd# + n_medium*z_p + n_glass*t_glass + n_oil*t_oil - n_glass0*t_glass0 - n_oil0*t_oil0

def opd_ref(p):
    z_focus = p.z_focus*10**-6
    # The reference originates at the interface, so 
    return -n_oil*z_focus


def snells_law(n1, angle1, n2):
    """
    Return angle2 given n1*angle1 = n2*angle2
    """
    return np.arcsin(n1 / n2 * np.sin(angle1))
    

def t_p(n1, angle1, n2, angle2):
    return 2*n1*np.cos(angle1)/(n2*np.cos(angle1) + n1*np.cos(angle2))

def t_s(n1, angle1, n2, angle2):
    return 2*n1*np.cos(angle1)/(n1*np.cos(angle1) + n2*np.cos(angle2))


def B(n, angle_oil, r, p: DesignParams):
    wavelen = p.wavelen*10**-9
    n_oil = p.n_oil
    #n_scat = p.n_scat
    #n_medium = p.n_medium
    #a = p.a*10**-9

    k = -2*np.pi/wavelen
    #angle_medium = snells_law(n_oil, angle_oil, n_medium)
    #mu = np.cos(angle_medium)
    #x_mie = 2*np.pi*a*n_medium / wavelen
    #profile = mie.i_par(n_scat/n_medium, x_mie, mu)
    return np.sqrt(np.cos(angle_oil))*np.sin(angle_oil)*jv(n, k*r*n_oil*np.sin(angle_oil))*np.exp(1j*k*opd(angle_oil, p))


def Integral_0(r, p, scatter_field):
    n_medium = p.n_medium
    n_glass = p.n_glass
    n_oil = p.n_oil
    NA = p.NA
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    sfield = 1
    if scatter_field.ndim > 1:
        angles = np.linspace(0, capture_angle_medium, scatter_field.shape[0])
        sfield = interp1d(angles, scatter_field)
    else:
        sfield = lambda angle: 1

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_medium, angle_medium, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        t_s_1 = t_s(n_medium, angle_medium, n_glass, angle_glass)
        t_s_2 = t_s(n_glass, angle_glass, n_oil, angle_oil)
        return (sfield(angle_oil)*B(0, angle_oil, r, p)*
            (t_s_1*t_s_2 + t_p_1*t_p_2/n_medium*np.sqrt(n_medium**2 - n_oil**2*np.sin(angle_oil)**2)))

    return quad(integrand, 0, capture_angle_medium, complex_func=True)[0]

def Integral_1(r, p, scatter_field):
    n_medium = p.n_medium
    n_glass = p.n_glass
    n_oil = p.n_oil
    NA = p.NA
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    if scatter_field.ndim > 1:
        angles = np.linspace(0, capture_angle_medium, scatter_field.shape[0])
        sfield = interp1d(angles, scatter_field)
    else:
        sfield = lambda angle: 1
    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_glass, angle_glass, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        return (sfield(angle_oil)*B(1, angle_oil, r, p)*
            t_p_1*t_p_2 * n_oil/n_glass * np.sin(angle_oil))

    return quad(integrand, 0, capture_angle_medium, complex_func=True)[0]

def Integral_2(r, p, scatter_field):
    n_medium = p.n_medium
    n_glass = p.n_glass
    n_oil = p.n_oil
    NA = p.NA
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    if scatter_field.ndim > 1:
        angles = np.linspace(0, capture_angle_medium, scatter_field.shape[0])
        sfield = interp1d(angles, scatter_field)
    else:
        sfield = lambda angle: 1
    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_glass, angle_glass, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        t_s_1 = t_s(n_medium, angle_medium, n_glass, angle_glass)
        t_s_2 = t_s(n_glass, angle_glass, n_oil, angle_oil)
        return (sfield(angle_oil)*B(2, angle_oil, r, p)*
            (t_s_1*t_s_2 - t_p_1*t_p_2/n_medium*np.sqrt(n_medium**2 - n_oil**2*np.sin(angle_oil)**2)))

    return quad(integrand, 0, capture_angle_medium, complex_func=True)[0]

def calculate_intensities(scatter_field, p: DesignParams, camera: Camera, r_resolution: int = 50):
    rs = np.linspace(0, np.max(camera.r), r_resolution)
    wavelen = p.wavelen*10**-9 # m
    n_oil = p.n_oil
    k = -2*np.pi/wavelen
    scatter_field_polarization = (scatter_field.real.T/np.linalg.norm(scatter_field.real,axis=-1)).T
    scatter_field_amplitude = np.sum(scatter_field*scatter_field_polarization, axis=-1)
    if p.unpolarized:
        
        I_0 = interp1d(rs, np.array([Integral_0(r, p, scatter_field_amplitude) for r in rs]))(camera.r)
        I_1 = interp1d(rs, np.array([Integral_1(r, p, scatter_field_amplitude) for r in rs]))(camera.r)
        I_2 = interp1d(rs, np.array([Integral_2(r, p, scatter_field_amplitude) for r in rs]))(camera.r)

        scatter_intensity = 8*np.pi/3 * np.real(I_0**2 + 2*I_1**2 + I_2**2)
        reference_intensity = np.ones_like(scatter_intensity)
        interference_intensity = np.zeros_like(scatter_intensity)
        return {'if':interference_intensity,
            'scat':scatter_intensity,
            'ref':reference_intensity,
            'sig':scatter_intensity+interference_intensity,
            'tot':interference_intensity+scatter_intensity+reference_intensity}
    else:
        I_0 = interp1d(rs, np.array([Integral_0(r, p, scatter_field_amplitude) for r in rs]))(camera.r)
        I_1 = interp1d(rs, np.array([Integral_1(r, p, scatter_field_amplitude) for r in rs]))(camera.r)
        I_2 = interp1d(rs, np.array([Integral_2(r, p, scatter_field_amplitude) for r in rs]))(camera.r)
        
        e_x = 1j*(I_0 + I_2*np.cos(2*camera.phi))
        e_y = 1j*I_2*np.sin(2*camera.phi)
        e_z = 2*I_1*np.cos(camera.phi)

        
        detector_field =  np.array([e_x, e_y, e_z]).T #*np.exp(1j*gouy(p))# -k*n_oil/2 *
        if scatter_field.ndim == 1:
            detector_field *= scatter_field
        else:
            detector_field *= scatter_field_polarization[0]
        
        # Reference field also has opd
        reference_field = np.ones_like(detector_field)*np.array([np.cos(np.radians(p.azimuth)), np.sin(np.radians(p.azimuth)),0])
        # correct detector field for reference phase
        detector_field /= np.exp(1j*k*opd_ref(p))
        reference_intensity = np.sum(np.abs(reference_field)**2, axis=-1)
        interference_intensity = 2*np.sum(np.real((detector_field*np.conj(reference_field))), axis=-1)/reference_intensity
        scatter_intensity = np.sum(np.abs(detector_field)**2, axis=-1)/reference_intensity
        
        return {'if':interference_intensity,
            'scat':scatter_intensity,
            'ref':reference_intensity,
            'sig':scatter_intensity+interference_intensity,
            'tot':interference_intensity+scatter_intensity+reference_intensity,
            'fields':(detector_field, scatter_field)}

def calculate_scatter_field(p, multipolar=True, angular=False):
    if multipolar:
        return calculate_scatter_field_mie(p, angular)
    else:
        return calculate_scatter_field_dipole(p)

def calculate_scatter_field_dipole(p):
    n_glass = p.n_glass
    n_medium = p.n_medium
    n_oil = p.n_oil
    n_scat = p.n_scat
    # polarization = np.array([np.cos(np.radians(p.inclination))*np.cos(np.radians(p.azimuth)), 
    #                          np.cos(np.radians(p.inclination))*np.sin(np.radians(p.azimuth)), 
    #                          np.sin(np.radians(p.inclination))])
    a = p.a*10**-9
    NA = p.NA
    wavelen = p.wavelen*10**-9

    scatter_field = 1 + 0j
    # Reference field polarization, phase and magnitude
    E_reference = np.array([np.cos(np.radians(p.azimuth)), np.sin(np.radians(p.azimuth)),0])

    # Backpropagate reference to excitation
    r_gm = (n_glass - n_medium)/(n_glass + n_medium)
    t_gm = 2*n_glass/(n_glass + n_medium)
    t_go = 2*n_glass/(n_glass + n_oil)
    E_excitation = E_reference/t_go/r_gm*t_gm
    scatter_field = E_excitation*scatter_field

    # Magnitude and phase of scatter field
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    collection_efficiency = capture_angle_medium/np.pi
    k = -2*np.pi*n_medium/wavelen
    e_scat = n_scat**2
    e_medium = n_medium**2
    polarizability = 4*np.pi*a**3*(e_scat-e_medium)/(e_scat + 2*e_medium)

    # if (x_mie > 0.1):
    #     print("Exceeded bounds of Rayleigh approximation")
    scatter_cross_section = k**4/6/np.pi *polarizability**2
    
    scatter_field *= np.sqrt(scatter_cross_section)*collection_efficiency
    return scatter_field

def calculate_scatter_field_mie(p, angular):
    import miepython as mie
    n_glass = p.n_glass
    n_medium = p.n_medium
    n_oil = p.n_oil
    n_scat = p.n_scat
    a = p.a*10**-9
    NA = p.NA
    # polarization = np.array([np.cos(np.radians(p.inclination))*np.cos(np.radians(p.azimuth)), 
    #                          np.cos(np.radians(p.inclination))*np.sin(np.radians(p.azimuth)), 
    #                          np.sin(np.radians(p.inclination))])
    wavelen = p.wavelen*10**-9

    scatter_field = 1 + 0j
    # Reference field: in plane it is parallel to the particle
    E_reference = np.array([np.cos(np.radians(p.azimuth)), np.sin(np.radians(p.azimuth)),0])

    # Backpropagate reference to excitation
    r_gm = (n_glass - n_medium)/(n_glass + n_medium)
    t_gm = 2*n_glass/(n_glass + n_medium)
    t_go = 2*n_glass/(n_glass + n_oil)
    E_excitation = E_reference/t_go/r_gm*t_gm
    scatter_field = (E_excitation*scatter_field)

    # Magnitude and phase of scatter field
    # capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    # collection_efficiency = capture_angle_medium/np.pi
    x_mie = 2*np.pi*a*n_medium / wavelen
    m = n_scat/n_medium
    if angular:
        scatter_field = scatter_field[:,None]
        capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
        angles = np.linspace(np.pi, np.pi-capture_angle_medium, 100)
        mu = np.cos(angles)
    else:
        mu = -1


    # S1 and S2 give the scatter amplitudes perpendicular (S2) and parallel (S1) to the incoming light
    S1, S2 = mie.S1_S2(m, x_mie, mu)
    # In the scattered, the real part is the amplitude and the angle gives the scatter phase.
    scatter_phase = np.angle(S1)
    scatter_amplitude = np.real(S1)*np.cos(np.radians(p.inclination))
    #scattered_amplitude_z = np.sin(np.radians(p.inclination))*S2
    scatter_field = scatter_field*scatter_amplitude*np.exp(1j*scatter_phase)
    
    return scatter_field.T