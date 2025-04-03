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
                n_oil0: float = 1.5,
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
                magnification: float = 80,
                scat_mat: str = "polystyrene",
                x0: float = 0,
                y0: float = 0):
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
        self.scat_mat = scat_mat
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
    
    @property
    def scat_mat(self):
        return self._scat_mat
        
    @scat_mat.setter
    def scat_mat(self, value: str):
        if value == "gold":
            self.n_scat = n_gold(self._wavelen)
            self._scat_mat = value
        elif value == "polystyrene":
            self.n_scat = n_ps
            self._scat_mat = value
        else:
            raise ValueError(f"Material {value} not found")
        
    @property
    def wavelen(self):
        return self._wavelen
    
    @wavelen.setter
    def wavelen(self, value):
        self._wavelen = value
        if self.scat_mat == "gold":
            self.n_scat = n_gold(value)
    
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

    opd = (n_oil*np.cos(angle_oil)*(z_p - z_focus + n_oil*(-z_p/n_medium - t_glass/n_glass + t_glass0/n_glass0 + t_oil0/n_oil0)) +
        z_p*np.sqrt(n_medium**2 - n_oil**2 *np.sin(angle_oil)**2) + t_glass*np.sqrt(n_glass**2 - n_oil**2 *np.sin(angle_oil)**2) -
        t_glass0*np.sqrt(n_glass0**2 - n_oil**2 *np.sin(angle_oil)**2) - t_oil0*np.sqrt(n_oil0**2 - n_oil**2 *np.sin(angle_oil)**2))
    return opd + n_medium*z_p + n_glass*t_glass + n_oil*t_oil - n_glass0*t_glass0 - n_oil0*t_oil0


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

    k = 2*np.pi/wavelen
    #angle_medium = snells_law(n_oil, angle_oil, n_medium)
    #mu = np.cos(angle_medium)
    #x_mie = 2*np.pi*a*n_medium / wavelen
    #profile = mie.i_par(n_scat/n_medium, x_mie, mu)
    return np.sqrt(np.cos(angle_oil))*np.sin(angle_oil)*jv(n, k*r*n_oil*np.sin(angle_oil))*np.exp(1j*k*opd(angle_oil, p))


def Integral_0(r, p):
    n_medium = p.n_medium
    n_glass = p.n_glass
    n_oil = p.n_oil
    NA = p.NA
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_medium, angle_medium, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        t_s_1 = t_s(n_medium, angle_medium, n_glass, angle_glass)
        t_s_2 = t_s(n_glass, angle_glass, n_oil, angle_oil)
        return (B(0, angle_oil, r, p)*
            (t_s_1*t_s_2 + t_p_1*t_p_2/n_medium*np.sqrt(n_medium**2 - n_oil**2*np.sin(angle_oil)**2)))

    return quad(integrand, 0, capture_angle_medium, complex_func=True)[0]

def Integral_1(r, p):
    n_medium = p.n_medium
    n_glass = p.n_glass
    n_oil = p.n_oil
    NA = p.NA
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_glass, angle_glass, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        return (B(1, angle_oil, r, p)*
            t_p_1*t_p_2 * n_oil/n_glass * np.sin(angle_oil))

    return quad(integrand, 0, capture_angle_medium, complex_func=True)[0]

def Integral_2(r, p):
    n_medium = p.n_medium
    n_glass = p.n_glass
    n_oil = p.n_oil
    NA = p.NA
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_glass, angle_glass, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        t_s_1 = t_s(n_medium, angle_medium, n_glass, angle_glass)
        t_s_2 = t_s(n_glass, angle_glass, n_oil, angle_oil)
        return (B(2, angle_oil, r, p)*
            (t_s_1*t_s_2 - t_p_1*t_p_2/n_medium*np.sqrt(n_medium**2 - n_oil**2*np.sin(angle_oil)**2)))

    return quad(integrand, 0, capture_angle_medium, complex_func=True)[0]

def calculate_intensities(scatter_field, p: DesignParams, camera: Camera, r_resolution: int = 50):

    rs = np.linspace(0, np.max(camera.r), r_resolution)
    wavelen = p.wavelen*10**-9 # m
    n_oil = p.n_oil
    polarization = np.array([np.cos(np.radians(p.inclination))*np.cos(np.radians(p.azimuth)), 
                             np.cos(np.radians(p.inclination))*np.sin(np.radians(p.azimuth)), 
                             np.sin(np.radians(p.inclination))])
    k = 2*np.pi/wavelen
    if p.unpolarized:
        
        I_0 = interp1d(rs, np.array([Integral_0(r, p) for r in rs]))(camera.r)
        I_1 = interp1d(rs, np.array([Integral_1(r, p) for r in rs]))(camera.r)
        I_2 = interp1d(rs, np.array([Integral_2(r, p) for r in rs]))(camera.r)

        scatter_intensity = 8*np.pi/3 * np.real(I_0**2 + 2*I_1**2 + I_2**2)
        reference_intensity = np.ones_like(scatter_intensity)
        interference_intensity = np.zeros_like(scatter_intensity)
        return {'if':interference_intensity,
            'scat':scatter_intensity,
            'ref':reference_intensity,
            'sig':scatter_intensity+interference_intensity,
            'tot':interference_intensity+scatter_intensity+reference_intensity}
    else:
        I_0 = interp1d(rs, np.array([Integral_0(r, p) for r in rs]))(camera.r)
        I_1 = interp1d(rs, np.array([Integral_1(r, p) for r in rs]))(camera.r)
        I_2 = interp1d(rs, np.array([Integral_2(r, p) for r in rs]))(camera.r)
        # x, y = np.meshgrid(camera.xs, camera.ys)
        # xy_rs = np.argmin(np.abs(np.sqrt(x**2 + y**2)[:,:,np.newaxis] - rs), axis=2)
        # I_0 = np.array([Integral_0(r, p) for r in rs])[xy_rs]
        # I_1 = np.array([Integral_1(r, p) for r in rs])[xy_rs]
        # I_2 = np.array([Integral_2(r, p) for r in rs])[xy_rs]
        
        # e_x = 1j*I_0 + I_2*np.cos(2*phi)
        e_x = 1j*(I_0 + I_2*np.cos(2*camera.phi))
        e_y = 1j*I_2*np.sin(2*camera.phi)
        e_z = 2*I_1*np.cos(camera.phi)
        detector_field = -k*n_oil/2 * scatter_field*np.array([e_x, e_y, e_z])


        reference_field = np.ones_like(detector_field)*polarization[:,np.newaxis,np.newaxis]
        interference_intensity = 2*np.sum(np.real((detector_field*np.conj(reference_field))), axis=0)
        scatter_intensity = np.sum(np.abs(detector_field)**2, axis=0)
        reference_intensity = np.sum(np.abs(reference_field)**2, axis=0)
        return {'if':interference_intensity,
            'scat':scatter_intensity,
            'ref':reference_intensity,
            'sig':scatter_intensity+interference_intensity,
            'tot':interference_intensity+scatter_intensity+reference_intensity}


def calculate_scatter_field(p):
    n_glass = p.n_glass
    n_medium = p.n_medium
    n_oil = p.n_oil
    n_scat = p.n_scat
    polarization = np.array([np.cos(np.radians(p.inclination))*np.cos(np.radians(p.azimuth)), 
                             np.cos(np.radians(p.inclination))*np.sin(np.radians(p.azimuth)), 
                             np.sin(np.radians(p.inclination))])
    a = p.a*10**-9
    NA = p.NA
    wavelen = p.wavelen*10**-9

    scatter_field = 1 + 0j
    # Reference field polarization, phase and magnitude
    if p.unpolarized:
        E_reference = 1
    else:
        E_reference = polarization[:,np.newaxis,np.newaxis] # x_polarization with size 1

    # Backpropagate reference to excitation
    r_gm = (n_glass - n_medium)/(n_glass + n_medium)
    t_gm = 2*n_glass/(n_glass + n_medium)
    t_go = 2*n_glass/(n_glass + n_oil)
    E_excitation = E_reference/t_go/r_gm*t_gm
    scatter_field = E_excitation*scatter_field

    # Magnitude and phase of scatter field
    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    collection_efficiency = capture_angle_medium/np.pi
    x_mie = 2*np.pi*a*n_medium / wavelen
    k = 2*np.pi*n_medium/wavelen
    e_scat = n_scat**2
    e_medium = n_medium**2
    polarizability = 4*np.pi*a**3*(e_scat-e_medium)/(e_scat + 2*e_medium)

    # if (x_mie > 0.1):
    #     print("Exceeded bounds of Rayleigh approximation")
    scatter_cross_section = k**4/6/np.pi *polarizability**2
    
    scatter_field *= np.sqrt(scatter_cross_section)*collection_efficiency
    return scatter_field

# def calculate_scatter_field_mie(p):
#     import miepython as mie
#     n_glass = p.n_glass
#     n_medium = p.n_medium
#     n_oil = p.n_oil
#     n_scat = p.n_scat
#     a = p.a*10**-9
#     NA = p.NA
#     polarization = np.array([np.cos(np.radians(p.inclination))*np.cos(np.radians(p.azimuth)), 
#                              np.cos(np.radians(p.inclination))*np.sin(np.radians(p.azimuth)), 
#                              np.sin(np.radians(p.inclination))])
#     wavelen = p.wavelen*10**-9

#     scatter_field = 1 + 0j
#     # Reference field polarization, phase and magnitude
#     E_reference = polarization[:,np.newaxis,np.newaxis] # x_polarization with size 1

#     # Backpropagate reference to excitation
#     r_gm = (n_glass - n_medium)/(n_glass + n_medium)
#     t_gm = 2*n_glass/(n_glass + n_medium)
#     t_go = 2*n_glass/(n_glass + n_oil)
#     E_excitation = E_reference/t_go/r_gm*t_gm
#     scatter_field = E_excitation*scatter_field

#     # Magnitude and phase of scatter field
#     capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
#     collection_efficiency = capture_angle_medium/np.pi
#     x_mie = 2*np.pi*a*n_medium / wavelen
#     k = 2*np.pi*n_medium/wavelen
#     e_scat = n_scat**2
#     e_medium = n_medium**2
#     polarizability = 4*np.pi*a**3*(e_scat-e_medium)/(e_scat + 2*e_medium)

#     qext, qsca, qback, g = mie.mie(n_scat/n_medium, x_mie)
#     scatter_cross_section = qsca * np.pi * a**2

#     scatter_phase = np.angle(polarizability)
    
#     scatter_field *= np.sqrt(scatter_cross_section)*collection_efficiency*np.exp(1j*scatter_phase)
#     return scatter_field

# def calculate_scatter_field_mie_dipole(p):
#     n_glass = p.n_glass
#     n_medium = p.n_medium
#     n_oil = p.n_oil
#     n_scat = p.n_scat
#     a = p.a
#     NA = p.NA
#     polarization = np.array([np.cos(np.radians(p.inclination))*np.cos(np.radians(p.azimuth)), 
#                              np.cos(np.radians(p.inclination))*np.sin(np.radians(p.azimuth)), 
#                              np.sin(np.radians(p.inclination))])
#     wavelen = p.wavelen*10**-9 #m

#     scatter_field = 1 + 0j
#     # Reference field polarization, phase and magnitude
#     E_reference = polarization[:,np.newaxis,np.newaxis] # x_polarization with size 1

#     # Backpropagate reference to excitation
#     r_gm = (n_glass - n_medium)/(n_glass + n_medium)
#     t_gm = 2*n_glass/(n_glass + n_medium)
#     t_go = 2*n_glass/(n_glass + n_oil)
#     E_excitation = E_reference/t_go/r_gm*t_gm
#     scatter_field = E_excitation*scatter_field

#     # Magnitude and phase of scatter field
#     capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
#     collection_efficiency = capture_angle_medium/np.pi
#     x_mie = 2*np.pi*a*n_medium / wavelen
#     k = 2*np.pi*n_medium/wavelen
#     e_scat = n_scat**2
#     e_medium = n_medium**2
#     polarizability = 4*np.pi*a**3*(e_scat-e_medium)/(e_scat + 2*e_medium)

#     mu = 

#     Ipar = mie.i_par(n_scat/n_medium, x_mie, mu)
#     qext, qsca, qback, g = mie(n_scat/n_medium, x_mie)
#     scatter_cross_section = qsca * np.pi * a**2

#     scatter_phase = np.angle(polarizability)
    
#     scatter_field *= np.sqrt(scatter_cross_section)*collection_efficiency*np.exp(1j*scatter_phase)
#     return scatter_field
    