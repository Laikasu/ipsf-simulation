"""
Implement functions to simulate and model an interferometric microscopy system on spherical particles.
It contains
1: scattering
2a: propagation towards objective
2b: projection from objective onto camera
"""

import numpy as np

import miepython as mie

from numpy.typing import NDArray
from scipy.special import jv
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d
from importlib import resources


# Gold

# Johnsonn and Christy data for gold
# https://refractiveindex.info/?shelf=main&book=Au&page=Johnson
with resources.files("model").joinpath('Johnson.csv').open("r") as file:
    gold = np.genfromtxt(file, delimiter=',', skip_header=1).T
_gold_wavelen = gold[0]
_n_gold = gold[1] - 1j*gold[2]
n_gold = interp1d(_gold_wavelen*10**-6, _n_gold, kind='cubic')

import scipy.constants as const

def drude_gold(wavelen):
    """
    Get the refractive index of gold according only to the Drude model.
    """
    # Johnson and Christy
    f = 1
    tau = 9*10**-15
    damping = const.hbar/tau/const.e
    res_p = 9.06
    

    freq_eV = const.h * const.c / (wavelen) / const.e

    drude = -f * res_p**2 / (freq_eV**2 + 1j*freq_eV*damping)
    epsilon = 9 + drude
    return np.sqrt(epsilon)



# Parameters

# constants
n_ps = 1.5537
n_air = 1
n_water = 1.333
n_glass = 1.499
n_oil = 1.518
n_glyc = 1.461

# default design parameters
defaults = {
    "aberrations": False,
    "n_medium": n_water,
    "n_glass": n_glass,
    "n_glass0": n_glass,
    "n_oil0": n_oil,
    "n_oil": n_oil,
    "t_oil0": 100,       # micron
    "t_oil": 100,       # micron
    "t_glass0": 170,    # micron
    "t_glass": 170,     # micron
    "diameter": 30,     # nm
    "z_p": 0,              # micron
    "defocus": 0,          # um
    "wavelen": 532,     # nm
    "NA": 1.4,
    "multipolar": False,
    "roi_size": 2,      # micron
    "pxsize": 3.45,     # micron
    "magnification": 60,
    "scat_mat": "gold",
    "n_custom": n_ps,
    "x0": 0,               # micron
    "y0": 0,               # micron
    "r_resolution": 30,
    "efficiency": 1.,   # Modification to the E_scat/E_ref ratio
    # Angles / Polarization
    "anisotropic": False,
    "azimuth": 0,
    "inclination": 0,
    "polarized": False,
    "dipole": False,
    "polarization_angle": 0,
    "aspect_ratio": 1,
}

# units
microns = {'z_p', 'defocus', 't_oil0', 't_glass0', 't_oil', 't_glass', 'roi_size', 'pxsize', 'x0', 'y0'}
nanometers = {'wavelen', 'diameter'}
degrees = {'azimuth', 'inclination', 'polarization_angle'}


def create_params(**kwargs) -> dict:
    """Insert defaults and convert units"""
    # Default insertion
    params = {**defaults, **kwargs}

    # Unit conversion
    for um in microns:
        params[um] = params[um]*10**-6
    for nm in nanometers:
        params[nm] = params[nm]*10**-9
    for deg in degrees:
        params[deg] = np.radians(params[deg])

    return params



class Camera():
    """Helper class to handle coordinate conversions"""
    def __init__(self, **kwargs):
        roi_size = kwargs['roi_size']
        pxsize = kwargs['pxsize']
        magnification = kwargs['magnification']
        x0 = kwargs['x0']
        y0 = kwargs['y0']

        self.roi_size = roi_size
        self.pxsize = pxsize
        self.magnification = magnification
        self.pxsize_obj = self.pxsize/self.magnification
        pixels = int(self.roi_size//self.pxsize_obj)
        self.pixels = pixels + 1 if pixels%2 == 0 else pixels
        xs = np.linspace(-self.roi_size/2, self.roi_size/2, pixels)
        ys = np.linspace(-self.roi_size/2, self.roi_size/2, pixels)
        self.x, self.y = np.meshgrid(xs, ys)
        self.r = np.sqrt((self.x-x0)**2 + (self.y-y0)**2)
        self.phi = np.arctan2(self.y-y0, self.x-x0)


def opd(angle_oil, **kwargs):
    """
    Optical path difference between the design path of the objective (in focus, z_p = 0, RI's match design, thicknesses match design etc)
    and the actual  path where all parameters can differ.
    """
    z_p = kwargs['z_p']
    defocus = kwargs['defocus']
    aberrations = kwargs['aberrations']
    n_oil = kwargs['n_oil']
    n_oil0 = kwargs['n_oil0']
    n_glass = kwargs['n_glass']
    n_glass0 = kwargs['n_glass0']
    n_medium = kwargs['n_medium']
    t_glass = kwargs['t_glass']
    t_glass0 = kwargs['t_glass0']
    t_oil0 = kwargs['t_oil0']

    if aberrations:
        # Full opd

        # effective RI in the z direction
        n_eff = lambda RI: np.sqrt(RI**2 - n_oil**2 *np.sin(angle_oil)**2)
        # extra phase incurred by excitation beam
        excitation_opd = n_medium*z_p

        # Phase differences in different media
        medium_opd = z_p*n_eff(n_medium)
        glass_opd = t_glass*n_eff(n_glass) - t_glass0*n_eff(n_glass0)
        
        # Practical focusing condition provides real t_oil (Gibson, 1991)
        t_oil = z_p - defocus + n_oil*(-z_p/n_medium - t_glass/n_glass + t_glass0/n_glass0 + t_oil0/n_oil0)
        n_eff_oil = n_oil*np.cos(angle_oil) # simplified

        oil_opd = t_oil*n_eff_oil - t_oil0*n_eff(n_oil0)

        return  excitation_opd + medium_opd + glass_opd + oil_opd
    else:
        # simplified for constant RI and thickness

        # extra phase incurred by excitation beam
        excitation_opd = n_medium*z_p

        Dt_oil = z_p*(1 - n_oil/n_medium) - defocus
        n_eff_oil = n_oil*np.cos(angle_oil) # simplified
        
        return n_eff_oil*Dt_oil + excitation_opd

def opd_ref(**kwargs):
    """
    Optical path difference of the reference beam from the glass-medium layer to the aperture
    It travels through glass and oil at orthogonal angle.
    """
    z_p = kwargs['z_p']
    defocus = kwargs['defocus']
    aberrations = kwargs['aberrations']
    n_oil = kwargs['n_oil']
    n_oil0 = kwargs['n_oil0']
    n_glass = kwargs['n_glass']
    n_glass0 = kwargs['n_glass0']
    n_medium = kwargs['n_medium']
    t_glass = kwargs['t_glass']
    t_glass0 = kwargs['t_glass0']
    t_oil0 = kwargs['t_oil0']

    if aberrations:
        # Phase differences in different media
        glass_opd = t_glass*n_glass - t_glass0*n_glass0
        
        # Practical focusing condition provides real t_oil (Gibson, 1991)
        t_oil = z_p - defocus + n_oil*(-z_p/n_medium - t_glass/n_glass + t_glass0/n_glass0 + t_oil0/n_oil0)

        oil_opd = t_oil*n_oil - t_oil0*n_oil0

        return glass_opd + oil_opd
    else:
        # simplified for constant RI and thickness

        Dt_oil = z_p*(1 - n_oil/n_medium) - defocus
        
        return n_oil*Dt_oil


def snells_law(n1, angle1, n2):
    """
    Return angle2 given n1*angle1 = n2*angle2
    """
    return np.arcsin(n1 / n2 * np.sin(angle1))
    

def t_p(n1, angle1, n2, angle2):
    return 2*n1*np.cos(angle1)/(n2*np.cos(angle1) + n1*np.cos(angle2))

def t_s(n1, angle1, n2, angle2):
    return 2*n1*np.cos(angle1)/(n1*np.cos(angle1) + n2*np.cos(angle2))


def B(n, angle_oil, rs, **kwargs):
    wavelen = kwargs['wavelen']
    n_oil = kwargs['n_oil']

    k_0 = -2*np.pi/wavelen
    return np.sqrt(np.cos(angle_oil))*np.sin(angle_oil)*jv(n, k_0*rs*n_oil*np.sin(angle_oil))*np.exp(1j*k_0*opd(angle_oil, **kwargs))


epsrel=1e-3
def Integral_0(rs, **kwargs):
    n_medium = kwargs['n_medium']
    n_glass = kwargs['n_glass']
    n_oil = kwargs['n_oil']
    NA = kwargs['NA']

    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_medium, angle_medium, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        t_s_1 = t_s(n_medium, angle_medium, n_glass, angle_glass)
        t_s_2 = t_s(n_glass, angle_glass, n_oil, angle_oil)
        E_s, E_p = calculate_scatter_field_angular(angle_medium, **kwargs)

        return (B(0, angle_oil, rs, **kwargs)*
            (E_s*t_s_1*t_s_2 + E_p*t_p_1*t_p_2))
    
    # Vectorized over rs
    return quad_vec(integrand, 0, capture_angle_medium, epsrel=epsrel)[0]

def Integral_1(rs, **kwargs):
    n_medium = kwargs['n_medium']
    n_glass = kwargs['n_glass']
    n_oil = kwargs['n_oil']
    NA = kwargs['NA']

    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_glass, angle_glass, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)

        return (B(1, angle_oil, rs, **kwargs)*
            t_p_1*t_p_2 * np.sin(angle_medium))
    
    # Vectorized over rs
    return quad_vec(integrand, 0, capture_angle_medium, epsrel=epsrel)[0]

def Integral_2(rs, **kwargs):
    n_medium = kwargs['n_medium']
    n_glass = kwargs['n_glass']
    n_oil = kwargs['n_oil']
    NA = kwargs['NA']

    capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

    def integrand(angle_medium):
        angle_glass = snells_law(n_medium, angle_medium, n_glass)
        angle_oil = snells_law(n_glass, angle_glass, n_oil)
        t_p_1 = t_p(n_medium, angle_medium, n_glass, angle_glass)
        t_p_2 = t_p(n_glass, angle_glass, n_oil, angle_oil)
        t_s_1 = t_s(n_medium, angle_medium, n_glass, angle_glass)
        t_s_2 = t_s(n_glass, angle_glass, n_oil, angle_oil)
        E_s, E_p = calculate_scatter_field_angular(angle_medium, **kwargs)

        return (B(2, angle_oil, rs, **kwargs)*
            (E_s*t_s_1*t_s_2 - E_p*t_p_1*t_p_2))
    

    # Vectorized over rs
    return quad_vec(integrand, 0, capture_angle_medium, epsrel=epsrel)[0]



def calculate_propagation(**kwargs):
    """
    Calculate the propagation from particle to camera.
    The mathematics is radial and the radial data is interpolated to project onto a grid
    Returns a matrix which converts an xyz polarized scatter field into an S and P component at the detector.
    [Es, Ep] = M [Ex, Ey, Ez]
    """
    wavelen = kwargs['wavelen']
    r_resolution = kwargs['r_resolution']
    n_medium = kwargs['n_medium']

    camera = Camera(**kwargs)
    rs = np.linspace(0, np.max(camera.r), r_resolution)

    I_0 = interp1d(rs, Integral_0(rs, **kwargs))(camera.r)
    I_1 = interp1d(rs, Integral_1(rs, **kwargs))(camera.r)
    I_2 = interp1d(rs, Integral_2(rs, **kwargs))(camera.r)
    
    # Components for x polarization
    e_px = I_0 + I_2*np.cos(2*camera.phi)
    e_py = I_2*np.sin(2*camera.phi)
    e_pz = -2j*I_1*np.cos(camera.phi)
    # Components for y polarization
    e_sx = I_2*np.sin(2*camera.phi)
    e_sy = I_0 - I_2*np.cos(2*camera.phi)
    e_sz = -2j*I_1*np.sin(camera.phi)

    k = 2*np.pi*n_medium/wavelen
    plane_wave_decomp = 1j*k/2/np.pi
    return -plane_wave_decomp*np.stack([[e_px, e_py, e_pz],
                                        [e_sx, e_sy, e_sz]]).transpose((2, 3, 0, 1))


def calculate_fields(**kwargs) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Propagate and project the scatter field onto the detector

    Parameters
    ----------
    scatter_field : array_like
        A polarized field to propagate and project
    p : DesignParams
        A custom class containing experimental data
    """
    wavelen = kwargs['wavelen']
    polarization_angle = kwargs['polarization_angle']
    polarized = kwargs['polarized']
    n_medium = kwargs['n_medium']
    efficiency = kwargs['efficiency']
    

    # Relative signal strength change due to layer boundaries
    r_gm = (n_glass - n_medium)/(n_glass + n_medium)
    E_reference = r_gm

    # takes 2x3 matrix M takes polarization p and Mp gives E in p and s components.
    detector_field_components = calculate_propagation(**kwargs)

    # Average over angles if unpolarized
    if not polarized:
        polarization_angle = np.linspace(0, 2*np.pi, 100)
        kwargs['polarization_angle'] = polarization_angle
    
    

    
    # the first is the x-component, the second y.
    scatter_field = calculate_scatter_field(**kwargs)*-1j

    if polarized:
        detector_field = detector_field_components@scatter_field
    else:
        detector_field = np.einsum('ijab,bk->ijka', detector_field_components, scatter_field)
    
    # Apply collection efficiency modification
    detector_field *= efficiency

    ref_polarization = np.array([np.cos(polarization_angle), np.sin(polarization_angle)]).T*np.ones_like(detector_field)
    reference_field = ref_polarization*E_reference
    
    # effect of inclination on opd
    k = -2*np.pi/wavelen
    detector_field /= np.exp(1j*k*opd_ref(**kwargs))


    return detector_field, reference_field
    
def calculate_intensities(**kwargs) -> NDArray[np.floating]:
    """
    Propagate and project the scatter field onto the detector

    Parameters
    ----------
    polarized: bool
    """
    polarized = kwargs['polarized']

    detector_field, reference_field = calculate_fields(**kwargs)
    
    interference_contrast = 2*np.sum(np.real((detector_field*np.conj(reference_field))), axis=-1)
    scatter_contrast = np.sum(np.abs(detector_field)**2, axis=-1)

    

    # Average over all polarization angles if unpolarized
    if not polarized:
        interference_contrast = np.mean(interference_contrast, axis=-1)
        scatter_contrast = np.mean(scatter_contrast, axis=-1)

    
    return np.stack([interference_contrast, scatter_contrast])

def calculate_scatter_field_angular(angle, multipolar=True, **kwargs):
    if multipolar:
        return calculate_scatter_field_mie(angle, **kwargs)/calculate_scatter_field_mie(0, **kwargs)
    else:
        return np.array([1, np.cos(angle)])

def scatter_RI(**kwargs):
    scat_mat = kwargs['scat_mat']
    wavelen = kwargs['wavelen']
    
    # Check input
    if scat_mat not in {'gold', 'polystyrene', 'custom'}:
        raise ValueError(f'Scattering material {scat_mat} not implemented. Possible values: gold, polystyrene, custom')

    if scat_mat == 'gold':
        return n_gold(wavelen)
    elif scat_mat =='polystyrene':
        return n_ps
    else:
        return kwargs['n_custom']

def calculate_scatter_field_dipole(**kwargs):
    n_medium = kwargs['n_medium']
    diameter = kwargs['diameter']
    wavelen = kwargs['wavelen']
    polarization_angle = kwargs['polarization_angle']
    azimuth = kwargs['azimuth']
    inclination = kwargs['inclination']

    a = diameter/2
    n_scat = scatter_RI(**kwargs)

    
    k = 2*np.pi*n_medium/wavelen
    e_scat = n_scat**2
    e_medium = n_medium**2
    polarizability = 4*np.pi*a**3*(e_scat-e_medium)/(e_scat + 2*e_medium)


    rel_angle = polarization_angle-azimuth
    
    dir = np.cos(inclination)*np.array([np.cos(rel_angle)*np.cos(inclination)*np.cos(azimuth),
                    np.cos(rel_angle)*np.cos(inclination)*np.sin(azimuth),
                    np.cos(rel_angle)*np.sin(inclination)])
    return k**2/4/np.pi*polarizability*dir



def calculate_scatter_field_anisotropic(**kwargs):
    n_medium = kwargs['n_medium']
    diameter = kwargs['diameter']
    wavelen = kwargs['wavelen']
    aspect_ratio = kwargs['aspect_ratio']
    azimuth = kwargs['azimuth']
    inclination = kwargs['inclination']
    polarization_angle = kwargs['polarization_angle']

    n_scat = scatter_RI(**kwargs)

    # Magnitude and phase of scatter field
    k = 2*np.pi*n_medium/wavelen
    e_scat = n_scat**2
    e_medium = n_medium**2

    a = diameter/2
    c = a*aspect_ratio
    e = np.sqrt(1 - 1/aspect_ratio**2)
    if np.isclose(e, 0):
        L_parallel = 1/3
    elif np.isclose(e, 1):
        L_parallel = 0
    else:
        L_parallel = (1-e**2)/e**2 * (np.log((1+e)/(1-e))/2/e - 1)
    L_perp = (1 - L_parallel)/2

    V = np.pi*a**2*c*4/3
    polarizability = lambda L: V*(e_scat - e_medium)/(e_medium + L*(
         e_scat - e_medium))

    
    a_parallel = polarizability(L_parallel)
    a_perp = polarizability(L_perp)
    # both zero -> x a_parallel
    # polarization 45 -> x 1/sqrt(2)

    orientation_r = np.array([np.cos(inclination)*np.cos(azimuth), np.cos(inclination)*np.sin(azimuth), np.sin(inclination)])
    phi = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    theta = np.array([-np.sin(inclination)*np.cos(azimuth), -np.sin(inclination)*np.sin(azimuth), np.cos(inclination)])


    reference = np.array([np.cos(polarization_angle), np.sin(polarization_angle), np.zeros_like(polarization_angle)])



    polarization = np.squeeze(
                    np.outer(a_parallel*orientation_r, orientation_r@reference) + 
                    np.outer(a_perp*phi, phi@reference) + 
                    np.outer(a_perp*theta, theta@reference))

    return k**2/4/np.pi*polarization
    # if (x > 0.1):
    #     print("Exceeded bounds of Rayleigh approximation")
    
    # 1j delay is not captured in polarizability. The physical origin is the scattered wave being spherical and the incoming being planar.
    # For the radius of curvature to match it needs an i phase change


def calculate_scatter_field_mie(angle, **kwargs):
    """
    Supports array inputs
    """
    n_medium = kwargs['n_medium']
    diameter = kwargs['diameter']
    wavelen = kwargs['wavelen']
    a = diameter/2
    
    n_scat = scatter_RI(**kwargs)

    # Magnitude and phase of scatter field
    # capture_angle_medium = np.arcsin(min(NA/n_medium, 1))
    # collection_efficiency = capture_angle_medium/np.pi
    k = 2*np.pi*n_medium/wavelen
    x_mie = k*a
    m = n_scat/n_medium

    # Benchmark

    # # S1 and S2 give the scatter amplitudes parallel (S1) and perpendicular (S2) to the scattering plane (k_in and k_out)
    # # for backscattering S1 = -S2, I take S1
    # S1_0, S2_0 = mie.S1_S2(m, x_mie, 1, norm='wiscombe')
    # S1_pi, S2_pi = mie.S1_S2(m, x_mie, -1, norm='wiscombe')

    # backscatter_field = S1_pi/k

    # q_ext = 4*np.real(S1_0)/x_mie**2

    # # def integrand(angle):
    # #     mu = np.cos(angle)
    # #     S1, S2 = mie.S1_S2(m, x_mie, mu, norm='wiscombe')
    # #     F = (np.abs(S1)**2+np.abs(S2)**2)/2
    # #     return F*np.sin(angle)
    
    # def integrand(angle):
    #     mu = np.cos(angle)
    #     S1, S2 = mie.S1_S2(m, x_mie, mu, norm='wiscombe')
    #     S = np.squeeze([S1, S2])/np.sqrt(2)
    #     E = np.sqrt(np.sum(S**2))
    #     return np.abs(E)**2*np.sin(angle)
    
    # q_sca = 2*quad(integrand, 0, np.pi)[0]/x_mie**2

    # q_back = 4*np.abs(backscatter_field)**2/a**2


    # print(q_ext, q_sca, q_back)
    # # Other function for benchmark
    # qext, qsca, qback, g = np.vectorize(mie.efficiencies_mx)(m, x_mie)
    # print(qext, qsca, qback)
    # exit()

    mu = -np.cos(angle)
    S1, S2 = mie.S1_S2(m, x_mie, mu, norm='wiscombe')
    S = np.squeeze([S1, S2])
    return S/1j/k

def calculate_scatter_field(multipolar=True, dipole=False, **kwargs):
    polarization_angle = kwargs['polarization_angle']
    if multipolar:
        return calculate_scatter_field_mie(0, **kwargs)[0]*np.array([np.cos(polarization_angle), np.sin(polarization_angle), np.zeros_like(polarization_angle)])
    
    if dipole:
        return calculate_scatter_field_dipole(**kwargs)
    
    return calculate_scatter_field_anisotropic(**kwargs)




# User convenience functions for elegant numpy usage

# backscattering
def simulate_backscattering(**kwargs):
    """Simulate Mie/dipole scattering"""
    params = create_params(**kwargs)

    scatter_field = np.squeeze(vectorize_array(calculate_scatter_field, **params))
    return scatter_field[0]

def vectorize_array(func, **kwargs):
    """Vectorization with array output"""
    # Vectorize loses type
    ret = np.vectorize(func, otypes=[np.ndarray])(**kwargs)
    
    stack = np.array(ret)
    if stack.dtype == object:
        stack = np.array(stack.tolist())
    
    # Force the type
    #stack = np.array(stack.tolist(), dtype=dtype)
    # 1D
    if ret.shape == stack.shape:
        return stack
    return np.moveaxis(stack,ret.ndim, 0)

def simulate_center(**kwargs) -> NDArray[np.float64]:
    """Simulate only the center pixel"""
    params = create_params(**kwargs)
    # Single pixel
    params['r_resolution'] = 2
    roi_size = params['pxsize']/params['magnification']
    params['roi_size'] = roi_size

    return np.squeeze(vectorize_array(calculate_intensities, **params))


def simulate_field(**kwargs) -> NDArray[np.complex128]:
    """Simulate the field at the center pixel"""
    params = create_params(**kwargs)
    # Single pixel
    params['r_resolution'] = 2
    roi_size = params['pxsize']/params['magnification']
    params['roi_size'] = roi_size

    return np.squeeze(vectorize_array(calculate_fields, **params))

def simulate_camera(**kwargs) -> NDArray[np.complex128]:
    """Simulate an entire sensor"""
    # Unit conversions
    params = create_params(**kwargs)
    return vectorize_array(calculate_intensities, **params)


def polarization(vector: np.ndarray) -> np.ndarray:
    v = np.abs(vector)
    return v/np.linalg.norm(v,axis=-1,keepdims=True)*np.sign(v[...,0:1])

def magnitude(vector: np.ndarray) -> np.complex128:
    # dot product
    return np.einsum('...i,...i', polarization(vector), np.conj(vector))