#%% Imports
import matplotlib.pyplot as plt
import numpy as np
#import miepython
from scipy.special import jv
from scipy.integrate import quad

#%% Values
e_scat = -4.2137 + 2.5289j #https://refractiveindex.info gold for 525nm
#n_scat = np.sqrt(e_scat)
#n_scat = np.real(n_scat) - np.imag(n_scat)*1j
n_scat = 1.5537 
n_air = 1
n_water = 1.333
n_glass = 1.5
n_oil = 1.515
n_glyc = 1.461
n_medium = n_glyc
m = n_scat/n_medium
e_medium = n_medium**2
t_glass = 170*10**-6
t_oil = 0*10**-6


#nominal
n_oil0 = n_oil
n_glass0 = n_glass
t_glass0 = t_glass
t_oil0 = t_oil


wavelen = 525*10**-9
k = 2*np.pi/wavelen
a = 20*10**-9 /2
#x = 2*np.pi*a / wavelen
NA = 1.4 #1.49
z_focus = 0
z_p = 0

# The polarization of the scattered radiation equals the polarization of the scatterer
polarization = np.array([1, 0, 0]) # x polarization


#%% Functions


def opd(angles):
    opd = (n_oil*np.cos(angles)*(z_p - z_focus + n_oil*(-z_p/n_medium - t_glass/n_glass + t_glass0/n_glass0 + t_oil0/n_oil0)) +
        z_p*np.sqrt(n_medium**2 - n_oil**2 *np.sin(angles)**2) + t_glass*np.sqrt(n_glass**2 - n_oil**2 *np.sin(angles)**2) -
        t_glass0*np.sqrt(n_glass0**2 - n_oil**2 *np.sin(angles)**2) - t_oil0*np.sqrt(n_oil0**2 - n_oil**2 *np.sin(angles)**2))
    return opd + n_medium*z_p + n_glass*t_glass + n_oil*t_oil - n_glass0*t_glass0 - n_oil0*t_oil0


def snells_law(n1, n2, incident_angle):
    transmitted_angle = np.arcsin(n1 / n2 * np.sin(incident_angle))
    return transmitted_angle
    

def t_p(n1, n2, incident_angle, transmitted_angle):
    return 2*n1*np.cos(incident_angle)/(n2*np.cos(incident_angle) + n1*np.cos(transmitted_angle))

def t_s(n1, n2, incident_angle, transmitted_angle):
    return 2*n1*np.cos(incident_angle)/(n1*np.cos(incident_angle) + n2*np.cos(transmitted_angle))


def B(n, angle, r):
    return np.sqrt(np.cos(angle))*np.sin(angle)*jv(n, k*r*n_oil*np.sin(angle))*np.exp(1j*k*opd(angle))


def Integral_0(r):
    def integrand(angle):
        glass_angle = snells_law(n_medium, n_glass, angle)
        aperture_angle = snells_law(n_glass, n_oil, glass_angle)
        t_p_1 = t_p(n_medium, n_glass, angle, glass_angle)
        t_p_2 = t_p(n_glass, n_oil, glass_angle, aperture_angle)
        t_s_1 = t_s(n_medium, n_glass, angle, glass_angle)
        t_s_2 = t_s(n_glass, n_oil, glass_angle, aperture_angle)
        return B(0, aperture_angle, r)*(t_s_1*t_s_2 + t_p_1*t_p_2/n_medium*np.sqrt(n_medium**2 - n_oil**2*np.sin(aperture_angle)**2))

    return quad(integrand, 0, capture_angle_oil, complex_func=True)[0]

def Integral_1(r):
    def integrand(angle):
        glass_angle = snells_law(n_medium, n_glass, angle)
        aperture_angle = snells_law(n_glass, n_oil, glass_angle)
        t_p_1 = t_p(n_medium, n_glass, angle, glass_angle)
        t_p_2 = t_p(n_glass, n_oil, glass_angle, aperture_angle)
        return B(1, aperture_angle, r)* t_p_1*t_p_2 * n_oil/n_glass * np.sin(aperture_angle)

    return quad(integrand, 0, capture_angle_oil, complex_func=True)[0]

def Integral_2(r):
    def integrand(angle):
        glass_angle = snells_law(n_medium, n_glass, angle)
        aperture_angle = snells_law(n_glass, n_oil, glass_angle)
        t_p_1 = t_p(n_medium, n_glass, angle, glass_angle)
        t_p_2 = t_p(n_glass, n_oil, glass_angle, aperture_angle)
        t_s_1 = t_s(n_medium, n_glass, angle, glass_angle)
        t_s_2 = t_s(n_glass, n_oil, glass_angle, aperture_angle)
        return B(2, aperture_angle, r)*(t_s_1*t_s_2 - t_p_1*t_p_2/n_medium*np.sqrt(n_medium**2 - n_oil**2*np.sin(aperture_angle)**2))

    return quad(integrand, 0, capture_angle_oil, complex_func=True)[0]

def calculate_scatter_field():
    global scatter_field
    global E_reference

    scatter_field = 1 + 0j
    # Reference field polarization, phase and magnitude
    E_reference = polarization[:,np.newaxis,np.newaxis] # x_polarization with size 1

    r_gm = (n_glass - n_medium)/(n_glass + n_medium)
    t_gm = 2*n_glass/(n_glass + n_medium)
    t_go = 2*n_glass/(n_glass + n_oil)
    E_excitation = E_reference/t_go/r_gm*t_gm
    E_reference *= np.sign(r_gm).astype(E_reference.dtype)
    scatter_field = E_excitation*scatter_field

    # Magnitude and phase of scatter field
    collection_efficiency = capture_angle_medium/np.pi
    polarizability = 4*np.pi*a**3*(e_scat-e_medium)/(e_scat + 2*e_medium)
    scatter_phase = np.angle(polarizability)
    scatter_cross_section = k**4/6/np.pi *np.abs(polarizability)**2
    scatter_field *= np.sqrt(scatter_cross_section)*collection_efficiency*np.exp(1j*scatter_phase)

def calculate_intensities():
    I_0 = np.array([Integral_0(r) for r in rs])[xy_rs]
    I_1 = np.array([Integral_1(r) for r in rs])[xy_rs]
    I_2 = np.array([Integral_2(r) for r in rs])[xy_rs]

    detector_field = -1j/wavelen*scatter_field*np.array([I_0 + I_2*np.cos(2*phi), I_2*np.sin(2*phi), -2j*I_1*np.cos(phi)])


    reference_field = np.ones_like(detector_field)
    # get the initial field back
    reference_field *= E_reference
    interference_intensity = 2*np.real(np.sum((detector_field*np.conj(reference_field)), axis=0))
    scatter_intensity = np.sum(np.abs(detector_field)**2, axis=0)
    reference_intensity = np.sum(np.abs(reference_field)**2, axis=0)
    return {'if':interference_intensity,
        'scat':scatter_intensity,
        'ref':reference_intensity,
        'tot':interference_intensity+scatter_intensity+reference_intensity}
#%%
# Pre-interface

capture_angle_oil = np.arcsin(min(NA/n_oil, 1))
capture_angle_medium = np.arcsin(min(NA/n_medium, 1))

calculate_scatter_field()


#%%
# Camera definitions

camsize = 2 # micron
#sampling points for detector
xs = np.linspace(-camsize/2, camsize/2, 64)*10**-6
ys = np.linspace(-camsize/2, camsize/2, 64)*10**-6
x, y = np.meshgrid(xs, ys)


phi = np.arctan(y/x)
rs = np.linspace(0, np.sqrt(2)*camsize/2, 200)*10**-6
xy_rs = np.argmin(np.abs(np.sqrt(x**2 + y**2)[:,:,np.newaxis] - rs), axis=2)


#%%
# Single plot
diameter = 50*10**-9
z_focus = 0*10**-6
z_p = 0*10**-6
a = diameter/2
calculate_scatter_field()
intensity = calculate_intensities()

#%%
# Single plot plot
fig, ax = plt.subplots(1, 2)
p1 = ax[0].imshow(intensity['if'], extent=[-camsize, camsize, -camsize, camsize], cmap = 'gray')
fig.colorbar(p1, ax=ax[0])
ax[0].set_xlabel('x position(micron)')
ax[0].set_ylabel('y position(micron)')
ax[0].set_title('Interference intensity')
p2 = ax[1].imshow(intensity['tot'], extent=[-camsize, camsize, -camsize, camsize], cmap = 'gray')
ax[1].set_xlabel('x position(micron)')
ax[1].set_ylabel('y position(micron)')
fig.colorbar(p2, ax=ax[1])
ax[1].set_title('Total intensity')
plt.show()

#%%
# Contrast plot

z_focus = 0*10**-6
z_p = 0*10**-6
diameters = np.linspace(10, 80, 10)*10**-9
radii = diameters/2
contrast = np.zeros_like(radii)
scatter_intensity = np.zeros_like(radii)
interference_intensity = np.zeros_like(radii)

for i, a in enumerate(radii):
    calculate_scatter_field()
    intensity = calculate_intensities()
    scatter_intensity[i] = np.max(np.abs(intensity['scat']))
    interference_intensity[i] = np.max(np.abs(intensity['if']))
    # peak intensity vs reference
    contrast[i] = np.max(np.abs(intensity['tot'] - 1))


#%%
# Contrast plot plot

fig, ax = plt.subplots()
ax.plot(diameters/10**-9, contrast, label='total contrast')
ax.plot(diameters/10**-9, scatter_intensity, label='scatter contrast', color='red')
ax.plot(diameters/10**-9, interference_intensity, label='interference contrast', color='green')
ax.legend()
ax.set_xlabel('Polystyrene diameter(nm)')
ax.set_ylabel('Contrast')
ax.set_title('Polystyrene size and contrast')
plt.show()

#%%
# Move plot

# 1x5 fixed focus, variable particle position
z_pp = np.array([0.27, 0.82, 1.5, 2.15, 3.12])*10**-6 #micron
z_focus = 1.5*10**-6


move_intensities = np.zeros((len(z_pp), len(xs), len(ys)))
for m, z_p in enumerate(z_pp):
    intensity = calculate_intensities()
    move_intensities[m] = intensity['tot']

#%%
# Move plot plot
fig, ax = plt.subplots(5)
for i in range(5):
    ax[i].imshow(move_intensities[i].T, cmap = 'gray')
    ax[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout()
plt.show()

#%%
# 5x5
z_focuss = np.linspace(3, 5, 5)*10**-6 #micron
z_ps = np.linspace(2.9, 3.1, 5)*10**-6


ff_intensities = np.zeros((len(z_focuss), len(z_ps), len(xs), len(ys)))
for m, z_focus in enumerate(z_focuss):
    for n, z_p in enumerate(z_ps):
        intensity = calculate_intensities()
        ff_intensities[m, n] = intensity['tot']

#%%
# 5x5 plot


fig, ax = plt.subplots(5, 5)
vmin = np.min(ff_intensities)
vmax = np.max(ff_intensities)

for i in range(5):
    for j in range(5):
        x = j
        y = -(i+1)
        ax[i, j].imshow(ff_intensities[x, y], cmap = 'gray', vmin=vmin, vmax=vmax)
        ax[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.show()
# %%
