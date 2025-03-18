# %%
import matplotlib.pyplot as plt
import numpy as np
#import miepython
from scipy.constants import c, epsilon_0

# %%

e_scat = -4.2137 + 2.5289j #https://refractiveindex.info gold for 525nm
n_scat = np.sqrt(e_scat)
n_scat = np.real(n_scat) - np.imag(n_scat)*1j
n_air = 1
n_water = 1.333
n_glass = 1.5
n_oil = 1.515
n_glyc = 1.461
n_medium = n_glyc
m = n_scat/n_medium
e_medium = n_medium**2
t_glass = 170*10**-6
t_oil = 10*10**-6


#nominal
n_oil0 = n_oil
n_glass0 = n_glass
t_glass0 = t_glass
t_oil0 = t_oil


wavelen = 525*10**-9
a = 40*10**-9 /2
x = 2*np.pi*a / wavelen
NA = 1.4 #1.49
k = 2*np.pi/wavelen
z_focus = 0
z_p = 0

# The polarization of the scattered radiation equals the polarization of the scatterer
polarization = np.array([1, 0, 0]) # x polarization

#%%

# Optical functions

def opd_correction(angles):
    opd = (n_oil*np.cos(angles)*(z_p - z_focus + n_oil*(-z_p/n_medium - t_glass/n_glass + t_glass0/n_glass0 + t_oil0/n_oil0)) +
        z_p*np.sqrt(n_medium**2 - n_oil**2 *np.sin(angles)**2) + t_glass*np.sqrt(n_glass**2 - n_oil**2 *np.sin(angles)**2) -
        t_glass0*np.sqrt(n_glass0**2 - n_oil**2 *np.sin(angles)**2) - t_oil0*np.sqrt(n_oil0**2 - n_oil**2 *np.sin(angles)**2))
    
    return opd + n_medium*z_p + n_glass*t_glass + n_oil*t_oil - n_glass0*t_glass0 - n_oil0*t_oil0

def richards_wolf(field, z_ang, xy_ang, capture_angle, x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan(y/x)
    xy_angles, z_angles = np.meshgrid(xy_ang, z_ang)
    return -1j/wavelen*np.sum(np.sum(
        field*
        np.exp(1j*k*r*n_air*np.sin(z_angles)*np.cos(xy_angles - phi))* # xy propagation phase
        np.exp(-1j*k*n_air*z_focus*np.cos(z_angles))* # z_focus propagation phase
        np.exp(1j*k*opd_correction(z_angles))* # opd phase
        np.sqrt(np.cos(z_angles))*np.sin(z_angles), # spherical projection factors
        axis=1)*np.diff(z_ang[:2]), axis=1)*np.diff(xy_ang[:2])

def snells_law(n1, n2, incident_angle):
    transmitted_angle = np.arcsin((n1 / n2 * np.sin(incident_angle)))
    return transmitted_angle

def refract(n1, n2, incident_field, incident_angles):
    xyangles, incident_angle = np.meshgrid(xy_angles, incident_angles)
    
    transmitted_angle = snells_law(n1, n2, incident_angle)
    incident_vectors = np.array([np.sin(incident_angle) * np.cos(xyangles), np.sin(incident_angle) * np.sin(xyangles), np.cos(incident_angle)])

    # Decompose incident field in s and p polarization
    s_polarization = np.array([incident_vectors[1], -incident_vectors[0], np.zeros_like(incident_vectors[2])])
    #s_polarization = np.cross(incident_vectors.T, surface_normal).T
    norm = np.sin(incident_angle)
    s_polarization[:,norm != 0] /= norm[norm != 0]

    # dot product
    E_s = np.sum(incident_field*s_polarization, axis=0)
    #print(E_s)
    E_s_pol = E_s*s_polarization
    E_p_pol = incident_field - E_s_pol
    E_p = np.linalg.norm(E_p_pol, axis=0)
    #print(E_p)
    p_polarization = E_p_pol/E_p

    # Fresnel transmission coefficients
    t_s = 2*n1*np.cos(incident_angle)/(n1*np.cos(incident_angle) + n2*np.cos(transmitted_angle))
    t_p = 2*n1*np.cos(incident_angle)/(n2*np.cos(incident_angle) + n1*np.cos(transmitted_angle))
    E_s *= t_s
    E_p *= t_p
    transmittance = E_s*s_polarization + E_p*p_polarization
    return incident_field*transmittance, snells_law(n1, n2, incident_angles)


#%%


# Mie scattering

xy_angles = np.linspace(0, 2, 100)*np.pi
capture_angle = np.arcsin(min(NA/n_medium, 1))
scatter_angles = np.linspace(0, capture_angle, 100)
# scatterring angle 0 is front scattering, but I want to look at backscattering so pi is added
#intensity = miepython.i_par(m, x, np.cos(scatter_angles + np.pi))

# Field first polarization, second axis zangle, third xyangle
#scatter_field = np.sqrt(intensity)[np.newaxis,:,np.newaxis]
scatter_field = 1

#%% 

# Excitation and scatter field magnitude
# Reference field to excitation field
r_mg = (n_glass - n_medium)/(n_glass + n_medium)
t_gm = 2*n_glass/(n_glass + n_medium)
t_go = 2*n_glass/(n_glass + n_oil)

E_reference = polarization[:,np.newaxis,np.newaxis] # x_polarization with size 1
E_excitation = E_reference/t_go/r_mg*t_go
scatter_field = E_excitation*scatter_field


collection_efficiency = capture_angle/np.pi
polarizability = 4*np.pi*a**3*(e_scat-e_medium)/(e_scat + 2*e_medium)
scatter_phase = np.angle(polarizability)
scatter_cross_section = k**4/6/np.pi *np.abs(polarizability)**2
scatter_field *= np.sqrt(scatter_cross_section)*collection_efficiency

#%%

# Propagation
scatter_field = scatter_field*np.exp(1j*scatter_phase)
glass_field, glass_angles = refract(n_medium, n_glass, scatter_field, scatter_angles)
aperture_field, aperture_angles = refract(n_glass, n_oil, glass_field, glass_angles)

#%% Simulation

camsize = 2 # micron
#sampling points for detector
xs = np.linspace(-camsize/2, camsize/2, 64)*10**-6
ys = np.linspace(-camsize/2, camsize/2, 64)*10**-6

capture_angle_aperture = np.arcsin(min(NA/n_oil, 1))

#%%
z_focuss = np.linspace(3, 5, 5)*10**-6 #micron
z_ps = np.linspace(2.9, 3.1, 5)*10**-6


ff_intensities = np.zeros((len(z_focuss), len(z_ps), len(xs), len(ys)))
for m, z_focus in enumerate(z_focuss):
    for n, z_p in enumerate(z_ps):
        detector_field = np.zeros((3, len(xs), len(ys)), dtype=np.complex128)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                detector_field[:,i,j] = richards_wolf(aperture_field, aperture_angles, xy_angles, capture_angle_aperture, x, y)


        # Normalize scattered signal
        detector_field /= np.max(np.linalg.norm(detector_field, axis=0))
        reference_field = np.ones_like(detector_field)
        # get the initial field back
        reference_field *= E_reference
        interference_intensity = np.real(np.sum((detector_field*np.conj(reference_field)), axis=0) + np.sum((np.conj(detector_field)*reference_field), axis=0))
        ff_intensities[m, n] = interference_intensity

#field = detector_field + reference_field
#intensity = c*n_air*epsilon_0/2* np.sum(np.abs(field)**2, axis=0)
#scat_intensity = c*n_air*epsilon_0/2* np.sum(np.abs(detector_field)**2, axis=0)
#reference_intensity = c*n_air*epsilon_0/2* np.sum(np.abs(reference_field)**2, axis=0)
#intensity = np.sum(np.abs(field)**2, axis=0)

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

#%%

# 1x5 fixed focus, variable particle position
z_pp = np.array([0.27, 0.82, 1.5, 2.15, 3.12])*10**-6 #micron
z_focus = 1.5*10**-6


var_focus_intensities = np.zeros((len(z_pp), len(xs), len(ys)))
for m, z_p in enumerate(z_pp):
    detector_field = np.zeros((3, len(xs), len(ys)), dtype=np.complex128)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            detector_field[:,i,j] = richards_wolf(aperture_field, aperture_angles, xy_angles, capture_angle_aperture, x, y)


    # Normalize scattered signal
    detector_field /= np.max(np.linalg.norm(detector_field, axis=0))
    reference_field = np.ones_like(detector_field)
    # get the initial field back
    reference_field *= E_reference
    interference_intensity = np.real(np.sum((detector_field*np.conj(reference_field)), axis=0) + np.sum((np.conj(detector_field)*reference_field), axis=0))
    var_focus_intensities[m] = interference_intensity


#%%

# 1x5 plot
fig, ax = plt.subplots(5)
for i in range(5):
    ax[i].imshow(var_focus_intensities[i].T, cmap = 'gray')
    ax[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout()
plt.show()

#%%
# single image plot

z_p = 1.5*10**-6
z_focus = 1.5*10**-6
detector_field = np.zeros((3, len(xs), len(ys)), dtype=np.complex128)
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        detector_field[:,i,j] = richards_wolf(aperture_field, aperture_angles, xy_angles, capture_angle_aperture, x, y)

detector_field /= np.max(np.linalg.norm(detector_field, axis=0))
reference_field = np.ones_like(detector_field)
# get the initial field back
reference_field *= E_reference
interference_intensity = np.real(np.sum((detector_field*np.conj(reference_field)), axis=0) + np.sum((np.conj(detector_field)*reference_field), axis=0))
scat_intensity = c*n_air*epsilon_0/2* np.sum(np.abs(detector_field)**2, axis=0)

#%%
fig, ax = plt.subplots(1, 2)
p1 = ax[0].imshow(scat_intensity.T, extent=[-camsize, camsize, -camsize, camsize], cmap = 'gray')
fig.colorbar(p1, ax=ax[0])
ax[0].set_xlabel('x position(micron)')
ax[0].set_ylabel('y position(micron)')
ax[0].set_title('Scattering intensity')
p2 = ax[1].imshow(interference_intensity.T, extent=[-camsize, camsize, -camsize, camsize], cmap = 'gray')
ax[1].set_xlabel('x position(micron)')
ax[1].set_ylabel('y position(micron)')
fig.colorbar(p2, ax=ax[1])
ax[1].set_title('Total intensity')
plt.show()