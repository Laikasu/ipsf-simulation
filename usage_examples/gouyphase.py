import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt

def gouy(p):
    z_p = p.z_p*10**-6
    wavelen = p.wavelen*10**-9
    z_focus = p.z_focus*10**-6
    NA = p.NA
    z = z_p - z_focus
    w0 = 0.61*wavelen/NA
    z_R = np.pi*w0**2/wavelen
    # gouy phase for beam traveling in negative z caries a sign
    return np.arctan(-z/z_R)

z_focus = np.linspace(-1, 1, 401)
phases = []
intens_if = []
intens = []
gouy_phases = []

params = DesignParams(scat_mat="gold", diameter=50)
scatter_field = calculate_scatter_field(params)
polarization = scatter_field.real/np.linalg.norm(scatter_field.real)*np.sign(scatter_field.real[0])
scatter_field_scalar = np.vdot(scatter_field, polarization)
scatter_phase = np.angle(scatter_field_scalar)

for z_f in z_focus:
    params.z_focus = z_f
    gouy_phases.append(gouy(params))
    # 1 pixel camera
    params.roi_size = params.pxsize/params.magnification
    camera = Camera(params)
    intensities = calculate_intensities(scatter_field, params, camera, r_resolution=2)
    detector_field = intensities['fields'][0]
    polarization = detector_field.real/np.linalg.norm(detector_field.real)*np.sign(detector_field.real[0])
    phase = np.angle(np.vdot(detector_field, polarization))
    magnitude = np.abs(np.vdot(detector_field, polarization))
    intens_if.append(np.sum(intensities['if']))
    intens.append(magnitude*2)
    phases.append(phase)

fig, ax = plt.subplots(2,1, sharex=True)

# unwrap phases
phases = np.array(phases)
diff = np.diff(phases)
idx = np.where(np.abs(diff) > 6)[0]
for i in idx:
    phases[i+1:] += -np.sign(diff[i])*2*np.pi
# set zero at focus
phases = phases - phases[len(phases)//2] + scatter_phase

ax[0].plot(z_focus, np.degrees(phases), label='model phase')
ax[0].plot(z_focus, np.degrees(gouy_phases+scatter_phase), label='gouy phase')
ax[0].axhline(np.degrees(scatter_phase), color='k', label='scatter phase', linewidth=1)
ax[0].legend()
#ax[0].set_yscale('log')
ax[0].set_xlabel('z_focus')
ax[0].set_ylabel(r'phase ($\degree$)')
ax[1].plot(z_focus, intens_if, label='model intensity')

ax[1].plot(z_focus, intens, label='magnitude')
ax[1].set_xlabel('z_focus')
ax[1].set_ylabel('intensity (au)')
ax[1].legend()
plt.show()