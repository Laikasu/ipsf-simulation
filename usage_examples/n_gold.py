import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt


params = DesignParams(scat_mat="gold", diameter=40)


num = 100
drude_lorenz = np.zeros(num, dtype=np.complex128)
measurement = np.zeros(num, dtype=np.complex128)
wavelens = np.linspace(252, 700, num)
for i, wavelen in enumerate(wavelens):
    measurement[i] = n_gold(wavelen)
    drude_lorenz[i] = np.sqrt(drude_lorenz_gold(wavelen))
    # params.wavelen = wavelen
    # scatter_field = calculate_scatter_field(params)
    # measurement[i] = np.sum(np.abs(scatter_field))
    # drude_lorenz[i] = np.degrees(np.angle(scatter_field[0]))

fig, ax = plt.subplots(1,2, sharex=True)
ax[0].plot(wavelens, np.real(measurement), label=f'measurement')
ax[0].plot(wavelens, np.real(drude_lorenz), label=f'drude')
ax[1].plot(wavelens, np.imag(measurement), label=f'measurement')
ax[1].plot(wavelens, -np.imag(drude_lorenz), label=f'drude')
#ax[2].plot(media, camera_intensities, label=f'n_medium={n:.2f}')


fig.suptitle('40nm AuNS plasmon resonances')
ax[0].set_title('scatter amplitude')
ax[0].set_xlabel('wavelength (nm)')
ax[0].set_ylabel('field strength (au)')
ax[0].legend()

ax[1].set_title('scatter phase')
ax[1].set_ylabel(r'phase ($\degree$)')
ax[1].set_xlabel('wavelength (nm)')
plt.show()