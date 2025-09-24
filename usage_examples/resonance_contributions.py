import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt



params = DesignParams(scat_mat="gold", diameter=40)

num = 100
scatter_phase = np.zeros(num, dtype=np.complex128)
intensity = np.zeros(num, dtype=np.complex128)
wavelens = np.linspace(200, 650, num)
for i, wavelen in enumerate(wavelens):
    params.wavelen = wavelen
    scatter_field = calculate_scatter_field(params)
    intensity[i] = np.sum(np.abs(scatter_field))
    # scatter_phase[i] = np.degrees(np.angle(scatter_field[0]))

fig, ax = plt.subplots(1,2, sharex=True)
ax[0].plot(wavelens, intensity, label=f'measurement')
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