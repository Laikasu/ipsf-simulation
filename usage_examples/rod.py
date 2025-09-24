import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt



params = DesignParams(scat_mat="gold", diameter=40)
print(params.gold_model)

num = 100
intensities = []
wavelens = np.linspace(450, 650, num)
for model in ['experiment', 'drude-lorenz']:
    params.gold_model = model
    intensity = np.zeros(num, dtype=np.complex128)
    for i, wavelen in enumerate(wavelens):
        params.wavelen = wavelen
        scatter_field = calculate_scatter_field(params)
        intensity[i] = np.sum(np.abs(scatter_field))
    
    intensities.append(intensity)

fig, ax = plt.subplots(1,2, sharex=True)
ax[0].plot(wavelens, intensities[0], label=f'measurement')
ax[0].plot(wavelens, intensities[1], label=f'measurement')


fig.suptitle('40nm AuNS plasmon resonances')
ax[0].set_title('scatter amplitude')
ax[0].set_xlabel('wavelength (nm)')
ax[0].set_ylabel('field strength (au)')
ax[0].legend()

ax[1].set_title('scatter phase')
ax[1].set_ylabel(r'phase ($\degree$)')
ax[1].set_xlabel('wavelength (nm)')
plt.show()