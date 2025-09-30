import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt


params = DesignParams(scat_mat="gold", diameter=40)


num = 100
drude_lorenz = np.zeros(num, dtype=np.complex128)
drude = np.zeros(num, dtype=np.complex128)
measurement = np.zeros(num, dtype=np.complex128)
wavelens = np.linspace(252, 1800, num)
for i, wavelen in enumerate(wavelens):
    measurement[i] = n_gold(wavelen)
    drude[i] = drude_gold(wavelen)
    drude_lorenz[i] = drude_lorenz_gold(wavelen,-1)

fig, ax = plt.subplots(1,2, sharex=True, figsize=(10,5))
ax[0].plot(wavelens, np.real(drude), label=f'free electrons')
ax[0].plot(wavelens, np.real(measurement - drude), label=f'bound')
ax[1].plot(wavelens, np.abs(np.imag(drude)), label=f'free electrons')
ax[1].plot(wavelens, np.abs(np.imag(drude_lorenz)) - np.abs(np.imag(drude)), label=f'bound electrons')


fig.suptitle('40nm AuNS plasmon resonances')
ax[0].set_title('real component')
ax[0].set_xlabel('wavelength (nm)')
ax[0].set_ylabel('n')
ax[0].legend()

ax[1].set_title('imaginary component')
ax[1].set_ylabel('k')
ax[1].set_xlabel('wavelength (nm)')
ax[1].legend()
plt.show()