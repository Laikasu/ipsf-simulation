import numpy as np
import sys, os
from model import *
import matplotlib.pyplot as plt


params = DesignParams(scat_mat="gold", diameter=40)


num = 100
spectra = []
ns = 1.333 + np.linspace(0, 0.1, 5)
wavelens = np.linspace(450, 600, num)
for n in ns:
    params.n_medium = n
    spectrum = np.zeros(num)
    for i, wavelen in enumerate(wavelens):
        params.wavelen = wavelen
        scatter_field = calculate_scatter_field(params)
        spectrum[i] = np.abs(scatter_field[0])
    spectra.append(spectrum)
    


fig, ax = plt.subplots(1)
for spectrum in spectra:
    norm = spectrum/np.max(spectrum)
    ax.plot(wavelens, norm)


fig.suptitle('40nm AuNS plasmon resonances')
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('scatter strength')
plt.show()