import numpy as np


import sys, os
from model import *

import matplotlib.pyplot as plt


params = DesignParams(
    scat_mat="gold",
    diameter=80,
    n_medium=1)


n_oils = np.linspace(params.t_glass0, params.t_glass0 + 0.1, 5)
oil_intensities = []
for n in n_oils:
    params.t_glass = n
    num = 100
    wavelens = np.linspace(480, 560, num)
    intensity = np.zeros(num)
    for i, wavelen in enumerate(wavelens):
        params.wavelen = wavelen
        scatter_field = calculate_scatter_field(params)
        # 1 pixel camera
        params.roi_size = params.pxsize/params.magnification
        camera = Camera(params)
        intensities = calculate_intensities(scatter_field, params, camera, r_resolution=2)
        intensity[i] = intensities['if'][0,0]
    oil_intensities.append(intensity)

fig, ax = plt.subplots(1,1)
for intensity, n in zip(oil_intensities, n_oils):
    ax.plot(wavelens, intensity, label=f'n_oil = {n:.5f}')
ax.set_title('Effect of aberrations (n_oil0 = 1.5)')
ax.legend()
ax.set_ylabel('contrast')

plt.show()