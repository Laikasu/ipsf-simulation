from model import *
import numpy as np


# Plotting
# from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import json
import os

# For file dialogs
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()


# Contrast plot

diameters = np.linspace(2, 80, 100)
wavelength = 520

params = DesignParams(
    wavelen = wavelength,
    n_medium = n_water)
    


contrast = np.zeros_like(diameters)
scatter_intensity = np.zeros_like(diameters)
interference_intensity = np.zeros_like(diameters)

# 1 pixel camera
params.roi_size = params.pxsize/params.magnification
camera = Camera(params)

for i, diameter in enumerate(diameters):
    params.diameter = diameter

    scatter_field = calculate_scatter_field(params)
    
    intensity = calculate_intensities(scatter_field, params, camera, r_resolution=2)
    scatter_intensity[i] = intensity['scat'][0,0]
    interference_intensity[i] = intensity['if'][0,0]
    contrast[i] = intensity['sig'][0,0]

# Contrast plot plot
fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'hspace': 0})
ax[0].set_yscale('symlog', linthresh=0.1)

ax[0].plot(diameters[contrast>0], contrast[contrast>0], label='total contrast')
ax[0].set_ylim(0, np.max(scatter_intensity))
ax[0].plot(diameters[scatter_intensity>0], scatter_intensity[scatter_intensity>0], label='scatter contrast', color='red')
ax[0].plot(diameters[interference_intensity>0], interference_intensity[interference_intensity>0], label='interference contrast', color='green')
ax[1].plot(diameters[contrast<0], contrast[contrast<0])
ax[1].plot(diameters[scatter_intensity<0], scatter_intensity[scatter_intensity<0])
ax[1].plot(diameters[interference_intensity<0], interference_intensity[interference_intensity<0])
ax[0].legend()
ax[1].grid()
ax[1].set_ylim(np.min(contrast)*1.2, 0)
ax[1].set_xlabel('GNP diameter(nm)')
ax[0].grid()
ax[0].set_ylabel('Contrast')
ax[0].set_title('GNP size and contrast')
plt.show()