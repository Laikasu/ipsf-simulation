from model import *
import numpy as np

# Plotting
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt

# Single plot
intensity = {}

for pol, azimuth, inclination in zip(('x', 'y', 'z'), (0, 90, 0), (0, 0, 90)):
    params = DesignParams(azimuth=azimuth, inclination=inclination, magnification=80, roi_size=1.4)
    camera = Camera(params)
    scatter_field = calculate_scatter_field(params)
    intensity[pol] = calculate_intensities(scatter_field, params, camera)['scat']


fig, ax = plt.subplots(1, 3, figsize=(12,4))

for i, intens in enumerate(['x', 'y', 'z']):
    p = ax[i].imshow(intensity[intens])
    ax[i].set_axis_off()
    scalebar = ScaleBar(params.pxsize)
    ax[i].add_artist(scalebar)
    fig.colorbar(p, ax=ax[i])
    ax[i].set_title(intens + ' polarization')

plt.show()