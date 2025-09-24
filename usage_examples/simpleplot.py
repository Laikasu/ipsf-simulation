import numpy as np
import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

# Plotting
#from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt

# Single plot

params = DesignParams(
    azimuth=0,
    inclination=90,
    wavelen = 525,
    scat_mat="gold",
    n_medium = n_water,
    diameter=100,
    roi_size=2,
    magnification=100)

camera = Camera(params)
scatter_field = calculate_scatter_field(params, multipolar=False)
intensity = calculate_intensities(scatter_field, params, camera)

plt.imsave('colormap_image.png', intensity['scat'], cmap='viridis')
fig, ax = plt.subplots()
p = ax.imshow(intensity['scat'])
ax.set_axis_off()
#fig.colorbar(p)
#ax.add_artist(ScaleBar(camera.pxsize_obj))
plt.show()



# 3x1 plot
# fig, ax = plt.subplots(1, 3, figsize=(12,4))

# for i, intens in enumerate(['if', 'scat', 'tot']):
#     p = ax[i].imshow(intensity[intens])
#     ax[i].set_axis_off()
#     scalebar = ScaleBar(pxsize)
#     ax[i].add_artist(scalebar)
#     fig.colorbar(p, ax=ax[i])
#     ax[i].set_title(intens + ' intensity')

# plt.show()
