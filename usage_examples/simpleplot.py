import numpy as np
import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

# Plotting
#from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt

# Single plot

wavelength = 525
diameter = 100*10**-9
radius = diameter/2

params = DesignParams(
    azimuth=90,
    #unpolarized=True,
    z_p = 0,
    z_focus = 0,
    wavelen = wavelength,
    scat_mat="gold",
    n_medium = n_water,
    a = radius,
    roi_size=1,
    magnification=100)

camera = Camera(params)

scatter_field = calculate_scatter_field(params)
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
