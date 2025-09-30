import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt


params = DesignParams(scat_mat="gold", diameter=40)


num = 100
bound = np.zeros(num)
total = np.zeros(num)
wavelens = np.linspace(300, 800, num)

# Free electrons only
params.gold_model = 'bound'
for i, wavelen in enumerate(wavelens):
    params.wavelen = wavelen
    scatter_field = calculate_scatter_field(params)
    bound[i] = np.abs(scatter_field[0])

# Experimental data (free + bound)
params.gold_model = 'experiment'
for i, wavelen in enumerate(wavelens):
    params.wavelen = wavelen
    scatter_field = calculate_scatter_field(params)
    total[i] = np.abs(scatter_field[0])


free = total - bound

fig, ax = plt.subplots(1, figsize=(10,5))
ax.plot(wavelens, free, label=f'free electrons')
ax.plot(wavelens, bound, label=f'bound electrons')
ax.plot(wavelens, total, label=f'total (experiment)')
ax.set_ylim(0, 1.1*max(total))


fig.suptitle('40nm AuNS spectrum contributions')
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('scatter strength (au)')
ax.legend()
plt.show()