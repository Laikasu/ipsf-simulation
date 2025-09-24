import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from model import n_gold

res_p=8.45
def drude_lorenz_gold(wavelen, f, res_0, damping):
    # Constants from Improved Drude-Lorentz dielectric function for gold nanospheres
    # Anthony Centeno
    freq_eV = const.h*const.c/wavelen/10**-9/const.e
    return 1 + np.sum(f*res_p**2/(res_0**2-freq_eV**2-1j*freq_eV*damping))


f = np.array([0.845, 0.065, 0.124, 0.011, 0.840, 5.646])
res_0 = np.array([0.0, 0.816, 4.481, 8.185, 9.083, 20.29])
damping = np.array([0.072, 3.886, 0.452, 0.065, 0.916, 2.419])

# f = np.array([0.98, 0.1222, 0.2492, 2.7845, 0.1082, 7.8214])
# res_0 = np.array([0, 3.5684, 4.2132, 9.1499, 2.9144, 38.9633])
# damping = np.array([0.044113, 0.97329, 1.1139, 0.4637, 0.70308, 0.48978])

wavelen = np.linspace(300, 1000, 100)
p0 = (f, res_0, damping)
p, pcov = curve_fit(drude_lorenz_gold, wavelen, n_gold(wavelen), p0)
print(p)