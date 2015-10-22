from random import random
import numpy as np
from matplotlib.pylab import *



def generateData(n):
    """
    generates a 2D linearly separable dataset with n samples.
    The thired element of the sample is the label
    """
    xb = (np.random.rand(n) * 2 -1) / 2 - 0.5
    yb = (np.random.rand(n) * 2 -1) / 2 + 0.5
    xr = (np.random.rand(n) * 2 -1) / 2 + 0.5
    yr = (np.random.rand(n) * 2 -1) / 2 - 0.5
    inputs = []
    inputs.extend([[xb[i], yb[i], 1] for i in range(n)])
    inputs.extend([[xr[i], yr[i], -1] for i in range(n)])
    return inputs

print(generateData(10))
