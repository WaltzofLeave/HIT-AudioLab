import numpy as np
import matplotlib as plt
from get import get

def energy(wave_data:np.ndarray)->np.ndarray:
    if len(wave_data.shape) == 1:
        wave_data = np.expand_dims(wave_data,axis=0)
    nframe = wave_data.shape[1]
    w = 1
    energy = []
    for x in wave_data:
        energy.append(np.sum((x**2)*((w**2)),axis=0))
    energy = np.array(energy)
    return energy

testing = True
if testing:
    x = [-1,-1,-1,1]
    y = [[1,2,3],[4,5,6]]
    x = np.array(x)
    y = np.array(y)
    print(energy(x))
    print(energy(y))

