import numpy as np
import matplotlib.pyplot as plt
from get import get


def get_labels(i):
    with open('../label/label_'+str(i)+'.txt') as f:
        labelstr = f.readline()
        #print(labelstr)
        labelstr = labelstr.strip('[')
        labelstr = labelstr.strip(']')
        labelstr = labelstr.strip(' ')
        #print(labelstr)
        labelarray = labelstr.split(',')
        #print(labelarray)
        labels = np.array(labelarray,dtype='f4')
        #print(labels)
        return labels
def get_dataset():
    dataset = None
    for i in range(1,10):
        wave_data = get(i)
        wave_label = get_labels(i)
        wave_label = np.expand_dims(wave_label,axis=0)
        wave_label = wave_label.T
        waves = np.hstack((wave_data,wave_label))
        print(waves.shape)
        if dataset is None:
            dataset = waves
        else:
            dataset = np.vstack((dataset,waves))
    return dataset
testing = True
if testing:
   dataset = get_dataset()
   print(dataset[0:50])
   print(dataset.shape)

