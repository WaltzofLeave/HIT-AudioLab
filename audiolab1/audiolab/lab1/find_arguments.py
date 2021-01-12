from double_thresh import double_thresh
import numpy as np
import matplotlib.pyplot as plt
from get import get

def onetest():
    MH = 1e6 * np.random.rand() + 1e6
    ML = 1e20
    while ML > MH:
        ML = 1e6 * np.random.rand() + 1e6
    Zs = np.random.rand()*0.1
    nframe = int(np.random.rand()*80+3)
    costsum = 0
    frames = 0
    for i in range(1,10):
        wave_data_i = get(i)
        ansi,ans01i = double_thresh(wave_data_i,MH,ML,Zs,nframe,i)
        with open('../label/label_'+str(i)+'.txt')as f:
            labelstr = f.readline()
            labelstr = labelstr.strip('[')
            labelstr = labelstr.strip(']')
            labelstr = labelstr.strip()
            labelstr = labelstr.split(',')
            labelstr = np.array(labelstr,dtype='f4')
        costsum += np.sum(np.abs(labelstr - ans01i))
        frames += wave_data_i.shape[0]
    accuracy = ((frames-costsum)/frames)
    #print(costsum)
    #print(str(accuracy*100) + '%')
    return MH,ML,Zs,nframe,costsum,accuracy
def writelog(log):
    with open('../result/log.txt','a') as f:
        f.write(str(log))
highest = 0
while True:
    Mh,Ml,Zs,nframe,costsum,accuracy = onetest()
    if accuracy > highest:
        highest = accuracy
        writelog([Mh,Ml,Zs,nframe,costsum,accuracy])
