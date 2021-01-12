from energy import energy
from zrate import zrate
import numpy as np
from get import get
import matplotlib.pyplot as plt
def save(filename:str,wave_data:np.ndarray,left:int,right:int)->None:
    pass
def double_thresh(wave_data:np.ndarray,Mh:float,Ml:float,Zs:float,highframesthre=20,filenum:int=0)->list:
    wave_energy = energy(wave_data)
    wave_zrate = zrate(wave_data)
    labels = np.zeros(wave_energy.shape)
    find = []
    ans = []
    row = wave_data.shape[0]
    ans01 = np.zeros(row)
    i = 0
    counter = -1
    while i < len(wave_energy):
        if wave_energy[i] >= Mh:
            start_high_pos = i
            while wave_energy[i] >= Mh and i < row -1:
                i += 1
            end_high_pos = i
            if end_high_pos - start_high_pos >= highframesthre:
                start_low_pos = start_high_pos
                end_low_pos = end_high_pos
                while wave_energy[end_low_pos] >= Ml and end_low_pos < row -1:
                    end_low_pos += 1
                while wave_energy[start_low_pos] >= Ml and end_low_pos < row -1:
                    if start_low_pos not in find:
                        start_low_pos -= 1
                    else:
                        break
                start = start_low_pos
                while wave_zrate[start] <= Zs and start > 0:
                    if start not in find:
                        start = start - 1
                    else:
                        break
                # [start+1,end_low_pos)
                for k in range(start+1,end_low_pos):
                    labels[k] = 1   #语音
                    find.append(k)
                counter += 1
                #save('result/lab1_2/'+str(filenum)+'_'+str(counter)+'.pcm',wave_data,start+1,end_low_pos)
                ans.append((start+1,end_low_pos-1))
                for kk in range(start+1,end_low_pos):
                    ans01[kk] = 1
                i = end_low_pos
            else:
                i = start_high_pos + 1
        else:
            i += 1
    ans01 = np.array(ans01)
    return ans,ans01
def double_thresh_test(Mh:float,Ml:float,Zs:float,highframesthre=20,filenum:int=0)->list:
    wave_energy = np.array([1,2,5,5,3,1,0,1,2,3,4,5,6,5,4,3,2,1,1,0])
    wave_zrate =  np.array([2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,2,2,2,2])
    labels = np.zeros(wave_energy.shape)
    find = []
    ans = []
    row = wave_energy.shape
    ans01 = np.zeros(row)
    i = 0
    counter = -1
    while i < len(wave_energy):
        if wave_energy[i] >= Mh:
            start_high_pos = i
            while wave_energy[i] >= Mh:
                i += 1
            end_high_pos = i
            if end_high_pos - start_high_pos >= highframesthre:
                start_low_pos = start_high_pos
                end_low_pos = end_high_pos
                while wave_energy[end_low_pos] >= Ml:
                    end_low_pos += 1
                while wave_energy[start_low_pos] >= Ml:
                    if start_low_pos not in find:
                        start_low_pos -= 1
                    else:
                        break
                start = start_low_pos
                while wave_zrate[start] <= Zs:
                    if start not in find:
                        start = start - 1
                    else:
                        break
                # [start+1,end_low_pos)
                for k in range(start+1,end_low_pos):
                    labels[k] = 1   #语音
                    find.append(k)
                counter += 1
                #save('result/lab1_2/'+str(filenum)+'_'+str(counter)+'.pcm',wave_data,start+1,end_low_pos)
                ans.append((start+1,end_low_pos-1))
                for kk in range(start + 1, end_low_pos):
                    ans01[kk] = 1
                i = end_low_pos
            else:
                i = start_high_pos + 1
        else:
            i += 1
    ans01 = np.array(ans01)
    return ans,ans01
ans ,ans01= double_thresh_test(4,2,1.5,3)
print("++++++++=====++++++")
print(ans)
print(ans01)
print("++++++++=====++++++")
