import numpy as np
from get import get
from energy import energy
from zrate import zrate
def lab1_main():
    for i in range(1,11):
        wave_data = get(i)
        wave_energy = energy(wave_data)
        wave_zrate = zrate(wave_data)
        with open('../result/lab1/'+str(i)+'_en.txt','w') as f:
            for num in wave_energy:
                f.write(str(num) + '\n')
        with open('../result/lab1/'+str(i)+'_zero.txt','w') as f:
            for num in wave_zrate:
                f.write(str(num) + '\n')
        return wave_data
Running = True
if Running:
    wave_data = lab1_main()
    print("Wave_data")
    print(wave_data)
