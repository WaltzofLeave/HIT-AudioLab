from lab1.get import get
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

def make_label(filenum):
    wave_data = get(filenum)
    saved_name = 'label_' + str(filenum) + '.txt'
    x = list(range(0, wave_data.shape[0] * wave_data.shape[1]))
    x = np.array(x)
    print(x[-1])
    x = np.reshape(x, (-1, 256))
    print(wave_data.shape)
    print(x.shape)
    print(x)
    flatten_data = wave_data.flatten()
    labels = []
    now = 0
    num = 10
    for i in range(0, wave_data.shape[0],10):
        plt.subplot(2, 1, 1)
        for j in range(0, i):
            plt.plot(x[j], wave_data[j], color='blue')
        mid = i + 10 if i+10 <= wave_data.shape[0] else wave_data.shape[0]
        for j in range(i,mid):
            plt.plot(x[j], wave_data[j], color='red')
        if mid < wave_data.shape[0]:
            for j in range(i +10, wave_data.shape[0]):
                plt.plot(x[j], wave_data[j], color='blue')
        plt.subplot(2,1,2)
        min = i - 30  if i - 30 >= 0 else 0
        max = i + 30  if i + 30 < wave_data.shape[0]  else wave_data.shape[0]
        for j in range(min, i):
            plt.plot(x[j], wave_data[j], color='blue')
        mid = i + 10 if i + 10 <= wave_data.shape[0] else wave_data.shape[0]
        flag = 0
        if not mid % 2 == 0:
            mid = mid - 1
            flag = 1
        for j in range(i,mid,2):
            plt.plot(x[j], wave_data[j], color='red')
            plt.plot(x[j+1], wave_data[j+1], color='green')
        if flag == 1:
            plt.plot(x[mid], wave_data[mid], color='red')
        if i + 10 < wave_data.shape[0]:
            for j in range(i + 10, max):
                plt.plot(x[j], wave_data[j], color='blue')

        plt.show()
        now = input()
        if now == '0':
            for k in range(0,10):
                labels.append(0)
        elif now == '1':
            for k in range(0,10):
                labels.append(1)
        else:
            valid = False
            tmparray = None
            s = None
            while not valid:
                s = input("please input detail:")
                tmparray = []
                valid = True
                for numstr in s:
                    if numstr == '0':
                        tmparray.append(0)
                    elif numstr == '1':
                        tmparray.append(1)
                    else:
                        print("Invalid")
                        valid = False
                if not len(tmparray) == 10:
                    valid = False
                    print("Not enough")
            for num in tmparray:
                labels.append(num)
    labels = labels[0:wave_data.shape[0]]
    with open(saved_name,'a') as f:
        f.write(str(labels))

for i in range(5,11):
    make_label(i)
