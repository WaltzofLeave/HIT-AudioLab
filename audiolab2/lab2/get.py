import wave
import numpy as np
import matplotlib.pyplot as plt

def get(num:int=1,frame_sep:bool=True,plotting:bool=False)->np.ndarray:
    with wave.open('../sound/'+str(num)+'.wav') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, framenums = params[:4]
        data_in_str = f.readframes(framenums)
        wave_data = np.frombuffer(data_in_str, dtype=np.short)
    assert len(wave_data.shape) == 1
    if plotting is True:
        plt.plot(wave_data)
        plt.show()
    if frame_sep is True:
        diff = len(wave_data) % 256
        for i in range(0,256-diff):
            wave_data = np.append(wave_data,wave_data[-1])
        wave_data = np.resize(wave_data,(-1,256))
    wave_data = np.array(wave_data,dtype=np.float)
    return wave_data

testing = False
if testing:
    x = get(plotting=True)
    print(x)
    print(x.shape)