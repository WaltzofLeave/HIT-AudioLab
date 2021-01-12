import numpy as np
import matplotlib.pyplot as plt
import wave
from get import get
def naive_encode(num:int):
    if num > 127:
        return 127
    elif num < -128:
        return -128
    else:
        return num
def factor_encode(num:int,a:float=5,bits=4):
    x = 2 ** (bits-1)
    if num > (x-1)*a:
        return (x-1)
    elif num < -x * a:
        return -x
    else:
        return num//a + 1
def naive_DPCM(wave_data:np.ndarray):
    x = wave_data
    x_hat = np.zeros(wave_data.shape)
    d = np.zeros(wave_data.shape)
    d[0] = x[0]
    x_hat[0] = wave_data[0]
    c = np.zeros(wave_data.shape)
    c[0] = naive_encode(d[0])
    for i in range(1,wave_data.shape[0]):
        d[i] = x[i] - x_hat[i-1]
        c[i] = naive_encode(d[i])
        x_hat[i] = x_hat[i-1] + c[i]
    return c
def factor_DPCM(wave_data:np.ndarray,a:float,bits:int=4):
    x = wave_data
    x_hat = np.zeros(wave_data.shape)
    d = np.zeros(wave_data.shape)
    d[0] = x[0]
    x_hat[0] = wave_data[0]
    c = np.zeros(wave_data.shape)
    c[0] = factor_encode(d[0],a,bits)
    for i in range(1, wave_data.shape[0]):
        d[i] = x[i] - x_hat[i - 1]
        c[i] = factor_encode(d[i],a,bits)
        x_hat[i] = x_hat[i - 1] + (c[i]-1)*a
    return c
def naive_decode_DPCM(c:np.ndarray):
    x_hat = np.zeros(c.shape)
    x_hat[0] = c[0]
    for i in range(1,c.shape[0]):
        x_hat[i] = x_hat[i-1] + c[i]
    return x_hat

def factor_decode_DPCM(c:np.ndarray,a:int,bits:int=4):
    x = 2 ** (bits - 1)
    x_hat = np.zeros(c.shape)
    x_hat[0] = c[0]
    for i in range(1, c.shape[0]):
        x_hat[i] = x_hat[i - 1] + (c[i]-1)*a
    return x_hat
def save_data(filename:str,code:np.ndarray,bits:int):
    mask ='0b'+ bits * '1'
    mask = int(mask,2)
    l = []
    for item in code:
        item = int(item)
        item = item & mask
        tmp = item
        for i in range(0,bits):
            l.append(tmp & 0b1)
            tmp = tmp >> 1
    while len(l) % 8 != 0:
        l.append(0)
    with open(filename,'wb') as f:
        for i in range(0,len(l)//8):
            x = '0b'
            for j in range(0,8):
                x = x + str((l[i*8+j]))
            x = int(x,2)
            x = [x]
            x = bytes(x)
            f.write(x)
    #print(l)
def load_data(filename,bits):
    l = []
    with open(filename,'rb') as f:
        while True:
            s = f.read(1)
            if not s:
                break
            #print(s)
            s = int.from_bytes(s, byteorder='little', signed=True)
            for i in range(0,8):
                l.append((s>>7)&1)
                s = s << 1
    l = np.array(l)
    l = l.reshape((-1,bits))
    ans = []
    def calculate(item):
        ans = 0
        for i in range(0,len(item)-1):
            ans = ans + item[i] * (2 ** i)
        ans = ans - item[len(item)-1] *(2 ** (len(item)-1))
        return ans
    for item in l:
        a = calculate(item)
        ans.append(a)
    return np.array(ans)

def SNR(orig:np.ndarray,new:np.ndarray):
    size = len(orig)
    up = 0
    down = 0
    for i in range(0,size):
        up += orig[i] ** 2
        down += (orig[i] - new[i]) ** 2
    if down == 0:
        return np.inf
    return 10 * np.log10(up/down)
def wav_to_pcm(filename,pcmfilename):
    f = open(filename,'rb')
    f.seek(0)
    f.read(44)
    data = np.fromfile(f,dtype=np.int16)
    data.tofile(pcmfilename)
def writeaspcm(filename:str,wave_data:np.ndarray):
    """
    filename should be a string without suffix
    :param filename:
    :param wave_data:
    :return:
    """
    with wave.open(filename+'.wav', 'wb') as f:
        data_to_write = wave_data.flatten()
        data_to_write = np.array(data_to_write, dtype=np.dtype(np.int16))
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.setcomptype('NONE', 'Uncompressed')
        f.writeframes(data_to_write)
    wav_to_pcm(filename+'.wav',filename+'.pcm')
def implement1():
    wave_data = get(1,False)
    y = naive_DPCM(wave_data)
    save_data('1_8bit.dpc',y,8)
    data = load_data('1_8bit.dpc',8)
    z = naive_decode_DPCM(data)
    print('SNR: '+str(SNR(wave_data,z)))
    writeaspcm('1_8bit',z)
def implement2():
    wave_data = get(1,False)
    y = naive_DPCM(wave_data)
    save_data('1_4bit.dpc',y,4)
    data = load_data('1_4bit.dpc',4)
    z = naive_decode_DPCM(data)
    print('SNR: '+str(SNR(wave_data,z)))
    writeaspcm('1_4bit',z)
def findarguments():
    wave_data = get(1, False)
    wave_data = np.array(wave_data,dtype=np.int16)
    SNRmax = 0
    amax = 1
    for a in range(570,600):
        y = factor_DPCM(wave_data,a)
        z = factor_decode_DPCM(y,a)
        snr = SNR(wave_data,z)
        if snr > SNRmax:
            SNRmax = snr
            amax = a
    print("The best a :" + str(amax))
    y = factor_DPCM(wave_data,amax,4)
    save_data('1_4bit.dpc', y, 4)
    data = load_data('1_4bit.dpc', 4)
    z = factor_decode_DPCM(data,amax,4)
    print('SNR: ' + str(SNR(wave_data, z)))
    writeaspcm('1_4bit', z)
findarguments()