import numpy as np

def wav_to_pcm(filename,pcmfilename):
    f = open(filename,'rb')
    f.seek(0)
    f.read(44)
    data = np.fromfile(f,dtype=np.int16)
    data.tofile(pcmfilename)

def convert():
    for i in range(1,11):
        filename = '../result/lab1_2/'+str(i)+".wav"
        pcmfilename = '../result/pcm/'+str(i)+'.pcm'
        wav_to_pcm(filename,pcmfilename)
convert()