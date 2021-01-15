import struct
import numpy as np
import matplotlib.pyplot as plt

names = ['kaiji','zuozhuan','youzhuan','qianjin','houtui']

path = 'C:\\Users\\DELL\\Desktop\\temp\\'
origpath = path + 'orig\\'
destpath = path + 'pattern\\'
origs = []
dests = []
for name in names:
    origs.append(origpath + name + '.wav')
    dests.append(destpath + name + '.mfc')
    for i in range(1,11):
        origs.append(origpath + name + str(i) + '.wav')
        dests.append(destpath + name + str(i) + '.mfc')
s = len(origs)
with open('list.scp','w') as f:
    for i in range(0,s):
        f.write(origs[i] + ' ' + dests[i] + '\n')
        import struct
testfile = destpath + 'kaiji.mfc'
testfile1 = destpath + 'kaiji2.mfc'

def getint(a):
    return int.from_bytes(a , byteorder='big', signed=True)
def getfloat(a):
    return struct.unpack('>f',a)[0]
def getonedata(filename):
    with open(filename,'rb') as f:
        nframes = getint(f.read(4))
        frate  = getint(f.read(4))
        nbytes = getint(f.read(2))
        feakind = getint(f.read(2))
        num = nbytes // 4
        data = []
        for j in range(0,nframes):
            tmp_data = []
            for i in range(0,num):
                tmp_data.append(getfloat(f.read(4)))
            data.append(tmp_data)
    return data
def getlfcdata(destpath=destpath):
    kaiji = getonedata(destpath + 'kaiji.mfc')
    qianjin = getonedata(destpath + 'qianjin.mfc')
    houtui = getonedata(destpath + 'houtui.mfc')
    zuozhuan = getonedata(destpath + 'zuozhuan.mfc')
    youzhuan = getonedata(destpath + 'youzhuan.mfc')
    lfc_data = [kaiji,qianjin,houtui,zuozhuan,youzhuan]
    return lfc_data
label_map = {1:"kaiji",2:"qianjin",3:"houtui",4:"zuozhuan",5:"youzhuan"}
def getorigdata(num,destpath=destpath):
    label_map = {1:"kaiji",2:"qianjin",3:"houtui",4:"zuozhuan",5:"youzhuan"}
    data = []
    for i in range(1,11):
        path = destpath + label_map[num] + str(i) + '.mfc'
        data.append(getonedata(path))
    return data
    
    import numpy as np
import matplotlib.pyplot as plt

def d(data1:np.ndarray,data2:np.ndarray):
    assert data1.shape == data2.shape
    return np.sum((data1 - data2) ** 2)
def DTW(data:np.ndarray,tpl:np.ndarray):
    """
    data is a sequence of characterastic,tpl is also a sequence of characterastic
    :param data:
    :param tpl:
    :return:
    """
    data_len = data.shape[1]
    tpl_len = tpl.shape[1]
    phi = np.zeros((tpl_len,data_len))
    pre = np.zeros((tpl_len,data_len))
    phi[0][0] = d(data[...,0],tpl[...,0])
    pre[0][0] = 100
    # left : -1 ,up : 1 ,leftup : 0  ,end : 100
    # tpl_len row  ,data_len column
    for i in range(1,data_len):
        phi[0][i] = phi[0][i-1] + d(data[...,i],tpl[...,0])
        pre[0][i] = -1
    for j in range(1,tpl_len):
        phi[j][0] = phi[j-1][0] + d(data[...,0],tpl[...,j])
        pre[j][0] = 1
    for z in range(1,(tpl_len if tpl_len < data_len else data_len)):
        for i in range(z,data_len):
            # confirm phi[z][i]
            leftdistant = phi[z][i-1] + d(data[...,i],tpl[...,z])
            updistant = phi[z-1][i] + d(data[...,i],tpl[...,z])
            leftupdistant = phi[z-1][i-1] + 2 * d(data[...,i],tpl[...,z])
            if leftdistant < updistant and leftdistant < leftupdistant:
                phi[z][i] = leftdistant
                pre[z][i] = -1
            elif updistant < leftdistant and updistant < leftupdistant:
                phi[z][i] = updistant
                pre[z][i] = 1
            else:
                phi[z][i] = leftupdistant
                pre[z][i] = 0
        for j in range(z,tpl_len):
            # confirm phi[j][z]
            leftdistant = phi[j-1][z] + d(data[..., z], tpl[..., j])
            updistant = phi[j][z-1] + d(data[..., z], tpl[..., j])
            leftupdistant = phi[j - 1][z - 1] + 2 * d(data[..., z], tpl[..., j])
            if leftdistant < updistant and leftdistant < leftupdistant:
                phi[j][z] = leftdistant
                pre[j][z] = -1
            elif updistant < leftdistant and updistant < leftupdistant:
                phi[j][z] = updistant
                pre[j][z] = 1
            else:
                phi[j][z] = leftupdistant
                pre[j][z] = 0
    return phi[tpl_len-1][data_len-1]
    
    def get_one_label(data,lfc):
    data = np.array(data).T
    d1 = DTW(data,np.array(lfc[0]).T)
    d2 = DTW(data,np.array(lfc[1]).T)
    d3 = DTW(data,np.array(lfc[2]).T)
    d4 = DTW(data,np.array(lfc[3]).T)
    d5 = DTW(data,np.array(lfc[4]).T)
    if d1 < d2 and d1 < d3 and d1 < d4 and d1 < d5:
        return 1
    elif d2 < d1 and d2 < d3 and d2 < d4 and d2 < d5:
        return 2
    elif d3 < d1 and d3 < d2 and d3 < d4 and d3 < d5:
        return 3
    elif d4 < d1 and d4 < d2 and d4 < d3 and d4 < d5:
        return 4
    else:
        return 5
lfc = getlfcdata()
label = get_one_label(getorigdata(3)[3],lfc)
def get_one_label(data,lfc):
    data = np.array(data).T
    d1 = DTW(data,np.array(lfc[0]).T)
    d2 = DTW(data,np.array(lfc[1]).T)
    d3 = DTW(data,np.array(lfc[2]).T)
    d4 = DTW(data,np.array(lfc[3]).T)
    d5 = DTW(data,np.array(lfc[4]).T)
    if d1 < d2 and d1 < d3 and d1 < d4 and d1 < d5:
        return 1
    elif d2 < d1 and d2 < d3 and d2 < d4 and d2 < d5:
        return 2
    elif d3 < d1 and d3 < d2 and d3 < d4 and d3 < d5:
        return 3
    elif d4 < d1 and d4 < d2 and d4 < d3 and d4 < d5:
        return 4
    else:
        return 5
lfc = getlfcdata()
label = get_one_label(getorigdata(3)[3],lfc)
def testaccuracy():
    acc = 0
    sum = 0
    for i in range(1,6):
        for j in range(0,10):
            sum += 1
            if int(get_one_label(getorigdata(i)[j],lfc)) == i:
                acc += 1
    return acc / sum

print(testaccuracy())