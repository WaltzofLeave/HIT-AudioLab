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
    print(phi)
    return phi[tpl_len-1][data_len-1]

def test():
    x = np.array([[1,4,2,3],[2,3,4,5]])
    y = np.array([[2,5,3,4,1],[2,1,2,1,3]])
    z = DTW(x,y)
    print(z)

test()