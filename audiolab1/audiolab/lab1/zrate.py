import numpy as np

def zrate(wave_data:np.ndarray)->np.ndarray:
    if len(wave_data.shape) == 1:
        wave_data = np.expand_dims(wave_data,axis=0)
    ans = []
    nframe = wave_data.shape[1]
    w = 1 / (2 * nframe)
    for x in wave_data:
        sign = np.sign(x)
        sign_left = sign[1:]
        sign = sign[:-1]
        res = np.sum(np.abs(sign-sign_left) * w,axis=0)
        ans.append(res)
    ans = np.array(ans)
    return ans

testing = False
if testing:
    x = np.array([1,1,1,1])
    y = np.array([1,-1,1,-1])
    z = np.array([1,-1,1,1])
    print(zrate(x))
    print(zrate(y))
    print(zrate(z))
    h = np.array([[1,1,1,1],[1,-1,1,-1]])
    print(zrate(h))
