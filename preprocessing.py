import pywt
import numpy as np
import matplotlib.pylab as plt

def data_wt(f,lenf_r, day, n, wavelet, mode):
    f_r = np.empty(shape=(lenf_r, day, pow(2, n), 3))
    for s in range(lenf_r):
        f_i = f[s:day+s]/np.mean(f[s:day+s])
        f_i_r = np.empty(shape=(day, pow(2, n), 3))
        for i in range(3):
            a = f_i[:,i]
            a_w = pywt.WaveletPacket(data=a, wavelet=wavelet, mode=mode, maxlevel=n)
            # 获取特定层数的所有节点
            npn = [node.path for node in a_w.get_level(n, 'natural')]
            for j in range(len(npn)):
                new_aw = pywt.WaveletPacket(data=None, wavelet=wavelet, mode=mode)
                new_aw[npn[j]] = a_w[npn[j]]
                a_r_j = new_aw.reconstruct(update=True)
                f_i_r[:, j, i] = a_r_j
        f_r[s] = f_i_r
    print(f_r[-1])
    c = np.zeros(shape=(day,))
    for i in range(pow(2,n)):
        c= c+f_r[-1,:,i,0]
    print(c-f_i[:,0])
    return f_r

def lable(f):
    f_lable = f[1:,2]-f[:-1,2]
    up_index = np.where(f_lable>=0)[0]
    dowm_index = np.where(f_lable<0)[0]
    f_l = np.zeros(shape=(len(f_lable),2))
    f_l[up_index,0] = 1
    f_l[dowm_index,1] = 1
    print(f_l)
    return f_l

if __name__=='__main__':
    f = np.loadtxt('GOLD.txt', usecols=[2, 3, 4], delimiter=',')
    day = 34  # n = 4时，18,34,50,66，n=5时, 34,66
    lenf_r = len(f) + 1 - day
    n = 5
    wavelet = 'sym2'
    mode = 'symmetric'
    f_r = data_wt(f,lenf_r, day, n, wavelet, mode)
    f_l = lable(f)

    data_x = f_r[:-1]
    data_y = f_l[-len(f_r)+1:]
    print(data_x.shape)
    print(data_y.shape)
    np.savetxt('data/data_x.txt',data_x.reshape((len(data_x), day*pow(2, n)*3,)))
    #(6822,34,32,3)
    np.savetxt('data/data_y.txt',data_y)
    #(6822,2)








