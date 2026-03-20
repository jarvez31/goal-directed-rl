#%%
import pickle, scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage
from matplotlib import cm
from iteration_utilities import flatten
from mpl_toolkits.mplot3d import Axes3D

def matlab_style_gauss2D(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    p = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    p[ p < np.finfo(p.dtype).eps*p.max() ] = 0
    sumh = p.sum()
    if sumh != 0:
        p /= sumh
    return p

def firing_rate_map(firposgrid, ot, firr, ep1, nodes, gamma1, fol=None):
    res = 15
    #firr = list(firr[0])
    x = np.arange(-1, 1, 1/res)
    y = np.arange(-1, 1, 1/res)
    fx,fy = np.meshgrid(y, x)
    firingmap = np.zeros(fx.shape)
    firingvalue = np.absolute(ot[firr])

    for ii in range(len(firposgrid)-1):
        q1 = np.argmin(abs(firposgrid[ii,0] - fx[0,:]))
        q2 = np.argmin(abs(firposgrid[ii,1] - fx[0,:]))
        firingmap[q1,q2] = firingvalue[ii]

    firingmap = firingmap/np.max(firingmap)
    gaussian = matlab_style_gauss2D([10, 10], 1.5)
    spikes_smooth = scipy.signal.convolve2d(gaussian, firingmap)
    
    # lay1 =len(nodes)
    plt.imshow(spikes_smooth.T, cmap=cm.jet ,origin= 'lower')
    plt.colorbar()
    plt.title("heat map of value function for plus maze")
    # plt.title("heat map of value function(ep:" + str(ep1) + ' hid-lay:' +str(lay1)+ ' gam:' +str(gamma1)+ ' neu: ' +str(nodes[0])+ ')')
    # plt.savefig(fol + "/heat map of value function(ep:" + str(ep1) + ' hid-lay:' +str(lay1)+ ').png')
    plt.show()

    return x

def hid_plot(hid_lay, posit, lim, shape, o_box, s_box=None):
    foldiak1respmat = []
    # ot_out = []
    sz_h = hid_lay.shape
    for i in range(sz_h[1]):
        plt.subplot(shape[0],shape[1],i+1)
        # w = np.reshape(T[i, :], (1,sz_T[1]))
        ot = hid_lay[:,i]
        # ot_out.append(ot)
        thresh = np.amax(ot) * lim
        firr = np.nonzero(abs(ot)>thresh)
        # foldiak1respmat.append(list(chain.from_iterable(np.multiply((abs(ot)>thresh), ot))))
        firposgrid = posit[firr[0], :]
        plt.plot(posit[:,0], posit[:,1], zorder = 0)
        plt.scatter(firposgrid[:,0], firposgrid[:,1], s = 1, color = 'red', marker='o', zorder = 1)
        plt.plot(o_box[:,0], o_box[:,1])
        # plt.plot(s_box[:,0], s_box[:,1])
    plt.suptitle('Firing field of all hidden layer neuron (threshold:' +str(lim) +', epochs:2 )')
    # print(firposgrid)
    # print(firr)
    # ot_out = np.asarray(ot_out).reshape(sz_T[0], PI1d.shape[1])
    plt.show()
    foldiak1respmat = np.asarray(foldiak1respmat)
    return foldiak1respmat

'''
traj_num = 'test3'
with open('traj_fourbowl_'+str(traj_num)+'.pk1', "rb") as f:
    d = pickle.load(f)
    f.close()
locals().update(d)
# print(x)
x = np.asarray(x)
y = np.asarray(y)
env = np.asarray(env)
pos = np.column_stack((x,y))
# print(go_in_ind)
# print(go_out_ind)
# plt.plot(x,y)
# plt.show()
# print(go_in_bowl)
# print(go_out_bowl)

# plt.plot(range(len(R)), R)
# plt.title('Reward for the time steps')
# plt.show()

epochs = 100
nodes = [200]
gamma = 0.8
with open('tf_data_'+str(traj_num)+'.pk1', "rb") as ff:
    d = pickle.load(ff)
    ff.close()
locals().update(d)
# print(go_in_bowl)

# remember uncommenting them
# go_out_ind.remove(go_out_ind[2])
# go_out_ind.remove(go_out_ind[2])
# print(go_in_ind)
# print(go_out_ind)

# go_out_ind.append(len(x)+1)
# for ii,jj in zip(go_in_ind[:-1], go_out_ind[:-1]):
    # for i in range(ii+7500,jj+1):
        # R[i] = 0
# 
# for j in range(go_in_ind[-1]+7500, len(R)):
    # R[j] = 0
# plt.show()


# print(R)
mod = 1
act = ''
fol = 'mod' + act +'_' + str(mod)
file_val = fol + '/value{epochs:'+str(epochs)+' lay:' + str(len(nodes)) + " neu:"
for k in nodes:
    file_val = file_val + str(k) +','
file_val = file_val +'.ckpt'

with open(file_val, 'rb') as jkl:
    pp = pickle.load(jkl)
    jkl.close()

# d = list(flatten(flatten(pp)))
V = np.asarray(pp)
thresh = np.amax(V) * 0.0
f = np.nonzero(np.absolute(V)>thresh)


file_hid = 'hid_(model_2_keras_fourbowl).pk1'
with open(file_hid, 'rb') as lmn:
    hid = pickle.load(lmn)
    lmn.close()
# locals().update(pp)

# print(f)
# print(delta)

# ------------------------------uncomment this------------------
plt.plot(range(len(V)), V)
g = pos[f[0], :]

R = [jjj for jjj in R ]
# for iii in go_in_ind:
    # e = range(iii,iii+7500)
    # plt.plot(e, [0]*len(e))
    # 
plt.plot(go_in_ind, [0]*len(go_in_ind), 'bo')
plt.plot(go_out_ind, [0]*len(go_out_ind), 'ro')
plt.plot(range(len(R)), R, color='Black')
epochs = 100
plt.title("value function after further training of " + str(epochs) + ' epochs')
plt.show()

firr = firing_rate_map(g, V, f, epochs, len(nodes), gamma)

kjh = hid_plot(hid, pos, 0.98, (5,2), env)
neu_no = 0
plt.plot(range(hid.shape[0]), hid[:,neu_no])
plt.title('raw for of hidden layer neuron number ' + str(neu_no +1) + 'epochs: 2')
plt.show()

# vv = np.asarray(V).reshape((len(V), 1))
# print(len(x)) 
# print(len(y))
# print(vv.shape)
# fig = plt.figure()
# ax= plt.axes(projection='3d')
# ax = Axes3D(fig)
# ax.plot_trisurf(x[:-1], y[:-1], V, cmap=cm.jet, linewidth= 0.1)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# plt.show()
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('wireframe')
# plt.plot(range(hid_fn.shape[0]), hid_fn[:,0])'''