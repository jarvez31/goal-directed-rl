#%%
import numpy as np
import pickle
import csaps
import matplotlib.pyplot as plt
from itertools import zip_longest,chain
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import interpolate
from scipy.signal import savgol_filter
from plot_value import firing_rate_map 


name = 'traj_GEN_sq_rf.pk1'
with open(name, "rb") as f:
    d = pickle.load(f)
    f.close()
locals().update(d)
# print(x)
x = np.asarray(x)
y = np.asarray(y)
env = np.asarray(env)
R = [j*5 for j in R]

print(len(x))
# print(x[61572:61700])
# print(y[61572:61700])
# print(value[61572:61700])

#below is the simple analysis of the value function and why it oscillates in the act.
ind = []
# for i in range(len(x)-1):
    # if (x[i+1] == x[i]) and (y[i] == y[i+1]):
        # print(i)
        # ind.append(i)

# for i in range(60000,65000):
    # p = value[i+1] - value[i]
    # if p < -0.5:
        # k=-1
    # if -0.5 < p < 0.5:
        # k=0
    # if p > 0.5:
        # k=1
    # ind.append(p) 

# print(ind)
# plt.plot(range(len(ind)), ind)
# plt.plot(range(len(value)), value)

# plt.plot(range(len(ind)), ind)
# plt.scatter(ind, [0]*len(ind))
# plt.plot(range(len(x)), x)
# plt.show()

xi = [m[0] for m in env ]
yi = [m[1] for m in env ]
# sz = np.linspace(0,1,len(x))
x = savgol_filter(x,51,3)
y = savgol_filter(y,51,3)
# print(d.keys())
# print(hid_out)
hidden = [x[0] for x in hid_out]
hidden = np.asarray(hidden)
print(hidden.shape)

#%%
'''
# plotting the trajectory in real time (comment this part if you only want the L2 output)
plt.plot(range(len(R)), R)
plt.show()
x_hist = []
y_hist = []

fig, ax = plt.subplots()
points, = ax.plot(x_hist, y_hist)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

x_env, y_env = [m[0] for m in env], [p[1] for p in env]
ax.plot(x_env,y_env)
plt.pause(0.001)
for k in objs:
    x1, y1 = [l[0] for l in k], [n[1] for n in k]
    ax.plot(x1,y1)
    plt.pause(0.001)


ll = 0
for i,j in zip(x,y):
    x_hist.append(i)
    y_hist.append(j)
    # points.set_linestyle(ls='--')
    points.set_data(x_hist, y_hist)

    # points.set_data(x_hist[-2:], y_hist[-2:])
    # points.set_marker(marker='.')
    plt.pause(0.001)
    ll+=1

'''

# printing the ot2 layer output (comment this part if you only want to plot the real time trajectory)
temp = []
for i in ot2:
    temp2 = [j[0] for j in i[:-1]]
    temp.append(temp2)

ot2 = np.asarray(temp)
print(ot2.shape)

ot2 = hidden # comment and uncomment this depending on what you need to plot, LAHN2 or hidden
# plt.plot(range(len(value)), value)
# plt.show()
pos = np.column_stack((x,y))
print(pos.shape)
lim = 0.93
foldiak1respmat = []
shape = (6,8)


# plotting all the cells L2 and hidden both in sections of 48 
# plt.plot(range(len(ot2[:,0])), ot2[:,0])
# plt.plot(range(len(R)), R)
# plt.show()
'''
# for i in range(ot2.shape[1]):
for i in range(0,4):
    for j in range(48):
        print(i*48 +j)
        ot = ot2[:,i*48 + j]
        plt.subplot(shape[0],shape[1],j+1)

        thresh = np.amax(abs(ot)) * lim
        firr = np.nonzero(abs(ot)>thresh)
        firposgrid = pos[firr[0], :]
        plt.plot(pos[:,0], pos[:,1], zorder = 0)
        plt.plot(xi,yi)
        plt.scatter(firposgrid[:,0], firposgrid[:,1], s = 8, color = 'red', marker='o', zorder = 1)
    plt.suptitle("Activity of hidden neurons "+str(i*48 + 1) +" to " + str(48*(i+1)) +" for Plus Maze")
    plt.show()
'''

for i in range(0,4):
    fig, axes = plt.subplots(nrows=6,ncols=8, sharex=True)
    for j in range(48):
        print(i*48 +j)
        # ot = ot2[:,i*48 + j]
        # plt.subplot(shape[0],shape[1],j+1)

        V = ot2[:,i*48 + j]
        # V = ot2[:,136] # k is the cell index you want to plot 
        thresh = np.amax(abs(V)) * 0.92
        firr = np.nonzero(np.absolute(V)>thresh)
        firposgrid = pos[firr[0], :]

        ep1= 20
        gamma1 = 0.5 
        nodes = [200]
        firr, im = firing_rate_map(firposgrid, V, firr, ep1, nodes, gamma1, axes.flat[j])
        # axes.flat[j].axis("off")        
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.suptitle("heat map of hidden neurons "+str(i*48 + 1) +" to " + str(48*(i+1)) +" for Plus Maze")
    # plt.savefig("results/rf/rate_maps/ffneu:" + str(i)+" rf.png")
    plt.show()
    plt.close()
    # plt.suptitle("Activity of hidden neurons "+str(i*48 + 1) +" to " + str(48*(i+1)) +" for Plus Maze")
# plotting the single cell and also plotting the firing rate map of specific cell.
V = ot2[:,136] # k is the cell index you want to plot 
thresh = np.amax(abs(V)) * 0.75
firr = np.nonzero(np.absolute(V)>thresh)
firposgrid = pos[firr[0], :]

ep1= 20
gamma1 = 0.5 
nodes = [200]
firr = firing_rate_map(firposgrid, V, firr, ep1, nodes, gamma1)