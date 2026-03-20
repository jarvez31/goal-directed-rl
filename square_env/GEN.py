#%%
###############################################################################################################################################################
###############################-------------------------CODE for GEN policy update and the actor network for the same--------------------####################
###############################################################################################################################################################

import pickle, os, timeit, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from shapely.geometry import LineString, LinearRing, box, Point, Polygon
import numpy as np
# from random import random
# from actor import actor, neural_network2
import matplotlib.pyplot as plt
from plot_value import firing_rate_map, hid_plot, matlab_style_gauss2D
from GEN_func import model_trained, HD, PI, all_plot, hd, speedpos, speedinit, rew, new_circle, cir_check, mirrorImage, sq_chk, updatestep, mirrorImage2 
from networks import neural_network
'''
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
# from traj_func import new_circle
# import pandas as pd
import pickle, os, matplotlib
import numpy as np
import sklearn, timeit
from GEN_func import hd, HD
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
k = tf.test.is_gpu_available()
'''

start = timeit.default_timer()
'''
def neural_network(nodes_given):
    # input and output
    x = tf.placeholder(tf.float32,[None, 49])
    # y = tf.placeholder('float')
    r = tf.placeholder('float')
    gamma = tf.placeholder('float')
  

    hid_lay1 = {'weights': tf.Variable(tf.random_normal([49, nodes_given[0]]),dtype=tf.float32, name='wts_hid_lay1'),
                'biases': tf.Variable(tf.random_normal([nodes_given[0]]))}
    
    # hid_lay2 = {'weights': tf.Variable(tf.random_normal([nodes[0], nodes[1]]), dtype=tf.float32, name='wts_hid_lay2'),
                # 'biases': tf.Variable(tf.random_normal([nodes[1]]))}
    
    output_lay = {'weights': tf.Variable(tf.random_normal([nodes_given[-1], 1]), dtype=tf.float32, name='wts_output_lay'),
                'biases': tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='bs_output_lay')}

    l1 = tf.add(tf.matmul(x, hid_lay1['weights']), hid_lay1['biases'])
    l1_n = tf.nn.sigmoid(l1)

    # l2 = tf.add(tf.matmul(l1_n, hid_lay2['weights']), hid_lay2['biases'])
    # l2_n = tf.nn.sigmoid(l2)
    
    inp = l1_n
    output = tf.add(tf.matmul(inp, output_lay['weights']), output_lay['biases'])

    pred = tf.placeholder('float')
    tar = r + gamma*pred
    loss = tf.reduce_mean(tf.losses.mean_squared_error(tar,output))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001).minimize(loss)

    saver = tf.train.Saver()
    
    return {"output":output,
            "hid_out":l1,
            "x":x,
            "loss":loss,
            "optimizer":optimizer,
            "r":r,
            "gamma":gamma,
            "pred":pred,
            "saver":saver}
'''

def value_cal(tf_session, path, data_pt, model):
    saver = model["saver"]
    saver.restore(tf_session, path)        
    x_t = data_pt
    val = tf_session.run(model["output"], feed_dict = {model["x"]:x_t.reshape((1,49))})
    hid_out = tf_session.run(model["hid_out"], feed_dict = {model["x"]:x_t.reshape((1,49))})
    
    return val, hid_out

#creating the environment----------------------------------------------
obj_c = [(0.85, 0), (0, 0.85), (-0.85, 0), (0, -0.85)]
obj_r = [0.7, 0.3] 
bowl_ind = [0,1,2,3]
circles = [Polygon(new_circle("box", True, pp, obj_r)) for pp in obj_c] # come back to this

out_r = 0
plus = new_circle("plus", False, (0,0), out_r)
objs = []
# plus = new_circle(flag=4, obj=False, c=(0,0))
for i in obj_c:
    objs.append(new_circle("box", True, i, obj_r))

#model details-------------------------------------------
nodes = [200]
nodes_act = [50]
out_nodes = 1
epochs = 100
retrain_eps = 20
mod = 1

fol1 = 'actormod_' + str(mod)
# os.mkdir(fol1)
fol = 'mod_' + str(mod)
save_path = fol + '/model{epochs:'+str(epochs)+' lay:' + str(len(nodes)) + " neu:"
for k in nodes:
    save_path = save_path + str(k) +','
save_path = save_path + '.ckpt'

'''
epochs=10
fol2 = 'mod_2' # + str(mod)
act = 'sigmoid'
LR = [0.001] #, 0.001, 0.0001]
save_path2 = fol2 + '/actor{epochs:'+str(epochs)+' lay:' + str(len(nodes)) + ' act: ' + act[0:3] + ' lr: '+ str(LR[0])+" neu:"
for k in nodes:
    save_path2 = save_path2 + str(k) +','
save_path2 = save_path2 + '.h5'
'''

nw = neural_network(nodes)
# nw2 = neural_network2(nodes_act)
gamma = 0.5

#GEN parameters-----------------------------
t_old = [0]*100
delV_high = 0.7
delV_low = -0.7 

# initial points for the trajectory-----------------------
x = 0.0
y = 0.0
rad = 0.003
x1 = random.uniform(x-rad, x+rad)
x2 = random.uniform(x1-rad, x1+rad)

y1 = random.uniform(y-rad, y+rad)
y2 = random.uniform(y1-rad, y1+rad)
x_hist = [x, x1, x2]
y_hist = [y, y1, y2]

speed_init = speedinit(x_hist,y_hist)
theta_init = hd(x_hist, y_hist) # check this code I find something fishy here. len(speed) and len(theta) not equal.

# kkk = tf.Session()
# jjj = tf.Session()
    # kkk = sess

# saving the speed, value, theta and LAHN outputs of the random initialization------------------------------------ 
ot2 = []
value = []
R = []
hidden_out = []
for k in range(len(speed_init)):
    ot2_part , t_old, head = model_trained(speed_init[k], theta_init[k+1], t_old, theta_init[0])
    ot2_part = ot2_part/(2*np.amax(ot2_part))
    head = head/(2*np.amax(head))
    reward = rew(x_hist[k+1], y_hist[k+1], circles)
    ot2_part = np.vstack((ot2_part, [reward]))
    R.append(reward)

    with tf.Session() as sess:
        valu, hid_temp = value_cal(sess, save_path, ot2_part, nw)
    value.append(valu[0][0])
    hidden_out.append(hid_temp)
    ot2.append(ot2_part)

# parameters for the circle checking-----------------------------------
firstvisit = False
visit_circ = [None]
go_in_ind = [] 

#delete these variables later
gen_output =[]
actor_output = []
# modelx = load_model(save_path2)

jj = []
# Main loop----------------------------------------
for T in range(120000):
    if T%1000 ==0:
        print(T)
    
    delV = value[-1] - value[-2]
    
    # continuous GEN update
    # delXprev = x_hist[-1] - x_hist[-2]
    # delYprev = y_hist[-1] - y_hist[-2]
    # xnew = updatestep(delXprev, delV, delV_high, delV_low) + x_hist[-1]
    # ynew = updatestep(delXprev, delV, delV_high, delV_low) + y_hist[-1]
    
    
    #getting the GEN update
    if delV > delV_high:
        xnew = 2*x_hist[-1] - x_hist[-2]
        ynew = 2*y_hist[-1] - y_hist[-2]

    if delV < delV_low:
        xnew = x_hist[-2]
        ynew = y_hist[-2]

    else:
        sigma = 0.35
        es = (delV**2)/(sigma**2)
        l = np.exp(-es)*np.random.normal(0,0.02)
        l2 = np.exp(-es)*np.random.normal(0,0.02)
        while (l == 0) and (l2==0):
            print(es)
            # l = np.exp(-es)*np.random.normal(0,0.008)
            # l2 = np.exp(-es)*np.random.normal(0,0.008)
        xnew, ynew = x_hist[-1] + l , y_hist[-1] + l2

    # gen_output.append([xnew, ynew])
    #getting the actor update by giving GEN as an input
    if (delV_low < delV) and (delV < delV_high):
        lr = 0.00001
    else:
        lr = 0.01
        
    delx = xnew - x_hist[-1]
    dely = ynew - y_hist[-1]
    # newdelX, newdelY = actor(ot2_part, fol1, delV, [delx, dely], nodes_act, T, nw2, head, lr)
    # if T>20000:
        # xnew = (newdelX/delV) + x_hist[-1]
        # ynew = (newdelY/delV) + y_hist[-1]

    # actor_output.append([xnew, ynew])

    #check if the points are outside the env and take mirror image
    while not Polygon(plus).contains(Point((xnew,ynew))):
        # chk = sq_chk(plus, xnew, ynew)
        chk_pts = [(x_hist[-1], y_hist[-1]), (xnew, ynew)]
        xnew, ynew = mirrorImage2(plus, chk_pts)
        # print(xnew,ynew)

        # print("yes")

    
    # calculating speed, theta and LAHN2 output for the next step
    x_hist.append(xnew), y_hist.append(ynew)
    speed = speedpos(x_hist[-2:], y_hist[-2:]) # in both of these you are passing entire lists, see if you can pass only the points
    if len(x_hist)>=10:
        theta = hd(x_hist[-10:], y_hist[-10:])
    else:
        theta = hd(x_hist, y_hist)

    ot2_part , t_old, head = model_trained(speed, theta[-1], t_old, theta_init[0])
    ot2_part = ot2_part/(2*np.amax(abs(ot2_part)))
    head = head/(2*np.amax(abs(head)))

    # check which circle is visited and save the circle and the index 
    check, circnum = cir_check(circles, xnew, ynew)
    old_len = len(visit_circ)
    if circnum in bowl_ind:
        if circnum != visit_circ[-1]:
            visit_circ.append(circnum)
            print(visit_circ)
    new_len = len(visit_circ)

    if new_len > old_len:
        t = T
        go_in_ind.append(t)
        print(t)    
        plt.plot(xnew, ynew, 'bo')

    # a second check if it is going out
    reward = rew(x_hist[-1], y_hist[-1], circles)
    if reward == -1:
        print("Baahar chala gaya")
        break
    
    # making sure the reward disappears later
    if check and T<t+1500:
        reward = rew(x_hist[-1], y_hist[-1], circles)
    else:
        reward = 0
    
    #saving and calculating the LAHN and the value outputs
    ot2_part = np.vstack((ot2_part, [reward]))
    ot2.append(ot2_part)
    R.append(reward)

    with tf.Session() as sess:
        valu, hid_temp = value_cal(sess, save_path, ot2_part, nw)
    value.append(valu[0][0])
    hidden_out.append(hid_temp)

    if T%1000==0:
        d = {'x':list(x_hist), 'y':list(y_hist), 'env': plus, 'objs': objs, 'R':R, 'ot2':ot2, 'hid_out':hidden_out, "value":value, "t_old":t_old} #, 'obj': small_box}
        with open('traj_GEN_sq_rx.pk1', 'wb') as f:
            pickle.dump(d, f)

print(sum(R))
plt.plot(x, y, 'bo')
plt.scatter(x_hist, y_hist)
plt.plot(x_hist, y_hist)
plt.show()        

#save the trajectory and all the required variables
d = {'x':list(x_hist), 'y':list(y_hist), 'env': plus, 'objs': objs, 'R':R, 'ot2':ot2, 'hid_out':hidden_out, "value":value, "t_old":t_old} #, 'obj': small_box}
with open('traj_GEN_sq_rx.pk1', 'wb') as f:
    pickle.dump(d, f)

d2 = {'gen_output':gen_output, 'actor_output':actor_output}
with open('del_var2.pk1', 'wb') as ff:
    pickle.dump(d2, ff)

stop = timeit.default_timer()
print('Time: ', stop - start)