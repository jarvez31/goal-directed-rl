import pickle, os, timeit, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from shapely.geometry import LineString, LinearRing, box, Point, Polygon
from actor import neural_network2
from GEN_func import model_trained, HD, PI, all_plot, hd, speedpos, speedinit, rew, new_circle, cir_check, mirrorImage, sq_chk
from actor_train_func import initial_points, initial_data, env, value_cal#, neural_network
import matplotlib.pyplot as plt
# from retrain2 import neural_network
from networks import neural_network

nodes_act = [50]
mod = 2
fol1 = 'mod_' + str(mod)
fol = 'actormod_' + str(mod)
epochs = 100
nodes= [200]
# os.mkdir(fol)

save_path2 = fol1 + '/model{epochs:'+str(epochs)+' lay:' + str(len(nodes)) + " neu:"
save_path = fol + '/actor{lay:' + str(len(nodes_act)) + " neu:"
for k in nodes_act:
    save_path = save_path + str(k) +','
save_path2 = save_path2 + str(200) +','
save_path = save_path + '.ckpt'
save_path2 = save_path2 + '.ckpt'
# print(k)
start = timeit.default_timer()
nw = neural_network(nodes)
nw2 = neural_network2(nodes_act)

bowl_rad = 0.25
bowl_cen = [(0.75,0), (0, 0.75), (-0.75, 0), (0, -0.75)]
bowl_ind = [0,1,2,3]
bowls, circles = env(bowl_cen, bowl_rad)
square = new_circle(flag=4, obj=False, c=(0,0))


Train = False


if Train:
    name = 'traj_GEN_new.pk1'
    with open(name, "rb") as f:
        d = pickle.load(f)
        f.close()
    locals().update(d)
    env = np.asarray(env)
    ot2 = np.asarray(ot2).reshape((len(ot2), 49))

    t = hd(x, y)
    heading = []
    for ii in t:
        _, head = HD(ii,t[0])
        heading.append(head/(2*max(head)))

    heading = np.asarray(heading)
    ot2_part = np.append(ot2, heading[1:,:], axis=1)

    delX = []
    delY = []
    delV = []
    # 
    for i in range(len(x)-1):
        delta_X = x[i] - x[i-1]
        delta_Y = y[i] - y[i-1]
        delta_V = value[i] - value[i-1]
        delX.append(delta_X)
        delY.append(delta_Y)
        delV.append(delta_V)

    pos = np.column_stack((np.asarray(delX), np.asarray(delY)))
    delV = np.column_stack((np.asarray(delV), np.asarray(delV)))
    # 
    num = 200000
    y_train = np.multiply(pos[:num, :], delV[:num, :])
    X_train = ot2_part[:num, :]


    delV_low = -0.8
    delV_high = 0.8
    # 

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        ep_loss = []
        epoch = 0
        epoch_loss = 1000 
        # for epoch in range(epochs):
        while epoch_loss > 0.06:
            epoch_loss = 0
            for t in range(X_train.shape[0]):
                x_t = X_train[t,:] 
                target = y_train[t,:]

                if (delV_low < delV[t,0]) and (delV[t,0] < delV_high):
                    rate = 0.000001
                else:
                    rate = 0.0001

                loss_, _, m = sess.run([nw2['loss'], nw2["optimizer"], nw2["lr"]], feed_dict = {nw2["x"]:x_t.reshape((1,149)), nw2["pred"]:target, nw2["lr"]:rate})
                output_update = sess.run(nw2["output"], feed_dict = {nw2["x"]:x_t.reshape((1,149))})

                epoch_loss+=loss_
            ep_loss.append(epoch_loss)
            print('loss of epoch ', epoch, 'loss:', epoch_loss)
            epoch+=1

            if epoch%20 ==0:
                fol1 = fol + '/mid_training_wts_epoch' + str(epoch)
                os.makedirs(fol1)
                saver = nw2["saver"]
                saver.save(sess, fol + '/' + save_path)

        plt.plot(range(len(ep_loss),), ep_loss)
        plt.show()
        # epoch = 250
        saver = nw2["saver"]
        saver.save(sess, save_path)

        # with open('error.pk1', 'wb') as dd:
            # pickle.dump(ep_loss, dd)
            # dd.close()


else:
    x, y, rad = 0.0, 0.0, 0.003
    x_hist, y_hist = initial_points(x,y,rad)

    speed_init = speedinit(x_hist,y_hist)
    theta_init = hd(x_hist, y_hist) # check this code I find something fishy here. len(speed) and len(theta) not equal.

    ot2, ot2_part, R, head, t_old, value, hidden_out = initial_data(x_hist, y_hist, circles, save_path2, nw)
    x_t = np.append(ot2_part, head) 
    delV = 0.001

    visit_circ = [None]
    go_in_ind = [] 

    for T in range(20000):
        if T%1000 == 0:
            print(T)
        delV = value[-1] - value[-2]

        with tf.Session() as sess2:
            saver2 = nw2["saver"]
            saver2.restore(sess2, save_path)

            output_update = sess2.run(nw2["output"], feed_dict = {nw2["x"]:x_t.reshape((1,149))})
        newdelX, newdelY = output_update[0][0], output_update[0][1]
            
        xnew = (newdelX/delV) + x_hist[-1]
        ynew = (newdelY/delV) + y_hist[-1]
            # print(xnew)

            #check if the points are outside the env and take mirror image
        while not Polygon(square).contains(Point((xnew,ynew))):
            chk_pts = [(x_hist[-1], y_hist[-1]), (xnew, ynew)]
            xnew, ynew = mirrorImage(square, chk_pts)

            # calculating speed, theta and LAHN2 output for the next step
        x_hist.append(xnew), y_hist.append(ynew)
        speed = speedpos(x_hist[-2:], y_hist[-2:]) # in both of these you are passing entire lists, see if you can pass only the points
            # print(x_hist)
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
        # delV+=0.01

        with tf.Session() as sess3:
            valu, hid_temp = value_cal(sess3, save_path2, ot2_part, nw)
        value.append(valu[0][0])
        hidden_out.append(hid_temp)

d = {'x':list(x_hist), 'y':list(y_hist), 'env': square, 'objs': bowls, 'R':R, 'ot2':ot2, 'hid_out':hidden_out, "value":value, "t_old":t_old} #, 'obj': small_box}
with open('traj_GEN_act_new.pk1', 'wb') as f:
    pickle.dump(d, f)



