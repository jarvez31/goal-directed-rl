#import libraries
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle, os, timeit
import numpy as np
import matplotlib.pyplot as plt
from plot_value import firing_rate_map, hid_plot, matlab_style_gauss2D
from networks import neural_network, value_cal

k = tf.test.is_gpu_available()
print(k)
# define the model 
start = timeit.default_timer()
act = ''

# load data from L2 layer
traj_num = 'sq_norm'
with open('tf_data_'+str(traj_num)+'.pk1', 'rb') as f:
    d = pickle.load(f)
    f.close()
locals().update(d)

with open('traj_fourbowl_'+str(traj_num)+'.pk1', 'rb') as f1:
    d1 = pickle.load(f1)
    f1.close()
locals().update(d1)

print(sum(R))
R = np.asarray(R[:-1]).reshape(len(R[:-1]), 1)
ot2 = ot2.T
ot2_part = []
for p in range(ot2.shape[0]):
    temp1 = ot2[p,:]/(2*np.amax(abs(ot2[p,:])))
    ot2_part.append(np.append(temp1, R[p]))
ot2_part = np.asarray(ot2_part)
# print(ot2_part.shape)


#network metaparameters
nodes = [200]
out_nodes = 1
Train = False 
epochs = 100
mod = 2

#saving is done in these paths (each paths correspond to some data that needs to be saved)
if Train:
    fol = 'mod_' +act  + str(mod)
    os.mkdir(fol)
else:
    fol = 'mod_' +act + str(mod)
save_path = fol + '/model{epochs:'+str(epochs)+' lay:' + str(len(nodes)) + " neu:"
save_path2 = fol + '/error{epochs:'+str(epochs)+' lay:' + str(len(nodes)) + " neu:"
save_path3 = fol + '/value{epochs:'+str(epochs)+' lay:' + str(len(nodes)) + " neu:"
for k in nodes:
    save_path = save_path + str(k) +','
    save_path2 = save_path2 + str(k) +','
    save_path3 = save_path3 + str(k) +','
save_path = save_path + '.ckpt'
save_path2 = save_path2 + '.ckpt'
save_path3 = save_path3 + '.ckpt'

nw = neural_network(nodes)


#Training and testing is done here
gamma = 0.5
if Train:
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        ep_loss = []
        epoch = 0
        epoch_loss = 1000 
        # for epoch in range(epochs):
        while epoch_loss > 6:
            epoch_loss = 0
            for t in range(ot2_part.shape[0]-1):
                x_t, x_t1 = ot2_part[t, :], ot2_part[t+1, :]

                prediction_t1 = sess.run(nw["output"], feed_dict = {nw["x"]:x_t1.reshape((1,49))})
                loss_, _ = sess.run([nw['loss'], nw["optimizer"]], feed_dict = {nw["x"]:x_t.reshape((1,49)), nw["pred"]:prediction_t1, nw["r"]:R[t], nw['gamma']:gamma})

                epoch_loss+=loss_
            ep_loss.append(epoch_loss)
            print('loss of epoch ', epoch, 'loss:', epoch_loss)
            epoch+=1

            if epoch%20 ==0:
                fol1 = fol + '/mid_training_wts_epoch' + str(epoch)
                os.makedirs(fol1)
                saver = nw["saver"]
                saver.save(sess, fol + '/' + save_path)

        plt.plot(range(len(ep_loss),), ep_loss)
        plt.show()
        # epoch = 250
        saver = nw["saver"]
        saver.save(sess, save_path)

        with open(save_path2, 'wb') as dd:
            pickle.dump(ep_loss, dd)
            dd.close()

else:
    with tf.Session() as sess:
        valu = value_cal(sess, save_path, ot2_part, nw)
        
    with open(save_path3, 'wb') as ddd:
        pickle.dump(valu, ddd)
        ddd.close()
    
    V = np.asarray(valu)
    thresh = np.amax(V) * 0.0
    f = np.nonzero(np.absolute(V)>thresh)

    plt.plot(range(len(V)), V)
    g = pos[f[0], :]

    R = [jjj for jjj in R ]

    plt.plot(range(len(R)), R, color='Black')
    plt.title("value function after training of " + str(epochs) + ' epochs')
    plt.show()

    firr = firing_rate_map(g, V, f, epochs, nodes, gamma, fol)
    plt.title("heat map of value function before training")
    plt.show()


stop = timeit.default_timer()
print('Time: ', stop - start)
