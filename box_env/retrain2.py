import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle, os, timeit
import numpy as np
import matplotlib.pyplot as plt
from plot_value import firing_rate_map, hid_plot, matlab_style_gauss2D
from networks import neural_network, value_cal

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
            "x":x,
            "loss":loss,
            "optimizer":optimizer,
            "r":r,
            "gamma":gamma,
            "pred":pred,
            "saver":saver}


def value_cal(tf_session, path, data, model):
    # with tf.Session() as sess:
    saver = model["saver"]
    saver.restore(tf_session, path)        
    value = []
    for t in range(data.shape[0]):
        x_t = data[t, :]
        val = tf_session.run(model["output"], feed_dict = {model["x"]:x_t.reshape((1,49))})
        value.append(val[0,0])
        print(t)
    
    return value
'''

traj_num = 'testcomp'
with open('tf_data_'+str(traj_num)+'.pk1', 'rb') as f:
    d = pickle.load(f)
    f.close()
locals().update(d)

with open('traj_fourbowl_'+str(traj_num)+'.pk1', 'rb') as f1:
    d1 = pickle.load(f1)
    f1.close()
locals().update(d1)

go_out_ind.append(len(x)+1)
for ii,jj in zip(go_in_ind[:-1], go_out_ind[:-1]):
    for i in range(ii+7500,jj+1):
        R[i] = 0

for j in range(go_in_ind[-1]+7500, len(R)):
    R[j] = 0
plt.scatter(go_out_ind, [0]*len(go_out_ind))
plt.plot(range(len(R)), R)
plt.show()

R = np.asarray(R[:-1]).reshape(len(R[:-1]), 1)
kk=np.where(ot2==np.amax(ot2))
print(kk)
ot2 = ot2/(np.amax(ot2))
ot2 = ot2.T
ot2 = np.hstack((ot2, R))
ot2_part = ot2
print(ot2_part.shape)

plt.plot(range(len(ot2[:,13])),ot2[:,13])
plt.plot(range(len(ot2[:,-1])),ot2[:,-1], color="Black")
plt.show()



# go_out_ind.remove(go_out_ind[1])

nodes = [200]
out_nodes = 1
Train = False
epochs = 100
retrain_eps = 20
mod = 2

if Train:
    fol = 'mod_' + str(mod)
    # os.mkdir(fol)
else:
    fol = 'mod_' + str(mod)
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
gamma = 0.5
if Train:
    with tf.Session() as sess:
            saver = nw["saver"]
            saver.restore(sess, save_path)
            ep_loss = []
            epoch_loss = 1000
            epoch = 0 
            # for epoch in range(epochs):
            while epoch_loss > 3:
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

            saver = nw["saver"]
            saver.save(sess, save_path)

            plt.plot(range(len(ep_loss)), ep_loss)
            plt.show()


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

    p = np.linspace(0,0.4,10)
    for l in go_out_ind:
        plt.plot([l]*len(p), p)
    plt.plot(go_in_ind, [0]*len(go_in_ind), 'bo')
    plt.plot(go_out_ind, [0]*len(go_out_ind), 'ro')
    plt.plot(range(len(R)), R, color='Black')
    plt.title("value function after further training of " + str(retrain_eps) + ' epochs')
    plt.show()

    firr = firing_rate_map(g, V, f, retrain_eps, nodes, gamma, fol)


stop = timeit.default_timer()
print('Time: ', stop - start)