import pickle, os, timeit, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def neural_network2(nodes):
    # input and output
    x = tf.placeholder(tf.float32,[None, 149])
    lr = tf.placeholder(tf.float32)
  

    hid_lay1 = {'actor_weights': tf.Variable(tf.random_normal([149, nodes[0]]),dtype=tf.float32, name='actor_wts_hid_lay1'),
                'actor_biases': tf.Variable(tf.random_normal([nodes[0]]))}
        
    output_lay = {'actor_weights': tf.Variable(tf.random_normal([nodes[-1], 2]), dtype=tf.float32, name='actor_wts_output_lay'),
                'actor_biases': tf.Variable(tf.random_normal([2]), dtype=tf.float32, name='actor_bs_output_lay')}

    l1 = tf.add(tf.matmul(x, hid_lay1['actor_weights']), hid_lay1['actor_biases'])
    l1_n = tf.nn.sigmoid(l1)
    
    inp = l1_n
    output = tf.add(tf.matmul(inp, output_lay['actor_weights']), output_lay['actor_biases'])
    output = tf.nn.sigmoid(output)

    pred = tf.placeholder('float')
    tar = pred
    loss = tf.reduce_mean(tf.losses.mean_squared_error(tar,output))
    optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)

    # print(lr)
    saver2 = tf.train.Saver()
    
    return {"output":output,
            "hid_out":l1,
            "x":x,
            "loss":loss,
            "optimizer":optimizer,
            "pred":pred,
            "saver":saver2,
            "lr": lr}



def actor(ot2, fol, delV, gen_output, nodes_act, T, model, head, rate):
    # if T%100 ==0:
        # print(T)
    nodes = nodes_act 
    out_nodes = 2
    # Train = True

    save_path = fol + '/actor{lay:' + str(len(nodes)) + " neu:"
    for k in nodes:
        save_path = save_path + str(k) +','
    save_path = save_path + '.ckpt'
    
    nw2 = model
    # nw2 = neural_network2(nodes)

    with tf.Session() as sess2:
        if T == 0:
            sess2.run(tf.initialize_all_variables())

            x_t = np.append(ot2, head) 
            target = [delV * k for k in gen_output]

            loss_, _, m = sess2.run([nw2['loss'], nw2["optimizer"], nw2["lr"]], feed_dict = {nw2["x"]:x_t.reshape((1,149)), nw2["pred"]:target, nw2["lr"]:rate})
            output_update = sess2.run(nw2["output"], feed_dict = {nw2["x"]:x_t.reshape((1,149))})
            # print(m)
            # ep_loss.append(loss_)
            file_writer = tf.summary.FileWriter(fol, sess2.graph)
            # var_name_list = [v for v in tf.trainable_variables()]
            saver2 = nw2["saver"]
            saver2.save(sess2, save_path)
            # print(len(tf.trainable_variables()))
            # print('dekho mai aaya hun')
            output_update = [gen_output]

        else:
            # print('mai ab second mein aaya hun')
            # with tf.Session() as sess2:
                # sess2.run(tf.initialize_all_variables())
            # print(len(tf.trainable_variables()))
            # print(tf.trainable_variables())
            # var_name_list = [v.name for v in tf.trainable_variables()]
            # print(var_name_list)
            saver2 = nw2["saver"]
            saver2.restore(sess2, save_path)        

            # ep_loss = []

            x_t = np.append(ot2, head) 

            target = [delV * k for k in gen_output]

            loss_, _, m = sess2.run([nw2['loss'], nw2["optimizer"], nw2["lr"]], feed_dict = {nw2["x"]:x_t.reshape((1,149)), nw2["pred"]:target, nw2["lr"]:rate})
            output_update = sess2.run(nw2["output"], feed_dict = {nw2["x"]:x_t.reshape((1,149))})
            # print(m)
            # ep_loss.append(loss_)

            saver = nw2["saver"]
            saver.save(sess2, save_path)
            # print(output_update)
    
    return output_update[0][0], output_update[0][1]




















