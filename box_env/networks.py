import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def neural_network(nodes_given):
    # input and output
    x = tf.placeholder(tf.float32,[None, 49])
    r = tf.placeholder('float')
    gamma = tf.placeholder('float')
  
    #layers
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
            "hid_out":l1,
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