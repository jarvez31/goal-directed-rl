# Import all needed libraries and sublibraries

import tensorflow as tf
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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
k = tf.test.is_gpu_available()
print(k)
start = timeit.default_timer()
with open('tf_data_testnorm.pk1', "rb") as f:
    d = pickle.load(f)
    f.close()
locals().update(d)
ot2 = ot2.T
print(ot2.shape)

with open('traj_fourbowl_sq_norm.pk1', "rb") as f:
    d = pickle.load(f)
    f.close()
locals().update(d)

# print(x)
x = np.asarray(x)
y = np.asarray(y)
env = np.asarray(env)
pos = np.column_stack((x,y))
# pos = y
# pos1 = x[:-1]
# pos2 = y[:-1]
# print(pos1.shape)

y_train = pos[:150000, :]
y_test = pos[150000:-1, :]
# X_train,X_test = train_test_split(ot2, test_size=0.2)
# y_train,y_test = train_test_split(pos1, test_size=0.2)
# 
X_train = ot2[:150000, :]
X_test= ot2[150000:, :]
# y_train_x = pos1[:180000]
# y_test_x = pos1[180000:]
# y_train_y = pos2[:180000]
# y_test_y = pos2[180000:]

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
Train = True
LR = [0.001] #, 0.001, 0.0001]
output_size = 2
epochs = 10
mod = 3
nodes = [400, 200]
act = 'tanh'

fol = 'mod_' + str(mod)
save_path = fol + '/model{epochs:'+str(epochs)+' lay:' + str(len(nodes)) + ' act: ' + act[0:3] + ' lr: '+ str(LR[0])+" neu:"
for k in nodes:
    save_path = save_path + str(k) +','
save_path = save_path + '.h5'

if Train:
    os.mkdir(fol)
    for i in LR:
        #Defines linear regression model and its structure
        model = Sequential()

        model.add(Dense(nodes[0], input_shape = (48,), activation = act)) #1
        model.add(Dense(nodes[1], activation = act)) #1
        # model.add(Dense(nodes[2], activation = act)) #1
        model.add(Dense(output_size, activation= act)) #5

        model.compile(Adam(lr=i), loss='mean_squared_error')

        #Fits model
        history = model.fit(X_train, y_train, epochs = epochs, validation_split = 0.1)
        history_dict=history.history

        #Plots model's training cost/loss and model's validation split cost/loss
        loss_values = history_dict['loss']
        val_loss_values=history_dict['val_loss']
        plt.figure()
        plt.plot(loss_values,'bo',label='training loss')
        plt.plot(val_loss_values,'r',label='val training loss')
        plt.savefig(fol + "/loss curve.png")
        plt.show()

        model.save(save_path)

# else:
    for i in LR:
        modelx = load_model(save_path)

        y_train_pred = modelx.predict(X_train)
        y_test_pred = modelx.predict(X_test)

        # Calculates and prints r2 score of training and testing data
        print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
        print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))

        ff = open(fol+ '/result.txt', 'w+')
        # for _ in range(2):
        ff.write("The R2 score on the Train set is: " + str(r2_score(y_train, y_train_pred)) +"\n")
        ff.write("The R2 score on the Test set is: " + str(r2_score(y_test, y_test_pred)))
        ff.close()

stop = timeit.default_timer()
print(stop-start)
