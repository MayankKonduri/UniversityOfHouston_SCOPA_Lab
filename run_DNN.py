import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import sys

def mse_loss(theta_true, theta_pred):
    return tf.reduce_mean(tf.square(theta_true - theta_pred))

path = '/Users/mayank/Documents/Github/pdeml'
chkfn = "chkpts.keras"

from SetupNN import *
from PerfMeasures import *

n_epoch = 300
l_rate = 0.001
n_units = 32
batch = 8
period = 200

n_test = 2000
#n_train = 1000
#n_train = 1000
#n_train = 2000
n_train = 4000
#n_train = 8000

if (n_train > 8000):
    print("we need 2K training samples")

data = load_nndata("rdiffEQ-nn-spl-10000.npz", n_test, n_train)
#data = load_nndata("rdiffEQ-nn-spl-10000-addnoise-delta=0.001.npz", n_test, n_train)
#data = load_nndata("rdiffEQ-nn-spl-10000-addnoise-delta=0.01.npz", n_test, n_train)
#data = load_nndata("rdiffEQ-nn-spl-10000-addnoise-delta=0.01.npz", n_test, n_train)
#data = load_nndata("rdiffEQ-nn-spl-10000-addnoise-delta=0.0.npz", n_test, n_train)

t_shape = data["theta_train"].shape
y_shape = data["y_train"].shape





nn_para = {
"n_layers":1,
"n_units":n_units,
"y_shape":y_shape,
"t_shape":1,
"final_actfctn":"swish"
}

nn_model = build_dnn( nn_para )

nn_model.summary()
opt = keras.optimizers.Adam( learning_rate=l_rate )
#nn_model.compile( opt, loss="MSE", metrics=["accuracy"] )
nn_model.compile( opt, loss=mse_loss, metrics=["accuracy"] )
nn_model.summary()

model_callback = tf.keras.callbacks.ModelCheckpoint( filepath= path +"/chkpts/" + chkfn,
    save_weights_only=False,
    #monitor='val_loss',
    monitor='loss',
    mode='accuracy',
#    period=period,
    save_best_only=False, save_freq='epoch')


y_train = data["y_train"]
t_train = data["theta_train"]


print("number of training data", y_train.shape[0])
print("number of features     ", y_train.shape[1])
#n_data = 1000
#y_train = y_train[:n_data]
#t_train = t_train[:n_data]


# train neural network
nn_model.fit( y_train, t_train, epochs=n_epoch, batch_size=batch, callbacks=model_callback)


data = load_nndata("rdiffEQ-nn-spl-10000-addnoise-delta=0.01.npz", n_test, n_train)

y_test = data["y_test"]
print(y_test.shape)
print("number of testing data", y_test.shape[0])

theta_pred = nn_model.predict( y_test )
theta_pred = theta_pred.reshape((theta_pred.shape[0],))


theta_test = data["theta_test"]
print("test", theta_test.shape)
#theta_test = np.float32( theta_test )

print( theta_test.dtype )
print( theta_pred.dtype )

# sqb,cmse = comp_cmse( theta_test, theta_pred )
#r2, mse, mae = eval_pred( theta_test, theta_pred )
sqb, cmse = eval_pred( theta_test, theta_pred )
print("cmse %10.9E" % cmse )
print("sqb  %10.9E" % sqb )



plt.figure(figsize=(10, 6))
plt.scatter(theta_test,theta_pred)
plt.plot(theta_test, theta_test, color='red', linewidth=2)
plt.xlim([0.0,0.01])
plt.ylim([0.0,0.01])
plt.xlabel(r'$\phi_{\text{true}}$',fontsize=20)
plt.ylabel(r'$\phi_{\text{pred}}$',fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
#plt.savefig("prediction-vs-true-2k-delta0d01.pdf", format="pdf", bbox_inches="tight")
#plt.savefig("prediction-vs-true-2k-delta0d1.pdf", format="pdf", bbox_inches="tight")
plt.savefig("prediction-vs-true-4k-delta0d1.pdf", format="pdf", bbox_inches="tight")
plt.show()
