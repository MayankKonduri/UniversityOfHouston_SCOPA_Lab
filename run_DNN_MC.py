import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import sys
#from HeatPDE import HeatPDE
from RDPDE import RDPDE
from tensorflow.keras.losses import Loss


class MCDNNLoss(Loss): # creating a custom loss function
    def __init__(self, rd_PDE, u0, lambda_weight=1.0, gamma_inv_sqrt=None, lambda_inv_sqrt=None): # rd pde lets you run PDE forward, u0 is the initial condition, and lambda_weight is your lambda, and (optional) Γ −1 weighting vector
        super().__init__()
        self.rd_PDE = rd_PDE #rd PDE simulator
        self.u0 = u0 #initial condition
        self.lambda_weight = lambda_weight #balances between both the loss componenets
        self.gamma_inv_sqrt = gamma_inv_sqrt  # optional inverse sqrt covariance vector
        self.lambda_inv_sqrt = lambda_inv_sqrt  # for weighting the data loss



    def call(self, y_true_and_theta_true, theta_pred): # theta_true is alpha, theta_pred is alpha hat, y_true is u(t=1)
        #y_true, theta_true = y_true_and_theta_true  # split input tuple into final state and true alpha
        y_true = y_true_and_theta_true[:, :-1]
        theta_true = tf.expand_dims(y_true_and_theta_true[:, -1], axis=-1)

        print("theta_true shape:", theta_true.shape)
        print("theta_pred shape:", theta_pred.shape)


        param_error = theta_true - theta_pred
        if self.gamma_inv_sqrt is not None:
            param_error = param_error * self.gamma_inv_sqrt  # apply weighting
        param_loss = 0.5 * tf.reduce_mean(tf.square(param_error))  # (1/2)|| Γ −1/2 * alpha_true-alpha_predicted||^2

        # 1. define a function that takes in a batch of predicted alpha values,
        # 2. for each one, runs the forward PDE solver to simulate the solution.
        # 3. then extract the final time step (u(t=1)) from each solution.

        # Forward consistency
        def pde_eval(alpha_batch):
            return np.array([self.rd_PDE.fwd_sol(self.u0, a, 0.05)[:, -1] for a in alpha_batch]) #for each predicted alpha, run fwd_sol(), this gives F(alpha hat), the PDE solution
            #[:,-1] extracts only the last time step, also known as u(t=1)


        u_pred = tf.py_function(func=pde_eval, inp=[theta_pred], Tout=tf.float32) # TensorFlow graphs don’t run regular Python/NumPy code.
        u_pred.set_shape(y_true.shape)  # match shape to enable loss computation

        data_error = y_true - u_pred
        if self.lambda_inv_sqrt is not None:
            data_error = data_error * self.lambda_inv_sqrt
        data_loss = 0.5 * tf.reduce_mean(tf.square(data_error))
        # compute the MSE between true final state and simulated final state
        # In formula terms, this is 0.5 * || Λ -1/2 * u(t=1) - F(alpha^) ||^2
        return param_loss + self.lambda_weight * data_loss


path = '/Users/amang/Research/code/pdeml'
chkfn = "chkpts.keras"

from SetupNN import *
from PerfMeasures import *


#heat_PDE = HeatPDE()
#heat_PDE.do_setup(n=128, nt=16)

rd_PDE = RDPDE()
rd_PDE.do_setup(n=128,nt=16)


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



y_train = data["y_train"]
t_train = data["theta_train"]

# Compute gamma^{-1/2} vector
gamma_inv_sqrt = 1.0 / np.sqrt(np.var(t_train, axis=0))
gamma_inv_sqrt = tf.convert_to_tensor(gamma_inv_sqrt, dtype=tf.float32)

# Compute lambda^{-1/2} vector
lambda_inv_sqrt = 1.0 / np.sqrt(np.var(y_train, axis=0))
lambda_inv_sqrt = tf.convert_to_tensor(lambda_inv_sqrt, dtype=tf.float32)


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

u0 = 1.0
custom_loss = MCDNNLoss(rd_PDE=rd_PDE, u0=u0, lambda_weight=1.0, gamma_inv_sqrt=gamma_inv_sqrt, lambda_inv_sqrt=lambda_inv_sqrt)

nn_model.compile(opt, loss=custom_loss)
nn_model.summary()

model_callback = tf.keras.callbacks.ModelCheckpoint( filepath= path +"/chkpts/" + chkfn,
    save_weights_only=False,
    #monitor='val_loss',
    monitor='loss',
    mode='min',
#    period=period,
    save_best_only=False, save_freq='epoch')


print("number of training data", y_train.shape[0])
print("number of features     ", y_train.shape[1])
#n_data = 1000
#y_train = y_train[:n_data]
#t_train = t_train[:n_data]

y_combined = np.concatenate([y_train, t_train.reshape(-1, 1)], axis=1)

tf.print("y_train shape:", tf.shape(y_train))
tf.print("t_train:", tf.shape(t_train))
tf.print("y_combined:", tf.shape(y_combined))

# train neural network
nn_model.fit( y_train, y_combined, epochs=n_epoch, batch_size=batch, callbacks=model_callback)

# train neural network
#nn_model.fit( y_train, (y_train, t_train), epochs=n_epoch, batch_size=batch, callbacks=model_callback)

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