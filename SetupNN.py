import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def load_nndata( filename, n_test, n_train ):
    #training sizes
#    n_train_list = [500,1000,2000,4000,8000]
#    n_test = int(n_s*0.2)
#    n_train = n_s - n_test

    # load samples
    npzfile = np.load( filename )

    # extract time series data and associated parameters
    y = npzfile['y']
    theta = npzfile['theta']

    nx = y.shape[1] - 1
    ns = y.shape[0]

    # load output of NN (model parameters theta)
    theta_test = np.copy(theta[0:n_test])
    theta_train = np.copy(theta[n_test:n_test+n_train])

    # load input of nn (observations y)
    y_test = np.copy(y[0:n_test])
    y_train = np.copy(y[n_test:n_test+n_train])

    data = {"y_test": y_test, "y_train": y_train,
            "theta_test": theta_test, "theta_train": theta_train}

    return data


def build_dnn( nn_para ):
    n_layers = nn_para['n_layers'] # number of layers
    n_units = nn_para['n_units'] # number of units

    # default parameters
    ker_init = 'glorot_uniform'

    i_shape = nn_para['y_shape'][1]
    x_shape = nn_para['t_shape']

    print("input shape", i_shape)
    print("output shape", x_shape)

    final_actfctn = nn_para['final_actfctn']

    inputs = keras.Input( shape=(i_shape,), name="time_series" )
    print(inputs.shape)

    flat = layers.Flatten()( inputs )
    x = layers.Dense( n_units, kernel_initializer=ker_init, name="dense_1" )( flat )
    x = layers.Activation("swish")(x)

    # creating hidden layers
    for i in range(0,n_layers-1):
        x = layers.Dense( n_units, kernel_initializer = ker_init, name="dense_"+str(i+1) )(x)
        x = layers.Activation("swish")(x)

    # set up final layer
    x = layers.Dense( x_shape, kernel_initializer=ker_init )(x)
    outputs = layers.Activation( final_actfctn )(x)
    model = keras.Model( inputs=inputs, outputs=outputs )

    return model


def eval_mse_loss(y_true, y_pred):
    # loss function
    val = (y_true - y_pred)**2
    return val
