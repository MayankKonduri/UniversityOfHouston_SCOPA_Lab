import numpy as np
import tensorflow as tf


def eval_pred( theta_true, theta_pred ):

    sqb, cmse = comp_cmse( theta_true, theta_pred )
#     r2 = compute_r2( theta_true, theta_pred )
#     mse = compute_mse( theta_true, theta_pred )
#     mae = compute_mae( theta_true, theta_pred )
    return sqb, cmse




# def compute_mse( theta_true, theta_pred ):
# #    mse = tf.metrics.mean_squared_error( theta_true, theta_pred )
# #    print(mse)
#     return 0.0
#
#
# def compute_mae( theta_true, theta_pred ):
# #    mae = tf.metrics.mean_absolute_error( theta_true, theta_pred )
# #    print(mae)
#     return 0.0
#
#
# def compute_r2( theta_true, theta_pred):
#     '''
#     Computes R^2 metric:
#         R^2 = 1 - mean( (y* - y^)^2 ) / mean( (y* - mean(y*)^2) )
#     where
#         y* are arrays of labels
#         y^ are arrays of predictions
#     '''
#     unexplained_error = tf.math.reduce_sum(tf.math.square(
#             tf.math.subtract( theta_true, theta_pred) ))
#     total_error = tf.math.reduce_sum(tf.math.square(
#             tf.math.subtract( theta_true, tf.math.reduce_mean(theta_pred)) ))
#     r2 = tf.math.subtract(tf.constant(1.0, dtype=theta_true.dtype), tf.math.divide(unexplained_error, total_error))
#     return r2


def comp_cmse( theta_true, theta_pred ):
    ''''
    Returns squared bias and CMSE for each parameter
    '''

    p = theta_true.ndim

    sqb = 1e6
    cmse = 1e6
    if p == 1:
        ns = theta_true.shape[0]

        t_true = np.copy( theta_true )
        t_pred = np.copy( theta_pred )

        true_mean = np.mean( t_true )
        pred_mean = np.mean( t_pred )

        sqb = np.subtract( true_mean, pred_mean )**2

        delta1 = np.subtract( t_true, true_mean )
        delta2 = np.subtract( t_pred, pred_mean )

        delta = np.subtract( delta1, delta2 )

        cmse = np.sum( delta**2 ) / float(ns)

    return sqb, cmse


# ###################################################
# def comp_r2(param_true, param_pred):
#     # Data must be numpy arrays
#     #assert they have the same shape
#     M = len(param_true)
#
#     #compute mean of true parameters
#     true_mean = np.mean(param_true,axis=0)
#     #print("true mean shape, ", true_mean.shape)
#     #print("true mean shape, ", true_mean[0])
#
#     #difference between predicted and true parameters
#     v = abs(np.subtract(param_true, param_pred))
#     #print('v max: ', np.max(v,axis=0))
#     #print('v min: ', np.min(v,axis=0))
#
#     #numerator of fraction
#     num = np.sum(v**2,axis=0)
#
#     temp3 = np.divide(v, abs(param_true))
#     #print('mape max: ', np.max(temp3,axis=0))
#     #print('mape min: ', np.min(temp3,axis=0))
#
#     #take median along each colum
#     MAPE = np.median(temp3,axis=0)
#
#     #For denominator
#     k=len(param_true[0]) #4 in this case
#
#     temp = np.zeros_like(param_true)
#     for i in range(k):
#         temp[:,i]=param_true[:,i]-true_mean[i]
#     denom = np.sum(temp**2,axis=0)
#     #print("Frac shape: ", np.divide(num,denom))
#
#     r2 = 1 - np.divide(num,denom)
#     #print(MAPE)
#
#     return r2, MAPE
