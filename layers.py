import tensorflow as tf
import tensorflow.contrib as tf_contrib
#import matplotlib.pyplot as plt
import numpy as np

weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None
biases_init = tf.zeros_initializer()


##################################################################################
# Activation Function
##################################################################################

def leak_relu(x, leak=0.2, name="leak_relu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)

def relu(x, name="relu"):
    with tf.variable_scope(name):
        return tf.nn.relu(x)

##################################################################################
# Normalization Function
##################################################################################

def instance_norm(x, scope="instance_norm"):

    with tf.variable_scope(scope):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1,2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset

        #out = tf_contrib.layers.instance_norm(x, epsilon=1e-05, center=True, scale=True)
        return out

##################################################################################
# Softmax Function
##################################################################################

def softmax(x):
    orig_shape=x.shape
    if len(x.shape)>1:
        #Matrix
        #shift max whithin each row
        constant_shift=np.max(x,axis=1).reshape(1,-1)
        x-=constant_shift
        x=np.exp(x)
        normlize=np.sum(x,axis=1).reshape(1,-1)
        x/=normlize
    else:
        #vector
        constant_shift=np.max(x)
        x-=constant_shift
        x=np.exp(x)
        normlize=np.sum(x)
        x/=normlize
    assert x.shape==orig_shape
    return x

##################################################################################
# Layer
##################################################################################

def general_conv2d(inputconv, channels=256, kernel=7, stride=1, padding="VALID", name="conv2d", do_norm=False, do_relu=False, relufactor=0):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv = tf.contrib.layers.conv2d(inputs=inputconv, num_outputs=channels, kernel_size=kernel, stride=stride, padding=padding, activation_fn=None, weights_initializer=weight_init, biases_initializer=biases_init)
        if do_norm:
            conv = instance_norm(conv, name+'down_ins_norm')

        if do_relu:
            if(relufactor == 0):
                conv = relu(conv, "relu")
            else:
                conv = leak_relu(conv, relufactor, "leak_relu")

        return conv


def general_deconv2d(inputconv, channels, kernel=7, stride=1, padding="VALID", name="deconv2d", do_norm=False, do_relu=False, relufactor=0):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        deconv = tf.contrib.layers.conv2d_transpose(inputs=inputconv, num_outputs=channels, kernel_size=kernel, stride=stride, padding=padding, activation_fn=None, weights_initializer=weight_init,biases_initializer=biases_init)
        if do_norm:
            deconv = instance_norm(deconv, name+'up_ins_norm')

        if do_relu:
            if(relufactor == 0):
                deconv = relu(deconv, "relu")
            else:
                deconv = leak_relu(deconv, relufactor, "leak_relu")

        return deconv

##################################################################################
# Residual-Block
##################################################################################

def resblock(x_init, channels, pad_size=1, name="resblock"):
    with tf.variable_scope(name):
        with tf.variable_scope("res1"):
            #x = tf.pad(x_init, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "REFLECT")
            x = tf.pad(x_init, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
            x = general_conv2d(x, channels, kernel=3, stride=1, padding="valid", name="rb_conv_1", do_norm=True, do_relu=True)

        with tf.variable_scope("res2"):
            #x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "REFLECT")
            x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
            x = general_conv2d(x, channels, kernel=3, stride=1, padding="valid", name="rb_conv_2", do_norm=True, do_relu=True)

        return x + x_init
