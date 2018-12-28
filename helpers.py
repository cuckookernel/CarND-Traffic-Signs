# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 08:48:41 2018

@author: mrestrepo
"""
import tensorflow as tf
import cv2
import numpy as np

from tensorflow.contrib.layers import flatten, variance_scaling_initializer

# W_INIT_STDDEV = 0.01
#def make_W_v0( shape ) :
#    init = tf.truncated_normal( shape, stddev= W_INIT_STDDEV, dtype=tf.float32 )
#    return tf.Variable( init )

def preproc_ls_1( img ) :
    h,w,_ = img.shape
    X_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS  )

    #h, s, v  = X_hsv[:,:,0], X_hsv[:,:,1], X_hsv[:,:,2]
    l, s  = X_hls[:,:,1], X_hls[:,:,2]

    edges_s = cv2.Canny(s, 100, 200)

    ret = [img, l.reshape((h,w,1)) , s.reshape((h,w,1)), edges_s.reshape((h,w,1)) ]
    return np.concatenate( ret, axis=2 )

def preproc_ls( X ) :
    X_hs   = np.stack( [preproc_ls_1( img ) for img in X], axis=0)
    X_mean = X_hs.mean( axis=(1,2), keepdims=True )
    X_std  = X_hs.std( axis=(1,2),  keepdims=True )
    return  (X_hs - X_mean) / (X_std + 1e-6)

def normalize_mean_std( X ) :
    Xmean = X.mean( axis=(1,2,3), keepdims=True )
    Xstd  = X.std( axis=(1,2,3), keepdims=True )
    return (X - Xmean) / Xstd


def make_W( shape, name ) :
    # Following :
    # https://stackoverflow.com/questions/43489697/tensorflow-weight-initialization

    initial = variance_scaling_initializer(uniform=False, factor=2.0,
                                           mode='FAN_AVG', dtype=tf.float32)
    return tf.get_variable(name, shape=shape, initializer=initial)

def make_b(shape, name=None ) :
    return tf.Variable( tf.zeros(shape, dtype=tf.float32 ), name=name )

def conv_layer( input_, W, b, strides, padding, nonlinear, name=None ) :
    x = tf.nn.conv2d( input_, W, strides, padding=padding )
    x = tf.nn.bias_add( x, b )
    return nonlinear( x, name=name )

def conv_layer_from_pars( input_, pars ) :
    assert pars['type'] == 'conv2d'

    if 'W' in pars :
        print( "Warning: using old version of conv_layer_from_pars")
        W = pars['W']
        b = pars['b']
    else :
        w, h, out_depth = pars['W_pars']

        input_shape = input_.shape.as_list()
        assert len(input_shape) == 4, "Invalid input_shape %s " % (input_shape,)
        in_depth = input_shape[-1]

        W = make_W( (w,h, in_depth, out_depth ), name=pars["name"] + "_W" )
        b = make_b( (out_depth,) )

    n_pars = get_n_pars_4( W ) + get_n_pars_1( b )

    #print( f"input_ : {input_} W={W}" )

    return n_pars, conv_layer( input_,
                       W=W, b=b,
                       strides=pars['strides'],
                       padding=pars.get('padding', 'VALID' ),
                       nonlinear=pars.get('nonlinear', tf.nn.relu ),
                       name=pars['name'] )

def max_pool_from_pars( input_, pars ):
    assert pars['type'] == 'max_pool'
    return 0, tf.nn.max_pool( input_,
                         ksize=pars['ksize'],
                         strides=pars['strides'],
                         padding=pars['padding'],
                         name=pars['name'])

def fully_connected( input_, W, b, nonlinear, name=None ) :
    x = tf.matmul( input_, W )
    x = tf.add( x , b )

    if nonlinear is None :
        print( "no non-linearity" )
        return x
    else :
        return nonlinear( x, name=name )


def fully_connected_from_pars( input_, pars ):
    assert pars['type'] == 'fully_connected'

    if 'shape' in pars :
        print( "Warning: using old api")
        shape = pars['shape']

    else :
        out_dim = pars['out_dim']
        input_shape = input_.shape.as_list()
        assert len(input_shape) == 2, "Invalid input_shape: " + str(input_shape)
        in_dim = input_shape[-1]
        shape = (in_dim, out_dim)

    W = make_W( shape, pars["name"] + "_W" )
    b = make_b( (shape[1],) )

    n_pars = get_n_pars_2( W ) + get_n_pars_1( b )

    return n_pars, fully_connected( input_, W, b, pars["nonlinear"] )


def get_n_pars_2( W ) :
    lst = W.shape.as_list()
    assert len(lst) == 2, f"{lst}"
    return lst[0] * lst[1]

def get_n_pars_4( W ) :
    lst = W.shape.as_list()
    assert len(lst) == 4, f"{lst}"
    return lst[0] * lst[1] * lst[2] * lst[3]

def get_n_pars_1( b ) :
    lst = b.shape.as_list()
    assert len(lst) == 1, f"{lst}"
    return lst[0]


def build_network( input_, params ) :
    prev = input_
    pars_per_layer = []
    layers = []
    for i,pars in enumerate( params ) :
        if pars is None:
            continue

        if pars["type"] == "conv2d" :
            n, layer = conv_layer_from_pars( prev, pars )
        elif pars["type"] == "max_pool" :
            n, layer = max_pool_from_pars( prev, pars )
        elif pars["type"] == "flatten" :
            n, layer = 0, flatten( prev )
        elif pars["type"] == "fully_connected" :
            n, layer = fully_connected_from_pars( prev, pars )
        else:
            assert False, f"Don't know type {pars['type']}"

        print( f"{i:2d} {pars['name']:12s} {pars['type']:15s} "
               f" {str(layer.shape.as_list()[1:]):16s} #params: {n:7d}")
        pars_per_layer.append(n)
        layers.append( layer )

        prev = layer

    print( "Total params: %7d" % sum(pars_per_layer))
    return layers

def batches_generator( X, y, batch_size, verbose=0 ) :
    total_n = X.shape[0]
    assert y.shape[0] == total_n

    n_batches = X.shape[0] // batch_size
    if verbose :
        print( f"Initializing batches_generator: batch_size={batch_size:4d}"
               f" n_batches={n_batches:3d}")

    for i in range( n_batches  ) :
        min_idx = i * batch_size
        max_idx = min( (i+1) * batch_size, total_n )

        yield X[ min_idx:max_idx , ...], y[ min_idx:max_idx ], i


def progress_log( epoch, batch, loss_v, print_loss_every, run_valid_every, accuracy_cb ) :
    if (batch + 1) % print_loss_every == 0 :
                # Calculate batch loss and accuracy
        print('Epoch {:>2}, Batch {:>3} Loss: {:>8.4f}'
                       .format( epoch + 1, batch + 1, loss_v ))

    if (batch + 1) % run_valid_every == 0 :
        valid_accu = accuracy_cb()
        print('Epoch {:>2}, Batch {:>3} Loss: {:>8.4f} Valid. accuracy: {:>8.4f}'
                    .format( epoch + 1, batch + 1, loss_v, valid_accu ))
