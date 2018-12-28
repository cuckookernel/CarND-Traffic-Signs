# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:36:00 2018

@author: mrestrepo
"""
import helpers as h
import tensorflow as tf


def arch_lenet_8_oldapi() :
    layer0_depth = 3
    layer1_depth = 8
    layer3_depth = 16

    return [ None, # index=0 won't be used
    # layer 1 : conv2d
    { 'type'    : 'conv2d',
      'W'       : h.make_W( (5, 5, layer0_depth, layer1_depth), "conv1_W" ),
      'strides' : ( 1, 1, 1, 1),
      'b'       : h.make_b( (layer1_depth, ) ),
      'name'    : 'conv1'},
    # layer 2 : max pool
    {  'type'    : 'max_pool',
       'ksize'   : (1, 2, 2, 1),
       'strides' : (1, 2, 2, 1),
       'padding' : 'SAME',
       'name'    : 'max_p1'  },

     # layer 3 : conv2d
    { 'type'    : 'conv2d',
      'W'       : h.make_W( (5, 5, layer1_depth, layer3_depth), "conv2_W" ),
      'strides' : ( 1, 1, 1, 1),
      'b'       : h.make_b( (layer3_depth, ) ),
      'name'    : 'conv2'},

    # layer 4 : max_pool
    {  'type'    : 'max_pool',
       'ksize'   : (1, 2, 2, 1),
       'strides' : (1, 2, 2, 1),
       'padding' : 'SAME',
       'name'    : 'max_p2'  },

     # layer 5 : flatten
    {  'type'    : 'flatten',
       'name'    : 'flat'},

    #layer 6 : fully_connected
    {  'type'    : 'fully_connected',
       'shape'   : (400,120),
       #'nonlinear' : lambda x : 1.7159 * tf.nn.tanh(x), #tf.nn.relu,
       'nonlinear' : tf.nn.relu,
       'name'    : 'fc1'},

    #layer 7 : fully_connected
    {  'type'    : 'fully_connected',
       'shape' : (120,84),
       #'nonlinear' : lambda x : 1.7159 * tf.nn.tanh(x), #tf.nn.relu,
       'nonlinear' : tf.nn.relu,
       'name' : 'fc2'},

    #layer 8 : fully_connected  - no relu afterwards
    {  'type'    : 'fully_connected',
       'shape' : (84,43),
       'nonlinear' : None,
       'name' : 'fc3' }
    ]


layer1_depth = 8
layer3_depth = 16

arch_lenet_8 = [ None, # index=0 won't be used
    # layer 1 : conv2d
    { 'type'  : 'conv2d', 'W_pars' : ( 5, 5, 8),  'strides' : ( 1, 1, 1, 1),
      'name'  : 'conv1'},
    # layer 2 : max pool
    {  'type'    : 'max_pool', 'ksize'   : (1, 2, 2, 1), 'strides' : (1, 2, 2, 1),
       'padding' : 'SAME', 'name'    : 'max_p1'  },
     # layer 3 : conv2d
    { 'type'    : 'conv2d', 'W_pars'  : (5, 5, layer3_depth), 'strides' : ( 1, 1, 1, 1),
      'name'  : 'conv2'},
    # layer 4 : max_pool
    {  'type'    : 'max_pool', 'ksize' : (1, 2, 2, 1), 'strides' : (1, 2, 2, 1),
       'padding' : 'SAME',   'name'  : 'max_p2'  },
     # layer 5 : flatten
    {  'type'    : 'flatten', 'name'    : 'flat1'},
    #layer 6 : fully_connected
    {  'type'    : 'fully_connected', 'out_dim' : 120, 'nonlinear' : tf.nn.relu, 'name'    : 'fc1'},
    #layer 7 : fully_connected
    {  'type'    : 'fully_connected', 'out_dim' : 84,  'nonlinear' : tf.nn.relu, 'name' : 'fc2'},
    #layer 8 : fully_connected  - no relu afterwards
    {  'type'    : 'fully_connected', 'out_dim' : 43,  'nonlinear' : None,   'name' : 'fc3' }
    ]

# The following yielded 93.17 % with depth-3 images Adam( lr=0.0005 )  batch_size=256, epoch=168
arch_3_3 = [ None, # index=0 won't be used
    { 'type'  : 'conv2d', 'W_pars' : ( 3, 3, 16),  'strides' : ( 1, 1, 1, 1),
      'name'  : 'conv1'},

    {  'type'    : 'max_pool', 'ksize'   : (1, 2, 2, 1), 'strides' : (1, 2, 2, 1),
       'padding' : 'SAME', 'name'    : 'max_p1'  },

    { 'type'  : 'conv2d', 'W_pars' : ( 3, 3, 32),  'strides' : ( 1, 1, 1, 1),
      'name'  : 'conv2'},

    { 'type'    : 'max_pool', 'ksize'   : (1, 2, 2, 1), 'strides' : (1, 2, 2, 1),
      'padding' : 'SAME', 'name'    : 'max_p2'  },

    { 'type'  : 'conv2d', 'W_pars' : ( 5, 5, 16),  'strides' : ( 1, 1, 1, 1),
      'name'  : 'conv3'},

    # layer 4 : max_pool
    {  'type'    : 'max_pool', 'ksize' : (1, 2, 2, 1), 'strides' : (1, 2, 2, 1),
       'padding' : 'SAME',   'name'  : 'max_p3'  },
     # layer 5 : flatten
    {  'type'    : 'flatten', 'name'    : 'flat1'},
    #layer 6 : fully_connected
    {  'type'    : 'fully_connected', 'out_dim' : 120, 'nonlinear' : tf.nn.relu, 'name'    : 'fc1'},
    #layer 7 : fully_connected
    {  'type'    : 'fully_connected', 'out_dim' : 84,  'nonlinear' : tf.nn.relu, 'name' : 'fc2'},
    #layer 8 : fully_connected  - no relu afterwards
    {  'type'    : 'fully_connected', 'out_dim' : 43,  'nonlinear' : None,   'name' : 'fc3' }
    ]



arch_3_3_b = [ None, # index=0 won't be used
    { 'type'  : 'conv2d', 'W_pars' : ( 3, 3, 32),  'strides' : ( 1, 1, 1, 1),
      'name'  : 'conv1'},

    {  'type'    : 'max_pool', 'ksize'   : (1, 2, 2, 1), 'strides' : (1, 2, 2, 1),
       'padding' : 'SAME', 'name'    : 'max_p1'  },

    { 'type'  : 'conv2d', 'W_pars' : ( 3, 3, 32),  'strides' : ( 1, 1, 1, 1),
      'name'  : 'conv2'},

    { 'type'    : 'max_pool', 'ksize'   : (1, 2, 2, 1), 'strides' : (1, 2, 2, 1),
      'padding' : 'SAME', 'name'    : 'max_p2'  },

    { 'type'  : 'conv2d', 'W_pars' : ( 5, 5, 16),  'strides' : ( 1, 1, 1, 1),
      'name'  : 'conv3'},

    # layer 4 : max_pool
    #{  'type'    : 'max_pool', 'ksize' : (1, 2, 2, 1), 'strides' : (1, 2, 2, 1),
    #   'padding' : 'SAME',   'name'  : 'max_p3'  },
     # layer 5 : flatten
    {  'type'    : 'flatten', 'name'    : 'flat1'},
    #layer 6 : fully_connected
    {  'type'    : 'fully_connected', 'out_dim' : 120, 'nonlinear' : tf.nn.relu, 'name'    : 'fc1'},
    #layer 7 : fully_connected
    {  'type'    : 'fully_connected', 'out_dim' : 84,  'nonlinear' : tf.nn.relu, 'name' : 'fc2'},
    #layer 8 : fully_connected  - no relu afterwards
    {  'type'    : 'fully_connected', 'out_dim' : 43,  'nonlinear' : None,   'name' : 'fc3' }
    ]



