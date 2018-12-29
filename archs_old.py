# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 08:56:01 2018

@author: mrestrepo
"""
import helpers as h

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

