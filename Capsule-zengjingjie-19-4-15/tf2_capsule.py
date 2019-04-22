#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import time
import pydot
import graphviz


# In[2]:


print("python", sys.version)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


# In[3]:


def squash(x,axis=-1):
    s_suqared_norm = K.epsilon()+tf.reduce_sum(tf.square(x),axis=axis,keepdims=True)
#     print (s_suqared_norm)
    scale = tf.sqrt(s_suqared_norm)/(1.+s_suqared_norm)
#     print(scale)
    return scale*x


# In[4]:


def softmax(x,axis=-1):
    ex = tf.exp(x - tf.reduce_max(x,axis=axis,keepdims=True))
    return ex/tf.reduce_sum(ex,axis=axis,keepdims=True)


# In[137]:


class Con1D_to_Capsule(keras.layers.Layer):
    def __init__(self, out_num_capsule,out_dim_capusle,routings=3,activation="squash", **kwargs):
        self.out_num_capsule = out_num_capsule
        self.out_dim_capusle = out_dim_capusle
        self.routings = routings
        self.activation = squash
        super(Con1D_to_Capsule, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.W_kernel = self.add_weight(name='capsule_kernel', 
                                      shape=(1,input_shape[-2], self.out_num_capsule * self.out_dim_capusle),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(Con1D_to_Capsule, self).build(input_shape)  

    def call(self, u_ves):
        print(self.W_kernel.shape)
        print("*****",u_ves.shape)
        u_ves = tf.transpose(u_ves,perm=[0,2,1])
        
        print("*****",u_ves.shape)
        u_hat_vecs = K.conv1d(u_ves,self.W_kernel)
        print("*****",u_hat_vecs.shape)
        batch_size = tf.shape(u_ves)[0]
        input_num_capsule = tf.shape(u_ves)[1]
        u_hat_vecs = tf.reshape(u_hat_vecs,(batch_size,input_num_capsule,self.out_num_capsule,self.out_dim_capusle))
        u_hat_vecs = tf.transpose(u_hat_vecs,perm=[0,2,1,3])  # finally shape = [N0ne,out_num_capsule,input_num_capsule,out_dim_capsule]
        
        
        # Dynamic routing
        b = tf.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [N0ne,out_num_capsule,input_num_capsule]
        for i in range(self.routings):
            c = softmax(b,1)
            output = K.batch_dot(c,u_hat_vecs,[2,2])
            output = self.activation(output)
            if i < self.routings - 1:
#                 o = tf.nn.l2_normalize(o,-1)
                b = b + K.batch_dot(output,u_hat_vecs,[2,3])
        pose = output
        print("pose is :",pose.shape)
        return pose

    def compute_output_shape(self, input_shape):
        return (None,self.num_capsule, self.dim_capusle)



class Capsule(keras.layers.Layer):
    def __init__(self, out_num_capsule,out_dim_capusle,routings=3,activation="squash", **kwargs):
        self.out_num_capsule = out_num_capsule
        self.out_dim_capusle = out_dim_capusle
        self.routings = routings
        self.activation = squash
        super(Capsule, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.W_kernel = self.add_weight(name='capsule_kernel', 
                                      shape=(1,input_shape[-1], self.out_num_capsule * self.out_dim_capusle),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(Capsule, self).build(input_shape)  

    def call(self, u_ves):
        
        u_hat_vecs = K.conv1d(u_ves,self.W_kernel)
        batch_size = tf.shape(u_ves)[0]
        input_num_capsule = tf.shape(u_ves)[1]
        u_hat_vecs = tf.reshape(u_hat_vecs,(batch_size,input_num_capsule,self.out_num_capsule,self.out_dim_capusle))
        u_hat_vecs = tf.transpose(u_hat_vecs,perm=[0,2,1,3])  # finally shape = [N0ne,out_num_capsule,input_num_capsule,out_dim_capsule]
        
        
        
        # Dynamic routing
        b = tf.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [N0ne,out_num_capsule,input_num_capsule]
        for i in range(self.routings):
            c = softmax(b,1)
            output = K.batch_dot(c,u_hat_vecs,[2,2])
            output = self.activation(output)
            if i < self.routings - 1:
#                 o = tf.nn.l2_normalize(o,-1)
                b = b + K.batch_dot(output,u_hat_vecs,[2,3])
        pose = output
        return pose

    def compute_output_shape(self, input_shape):
        return (None,self.num_capsule, self.dim_capusle)




