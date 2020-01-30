from __future__ import print_function, absolute_import, division

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time
from tensorlayer.layers import (BatchNorm, Conv2d, Dense, Flatten, Input, LocalResponseNorm, MaxPool2d)
from tensorlayer.models import Model

class IMMModel():

	def encoder(self,shape):
		filters=32
		#block_features=[]
		ni=Input(shape,name="input")

		nn=Conv2d(n_filter=filters,filter_size=(7,7),strides=(1,1),name="conv_1",act=tf.nn.relu)(ni)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),name="conv_2",act=tf.nn.relu)(nn)
		#block_features.append(nn)

		filters*=2
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(2,2),name="conv_3",act=tf.nn.relu)(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),name="conv_4",act=tf.nn.relu)(nn)
		#block_features.append(nn)

		filters*=2
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(2,2),name="conv_5",act=tf.nn.relu)(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),name="conv_6",act=tf.nn.relu)(nn)
		#block_features.append(nn)

		filters*=2
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(2,2),name="conv_7",act=tf.nn.relu)(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),name="conv_8",act=tf.nn.relu)(nn)
		#block_features.append(nn)

		return ni,nn

	def get_pose_encoder(self,shape,n_features):
		pose_ni,pose_nn=self.encoder(shape)
		pose_nn=Conv2d(n_filter=n_features,filter_size=(1,1),strides=(1,1),act=None)(pose_nn)

		return Model(inputs=pose_ni,outputs=pose_nn,name="pose_encoder")

	def get_heat_maps(self,pose_nn):

		def get_coordinate(x,other_axis,axis_size):
			prob=tf.reduce_mean(x,axis=other_axis)
			prob=tf.nn.softmax(prob,axis=1)
			coord_axis=tf.reshape(tf.linspace(0.,1.,axis_size),[1,axis_size,1])
			coord=tf.reduce_sum(prob*coord_axis,axis=1)
			return coord,prob

		res_shape=pose_nn.shape
		x_coord,x_coord_prob=get_coordinate(pose_nn,2,res_shape[1])
		y_coord,y_coord_prob=get_coordinate(pose_nn,1,res_shape[2])
		centers=tf.stack([x_coord,y_coord],axis=2)

		def get_gaussian_maps(centers,shape,std=0.1):
			inv_std=1/std
			center_x,center_y=centers[:,:,0:1],centers[:,:,1:2]
			center_x=tf.expand_dims(center_x,axis=-1)
			center_y=tf.expand_dims(center_y,axis=-1)
			x=tf.linspace(0.,1.,shape[0])
			y=tf.linspace(0.,1.,shape[1])
			x=tf.reshape(x,[1,1,shape[0],1])
			y=tf.reshape(y,[1,1,1,shape[1]])

			delta=(tf.square(x-center_x)+tf.square(y-center_y))*inv_std**2

			g_map=tf.exp(-delta)
			g_map=tf.transpose(g_map,[0,2,3,1])

			return g_map

		heat_maps=get_gaussian_maps(centers,(res_shape[1],res_shape[2]))
		return heat_maps


	def __init__(self,shape,n_features):
		image_ni,image_nn=self.encoder(shape)
		self.image_encoder=Model(inputs=image_ni,outputs=image_nn,name="image_encoder")

		self.pose_encoder=self.get_pose_encoder(shape,n_features)

	def train(self,input):
		output=self.pose_encoder(input,is_train=True)
		output=self.get_heat_maps(output)
		return output