from __future__ import print_function, absolute_import, division

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time
from tensorlayer.layers import BatchNorm, Conv2d, Conv3d, Dense, Flatten, Input, UpSampling2d
from tensorlayer.models import Model

class IMMModel():

	def encoder(self,shape):
		filters=32
		#block_features=[]
		ni=Input(shape,name="input")
		W_init=tl.initializers.truncated_normal(stddev=1e-2)
		W_init_2=tl.initializers.truncated_normal(stddev=1e-2)

		nn=Conv2d(n_filter=filters,filter_size=(7,7),strides=(1,1),
			act=tf.nn.relu,W_init=W_init,padding="SAME")(ni)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init_2,padding="SAME")(nn)
		#block_features.append(nn)

		filters*=2
		W_init=tl.initializers.truncated_normal(stddev=1e-2)
		W_init_2=tl.initializers.truncated_normal(stddev=1e-2)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(2,2),
			act=tf.nn.relu,W_init=W_init,padding="SAME")(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init_2,padding="SAME")(nn)
		#block_features.append(nn)

		filters*=2
		W_init=tl.initializers.truncated_normal(stddev=1e-2)
		W_init_2=tl.initializers.truncated_normal(stddev=1e-2)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(2,2),
			act=tf.nn.relu,W_init=W_init,padding="SAME")(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init_2,padding="SAME")(nn)
		#block_features.append(nn)

		filters*=2
		W_init=tl.initializers.truncated_normal(stddev=1e-2)
		W_init_2=tl.initializers.truncated_normal(stddev=1e-2)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(2,2),
			act=tf.nn.relu,W_init=W_init,padding="SAME")(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init_2,padding="SAME")(nn)
		#block_features.append(nn)

		return ni,nn

	def get_pose_encoder(self,shape,n_features):
		pose_ni,pose_nn=self.encoder(shape)
		W_init=tl.initializers.truncated_normal(stddev=1e-2)

		pose_nn=Conv2d(n_filter=n_features,filter_size=(1,1),strides=(1,1),
			act=None,W_init=W_init,padding="SAME")(pose_nn)

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

	def decoder(self,shape):
		ni=Input(shape,name="input")
		filters=256
		W_init=tl.initializers.truncated_normal(stddev=1e-2)
		W_init_2=tl.initializers.truncated_normal(stddev=1e-2)

		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init,padding="SAME")(ni)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init_2,padding="SAME")(nn)

		filters//=2
		W_init=tl.initializers.truncated_normal(stddev=1e-2)
		W_init_2=tl.initializers.truncated_normal(stddev=1e-2)
		nn=UpSampling2d(scale=(2,2),method="bilinear")(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init,padding="SAME")(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init_2,padding="SAME")(nn)

		filters//=2
		W_init=tl.initializers.truncated_normal(stddev=1e-2)
		W_init_2=tl.initializers.truncated_normal(stddev=1e-2)
		nn=UpSampling2d(scale=(2,2),method="bilinear")(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init,padding="SAME")(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init_2,padding="SAME")(nn)

		filters//=2
		W_init=tl.initializers.truncated_normal(stddev=1e-2)
		W_init_2=tl.initializers.truncated_normal(stddev=1e-2)
		nn=UpSampling2d(scale=(2,2),method="bilinear")(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init,padding="SAME")(nn)
		nn=Conv2d(n_filter=filters,filter_size=(3,3),strides=(1,1),
			act=tf.nn.relu,W_init=W_init_2,padding="SAME")(nn)

		return ni,nn


	def __init__(self,shape,n_features):
		image_ni,image_nn=self.encoder(shape)
		self.image_encoder=Model(inputs=image_ni,outputs=image_nn,name="image_encoder")

		self.pose_encoder=self.get_pose_encoder(shape,n_features)

		decoder_ni,decoder_nn=self.decoder((None,shape[1]//8,shape[2]//8,None))
		self.image_decoder=Model(inputs=decoder_ni,outputs=decoder_nn,name="image_decoder")

	def train(self,input):
		img1,img2=input[0],input[1]

		out1=self.image_encoder(img1,is_train=True)

		out2=self.pose_encoder(img2,is_train=True)
		out2=self.get_heat_maps(out2)

		combined_input=tf.concat([out1,out2],axis=3)
		combined_output=self.image_decoder(combined_input,is_train=True)
		return combined_output