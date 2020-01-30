from __future__ import print_function, absolute_import, division

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time
from tensorlayer.layers import (BatchNorm, Conv2d, Dense, Flatten, Input, LocalResponseNorm, MaxPool2d)
from tensorlayer.models import Model

class IMMModel():

	def encoder(self,shape,name):
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

		M=Model(inputs=ni,outputs=nn,name=name)

		return M

	def __init__(self,shape):
		self.maxima_extractor=self.encoder(shape,"maxima_extractor")
		self.original_encoder=self.encoder(shape,"original_encoder")

	def train(self,input):
		self.maxima_extractor.train()
		output=self.maxima_extractor(input)
		return output