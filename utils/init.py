from __future__ import print_function, absolute_import, division

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import tensorflow_datasets as tfds

def load_data(name,batch_size,split="train"):
	snorb_train=tfds.load(name=name,split=split,data_dir="../data/")
	snorb_train=snorb_train.repeat().shuffle(1024).batch(batch_size)
	snorb_train=snorb_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return snorb_train

def get_data(dataset,batch_size,shape=(128,128),n_channels=1):
	x,y=shape[0],shape[1]
	for data in dataset.take(1):
		img1,img2=data["image"],data["image2"]
	return img1,img2

def main():	
	batch_size=5
	smallnorb_dataset=load_data("smallnorb",batch_size,split="train")
	img1,img2=get_data(smallnorb_dataset,batch_size)

if __name__=="__main__":
	main()