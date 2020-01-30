from __future__ import print_function, absolute_import, division

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import tensorflow_datasets as tfds

def load_data(name="smallnorb",batch_size=50,split="train"):
	snorb_train=tfds.load(name=name,split=split,data_dir="data/")
	snorb_train=snorb_train.repeat().shuffle(1024).batch(batch_size)
	snorb_train=snorb_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return snorb_train

def get_data(dataset,batch_size=50):
	for data in dataset.take(1):
		img1=tf.cast(data["image"],dtype=tf.float32)/255.0
		img2=tf.cast(data["image2"],dtype=tf.float32)/255.0
	return img1,img2

def main():	
	batch_size=50
	smallnorb_dataset=load_data(name="smallnorb",batch_size=batch_size,split="train")
	img1,img2=get_data(smallnorb_dataset,batch_size)
	print(img1)

if __name__=="__main__":
	main()