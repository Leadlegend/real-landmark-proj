from __future__ import print_function, absolute_import, division

from utils.init import load_data,get_data
from utils.models import IMMModel
import tensorflow as tf
import tensorlayer as tl

batch_size=3
shape=(None,96,96,None)
n_features=10

def main():
	dataset=load_data(batch_size=batch_size)
	model=IMMModel(shape,n_features)
	img1,img2=get_data(dataset)

	img=model.train((img1,img2))
	print(img)

if __name__=="__main__":
	main()