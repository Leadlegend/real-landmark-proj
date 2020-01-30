from __future__ import print_function, absolute_import, division

from utils.init import load_data,get_data
from utils.models import IMMModel

batch_size=50
shape=(batch_size,96,96,1)
def main():
	dataset=load_data()
	model=IMMModel(shape)
	img1,img2=get_data(dataset)
	img3=model.train(img1)
	print(img3)

if __name__=="__main__":
	main()