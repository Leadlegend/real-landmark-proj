from __future__ import print_function, absolute_import, division

from utils.init import load_data, get_data
from utils.models import IMMModel
from utils.vgg19 import VGG19
import tensorflow as tf

batch_size = 3
shape = (None, 96, None)
IMAGE_SHAPE = (1, 256, 256, 3)
# for VGG19
n_features = 10
learning_rate = 0.001
losses_weight = tf.constant([[100.0], [1.6], [2.3], [1.8], [2.8], [100.0]])
# we use perceptual loss VGG19 model pre-trained for minimize the error for reconstruction


class VGG19Loss(tf.keras.losses.Loss):
	def call(self, y_true, y_pred):
		vgg_true = VGG19(image_shape=IMAGE_SHAPE, input_tensor=y_true)
		vgg_pred = VGG19(image_shape=IMAGE_SHAPE, input_tensor=y_pred)
		loss = tf.matmul(
			[
				tf.reduce_sum(tf.square(y_true - y_pred)),
				tf.reduce_sum(tf.square(vgg_true['conv1_2'] - vgg_pred['conv1-2'])),
				tf.reduce_sum(tf.square(vgg_true['conv2_2'] - vgg_pred['conv2-2'])),
				tf.reduce_sum(tf.square(vgg_true['conv3_2'] - vgg_pred['conv3-2'])),
				tf.reduce_sum(tf.square(vgg_true['conv4_2'] - vgg_pred['conv4-2'])),
			],
			losses_weight
		)
		return loss


def main():
	dataset = load_data(batch_size=batch_size)
	model = IMMModel(shape, n_features)
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	num_batches = int(batch_size)
	for batch_index in range(num_batches):
		img_src, img_trg = get_data(dataset)
		with tf.GradientTape() as tape:
			img_pred = model.train((img_src, img_trg))
			vgg = VGG19Loss()
			# custom loss function using pre-train VGG19 network
			loss = vgg.call(img_trg, img_pred)
			# calculate the error of reconstruction
			print("batch %d:loss %f" % (batch_index, loss.numpy()))
		grads = tape.gradient(loss, model.variables)
		optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


if __name__ == "__main__":
	main()
