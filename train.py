import argparse
from os import listdir
from numpy import asarray
from numpy import vstack

from tensorflow import keras

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from numpy import savez_compressed
from keras.models import load_model
import tensorflow as tf

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model

from tensorflow.keras import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot

from pathlib import Path
import os
import sys

import yaml

from utils.general import (
	LOGGER,
	check_file,
	check_yaml,
	colorstr,
	increment_path,
	print_args,
)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
#GIT_INFO = check_git_info()

# gpus = tf.config.experimental.list_physical_devices('GPU')    
# for gpu in gpus:
# 	tf.config.experimental.set_memory_growth(gpu, True)
# 	tf.config.experimental.set_virtual_device_configuration(
# 		gpu,
# 		[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) # Notice here

def load_images(path, size=(512,1024)):
	src_list = list()
	tar_list = list()
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + '/' + filename, target_size=size) # images are in PIL formate
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into colored and sketch. 256 comes from 512/2. The first part is colored while the rest is sketch
		color_img, bw_img = pixels[:, :512], pixels[:, 512:]
		src_list.append(bw_img)
		tar_list.append(color_img)
		
	return [asarray(src_list), asarray(tar_list)]

def parse_opt(known=False):
	parser =argparse.ArgumentParser()

	parser.add_argument("--batch-size", type=int, default=16, help="total batch size fo all GPUS")
	parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
	parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
	parser.add_argument("--data", type=str, default=ROOT / "data/ascii.yaml", help="dataset.yaml path")
	parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
	parser.add_argument("--name", default="exp", help="save to project/name")
		
	return parser.parse_known_args()[0] if known else parser.parse_args()

def plot_images(imgs, n_samples=3, col=0):
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i + col)
		pyplot.axis('off')
		pyplot.imshow(imgs[i].astype('uint8'))

# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(256,256,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, plotsave_dir, modelsave_dir, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	filename1 = plotsave_dir / filename1
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	filename2 = modelsave_dir / filename2
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

def train(d_model, g_model, gan_model, opt):

	save_dir, epochs, batch_size, dataset, plotresults_path = (
		Path(opt.save_dir),
		opt.epochs,
		opt.batch_size,
		opt.dataset,
		opt.plotresults_path
	)

	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / batch_size)
	# calculate the number of training iterations
	n_steps = bat_per_epo * epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, batch_size, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss), flush=True)
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset, opt)

	summarize_performance(n_steps, g_model, dataset=dataset, plotsave_dir=plotresults_path, modelsave_dir=save_dir)
	g_model.save(save_dir / 'model_final.h5')


def preprocess_data(opt):
	print(f'Process data out of {opt.data} into a zip file.')

def main(opt):
	
	#if RANK in {-1, 0}:
		#print('args')
		#print_args(vars(opt))
		#check_git_status()
		#check_requirements(ROOT / "requirements.txt")
	
	opt.data, opt.project = (
		check_file(opt.data),
		str(opt.project),
	)  # checks    	
	
	opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
	
	with open(opt.data, errors="ignore") as f:
		dat = yaml.safe_load(f)
		opt.data_path = dat['path']


	print(opt.data_path)
	#path = opt.data_directory

	print('Loading images from ', opt.data_path, flush=True)

	# load dataset
	[src_images, tar_images] = load_images(opt.data_path)
	print(f'Loaded: {src_images.shape}, {tar_images.shape}', flush=True)

	r = Path(opt.save_dir) / "plot_results"	# results directory
	r.mkdir(parents=True, exist_ok=True)  # make dir

	opt.plotresults_path = r

	filename = Path(opt.save_dir) / 'gan_train.npz'

	print(f'Compressing data set to {filename}.  This will take some time.  Please wait.', flush=True)
	# save as compressed numpy array
	savez_compressed(filename, src_images, tar_images)
	print(f'Saved compressed dataset: {filename}', flush=True)

	opt.compressed_dataset = filename

	src_images = None
	tar_images = None

	input("Press Enter to continue...")
	

	#data = load(filename)
	#src_images, tar_images = data['arr_0'], data['arr_1']
	#print(f'Loaded: {src_images.shape}, {tar_images.shape}', flush=True)    

	# # load image data
	dataset = load_real_samples(filename)
	opt.dataset = dataset

	# define input shape based on the loaded dataset
	image_shape = dataset[0].shape[1:]
	# define the models
	d_model = define_discriminator(image_shape)
	g_model = define_generator(image_shape)
	# define the composite model
	gan_model = define_gan(g_model, d_model, image_shape)	

	print(f'Training model with {opt.epochs} epochs and a batch size of {opt.batch_size}\nResults can be found at {opt.save_dir}', flush=True)

	#preprocess_data(opt)
	train(d_model, g_model, gan_model, opt)

	# train model
	#train(d_model, g_model, gan_model, dataset, opt.epochs, opt.batch_size)
	# plot_images(src_images, n_samples=3)
	# plot_images(tar_images, n_samples=3, col=3)

	#gan_model.save('model/model.h5')
	
	# pyplot.show()
	#load_model()


if __name__ == "__main__":
	opt = parse_opt()
	main(opt)
