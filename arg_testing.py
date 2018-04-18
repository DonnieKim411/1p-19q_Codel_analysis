import os
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

import argparse

class build_network():
	'''
	class that prepares for transfer learning + fine-turning using the pre-trained network and running the transfer learn
	'''

	def __init__(self, args, n_epochs, n_batchs):
		'''
		param args: argparse inputs
		param n_epochs: number of epochs to train the model. Pre-set with 40
		param n_batch: batch size to train with. Pre-set with 64
		'''

		self.args = args
		self.n_epochs = n_epochs
		self.n_batchs = n_batchs

	def prepare_data(self, train_dir, val_dir):
		'''
		param train_img: train img in size of 142 by 142
		param val_img: val img in size of 142 by 142
		param train_1p19q: train 1p19q label. either 0 or 1
		param val_1p19q: val 1p19q label. either 0 or 1
		'''
		train_img = np.load(train_dir + 'train_' + args.modality_name + '.npy')
		train_1p19q = np.load(train_dir + 'train_1p19q.npy') # 27 1s, 273 0s.

		val_img = np.load(val_dir + 'val_' + args.modality_name + '.npy')
		val_1p19q = np.load(val_dir + 'val_1p19q.npy')


		# Augment train data via imagedatagenerator
		train_datagen =  ImageDataGenerator( # z-score normalization has little effect, hence dropped
			# featurewise_center = True,
			# featurewise_std_normalization = True,
			rotation_range=30,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			vertical_flip=True
		)

		# compute quantities required for featurewise normalization such as mean, std, etc. probably not needed but double check
		train_datagen.fit(train_img)

		val_datagen = ImageDataGenerator()

		train_generator = train_datagen.flow(
			x = train_img,
			y = train_1p19q,
			shuffle = True,
			batch_size = self.n_epochs
			# save_to_dir = train_aug_dir
		)

		val_generator = val_datagen.flow(
			x = val_img,
			y = val_1p19q,
			shuffle = True,
			batch_size = self.n_epochs
			# save_to_dir = val_aug_dir
		)
		return train_generator, val_generator

	def transfer_finetune_setup(self, pre_trained_model_dir):
		# load model
		pre_trained_model = load_model(pre_trained_model_dir+ 'args.modality_name' + '_model.h5')

		# remove the last layaer
		pre_trained_model.pop()
		last_layer = pre_trained_model.layers[-1]
		x = last_layer.output
		# add dropout
		x = Dropout(0.7)(x)
		# add fully connected layer with sigmoid activatioin
		x = Dense(nb_classes, activation = 'sigmoid')(x)
		model = Model(input=pre_trained_model.input, output=x)

		return model
		
# modify last layer
def update_the_last_layer(base_model, nb_classes):
	# remove the FC layer of the base model
	base_model.layers.pop()

	last_layer = base_model.layers[-1]
	x = last_layer.output

	x = Dropout(0.7)(x)
	x = Dense(nb_classes, activation = 'sigmoid')(x)
	
	model = Model(input=base_model.input, output=x)
	return model

# By default, using Adam
def setup_optimizer(): 
	learning_rate = 0.0 #learning rate can be tuned with step_decay function
	beta_1 = 0.9 #d_w: a momentum like term
	beta_2 = 0.999 # d_w^2 
	epsilon = 1e-8 # no need to tune
	Adam = optimizers.Adam(lr = learning_rate, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon)
	return Adam

def setup_to_transfer_learn(model, base_model, optimizer):
	"""Freeze all layers and compile the model"""
	for layer in base_model.layers:
		layer.trainable = False

	model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

def setup_to_finetune(model, optimizer):
	"""Freeze all the layers but top 1 conv layers. 8 layers can be trained in this case
	
	Args:
		model: keras model
		optimizer: keras optimizer
	"""
	for layer_ind in range(-3,-12,-1):
		model.layers[layer_ind].trainable = True

	model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])



if __name__="__main":
	parser = argparse.ArgumentParser(description = "Run ResNet-34 transfer learning for 1p/19q classification")

	parser.add_argument('--gpus_vis', nargs='?', metavar='gpus_visible', help = "GPU number that is present to the model. Default is 0.", 
						type = str, default = "0", required = False)

	parser.add_argument('--modality_name', type = str, help = "modality name: T1, T1post, T2, or FLAIR", required = True)

	parser.add_argument("--model_load", type = str, 
						help = "Provide the name of the pre-trained model to load. EX) if calling pretrained T2 model, then /path/to/T2/model/T2_model.h5", required = True)

	parser.add_argument("--model_save", type = str, 
						help = "Provide the name of the model to save. EX) If training T2 model, then /path/to/model/to/save/T2_transfer_learned.h5", required = True)

	# parser.add_argument("--model_params", type = str, help = "Path to numpy file containing dictionary that has parameters for the model", required = True)
	
	args = parser.parse_args()

	n_epochs = 40
	n_batchs = 64

	# Training and validation data directory
	train_dir = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced/train/'
	val_dir = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced/val/'

	# pre-trained model directory
	pre_trained_model_dir = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/1p_19q/glioma_models';

	args = parser.parse_args()
	if args.modality_name is None or args.model_save is None:
	parser.print_help()
	sys.exit(1)

	net_build = build_network(args, n_epochs, n_batchs)
	train_generator, val_generator = net_build.prepare_data(train_dir, val_dir)
	
	net_build.transfer_finetune_setup(pre_trained_model_dir)


