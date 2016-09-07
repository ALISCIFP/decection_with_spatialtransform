import numpy as np
import cv2
from PIL import Image     
import scipy.io as sio

from os import listdir
from os.path import isfile, join
def get_datalabel(mypath):
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	n_files = len(onlyfiles)
	print(n_files)

	file = mypath + '/' + onlyfiles[0]
	struct = sio.loadmat(file)
	data = struct['affNISTdata']['image']
	label = struct['affNISTdata']['label_int']
	batch_size = data[0][0].shape[1]
	x = np.zeros((batch_size * n_files,data[0][0].shape[0]),dtype = 'uint8')
	y = np.zeros((batch_size * n_files,label[0][0].shape[0]),dtype = 'uint8')
	i = 0
	for file_name in onlyfiles:
		file = mypath + '/' + file_name
		struct = sio.loadmat(file)
		data = struct['affNISTdata']['image']
		label = struct['affNISTdata']['label_int']
		x[i*batch_size:(i+1)*batch_size,:] = np.transpose(data[0][0])
		y[i*batch_size:(i+1)*batch_size,:] = np.transpose(label[0][0])
		i = i + 1

	return x,y

train_path = '/home/menglin/tensorflow_models/models/transformer/data/training_batches'
x_train,y_train =  get_datalabel(train_path)


valid_path = '/home/menglin/tensorflow_models/models/transformer/data/validation_batches'
x_valid,y_valid =  get_datalabel(valid_path)

test_path = '/home/menglin/tensorflow_models/models/transformer/data/test_batches'
x_test,y_test =  get_datalabel(test_path)

np.savez('./data/mnist_sequence1_sample_5affine5x5.npz', X_train = x_train,
		 y_train = y_train,
         X_valid = x_valid,
		 y_valid = y_valid,
		 X_test = x_test,
		 y_test = y_test)