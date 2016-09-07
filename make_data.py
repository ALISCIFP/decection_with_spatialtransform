import numpy as np
import cv2
from PIL import Image     

def print_as_image( img ):
	"print an array as image"
	img = img * 256
	img = Image.fromarray(img.astype(np.uint8),'L')
	img.show()
   	return 

def combined_data(x,y):
	mul = 2
	n_object = x.shape[0]
	n_out = n_object / mul
	print('nout' + str(n_out))
	new_x = np.zeros((n_out, x.shape[1]*4),dtype = np.float32)
	new_y = np.zeros((n_out),dtype = np.int32)
	d = np.zeros((3))
	for i in range(0,(n_out-1)):
		rand1 = int(np.random.rand()*x.shape[0])
 		rand2 = int(np.random.rand()*x.shape[0])
 		rand3 = int(np.random.rand()*x.shape[0])
 		rand4 = int(np.random.rand()*x.shape[0])
 		local_matr = np.zeros((80,80),dtype = np.float32)
 		local_matr[0:40,0:40] = x[rand1,:].reshape((40,40,)).astype(np.float32)/256
 		local_matr[40:80,0:40] = x[rand2,:].reshape((40,40)).astype(np.float32)/256
 		local_matr[0:40,40:80] = x[rand3,:].reshape((40,40)).astype(np.float32)/256
 		local_matr[40:80,40:80] = x[rand4,:].reshape((40,40)).astype(np.float32)/256
 		new_x[i,:] =  local_matr.reshape(6400)
 		new_y[i] = y[rand1] + y[rand2] + y[rand3] + y[rand4]
 		if i == 1:
 			print_as_image( local_matr)
 			print(new_y[i])
 	return new_x,new_y	
 		
mnist_cluttered = np.load('./data/mnist_sequence1_sample_5affine5x5.npz')
X_train = mnist_cluttered['X_train']
y_train = mnist_cluttered['y_train']
X_valid = mnist_cluttered['X_valid']
y_valid = mnist_cluttered['y_valid']
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']

# img = X_train[1].reshape((40,40)) 
# print_as_image(img)
new_x_train,new_y_train = combined_data(X_train,y_train)
new_x_valid,new_y_valid = combined_data(X_valid,y_valid)
new_x_test,new_y_test = combined_data(X_test,y_test)

np.savez('./data/mnist_sequence1_sample_5distortions5x5_4channel.npz', X_train = new_x_train,
		 y_train = new_y_train,
         X_valid = new_x_valid,
		 y_valid = new_y_valid,
		 X_test = new_x_test,
		 y_test = new_y_test)
# scipy.misc.imsave('outfile.jpg', image_array)
# im = Image.fromarray(A)
# im.save("your_file.jpeg")
