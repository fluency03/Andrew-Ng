#!/usr/bin/python

#!/usr/bin/python

# import PIL.Image
import scipy.optimize, scipy.special, scipy.io
# import scipy.misc, 
from numpy import * 
from numpy import matlib
# import numpy.matlib

import pylab
from matplotlib import pyplot, cm
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.mlab as mlab


# Path of the files
PATH = '/home/cliu/Documents/github/Andrew-Ng/ex4/'


def sigmoid(z):
	return scipy.special.expit(z)
	# return 1.0 / (1.0 + exp( -z ))


def displayData( X, theta=None ):
	width = 20 # width of one digital 
	rows, cols = 10, 10 # row and column number of the display image in terms of digitals
	out = zeros( (width*rows, width*cols) ) # the initialization of digitals being displayed

	rand_index = random.permutation( 5000 )[0: rows*cols]
	# generate an array of random numbers with max of 5000, and there 100 of them

	counter = 0
	for y in range(0, rows):
		for x in range (0, cols):
			x_index = x*width
			y_index = y*width
			out[x_index:x_index+width, y_index:y_index+width] = X[ rand_index[counter] ].reshape(width, width).T
			counter += 1

	img = scipy.misc.toimage(out)
	pyplot.imshow(img)

	pyplot.show()


def feedForward( Theta1, Theta2, X, m ):
	# a1 = c_[ ones((m,1)), X ]
	# z2 = Theta1.dot(a1.T) 
	# a2 = c_[ ones((m,1)), sigmoid(z2).T ]
	# z3 = Theta2.dot(a2.T)
	# h  = sigmoid( z3.T )

	a1 = c_[ ones((m,1)), X ]
	z2 = a1.dot(Theta1.T) 
	a2 = c_[ ones((m,1)), sigmoid(z2) ]
	z3 = a2.dot(Theta2.T) 
	h  = sigmoid( z3 )
	# print shape(h)

	return a1, z2, a2, z3, h






def nnCostFunction( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
	Theta1 = nn_params[0]
	Theta2 = nn_params[1]
	m, n   = shape(X)
	a1, z2, a2, z3, h = feedForward( Theta1, Theta2, X, m )

	yVec = ( (matlib.repmat(arange(1, num_labels+1), m, 1) == matlib.repmat(y, 1, num_labels)) + 0)

	# cost = sum( - yVec * log(h) - (1 - yVec) * log(1 - h) ) / m
	term1 		= - yVec * log(h)
	term2 		= - (1 - yVec) * log(1 - h)
	left_term 	= sum(term1 + term2) / m

	Theta1ExcludingBias = Theta1[:,1:]
	Theta2ExcludingBias = Theta2[:,1:]
	right_term 	= ( sum(Theta1ExcludingBias ** 2) + sum(Theta2ExcludingBias ** 2) ) * lamda / (2*m)
	
	cost 		= left_term + right_term
	print cost


	Theta1_grad = zeros( shape(Theta1) )
	Theta2_grad = zeros( shape(Theta2) )
	delta1 		= zeros( shape(Theta1) )
	delta2 		= zeros( shape(Theta2) )









def part_1():
	data = scipy.io.loadmat( PATH + "ex4data1.mat" )
	X, y = data['X'], data['y']
	m, n = shape(X)
	print shape(X)

	displayData( X )

	input_layer_size  = 400 
	hidden_layer_size = 25 
	num_labels 		  = 10
	lamda			  = [0, 1]

	weight = scipy.io.loadmat( PATH + "ex4weights.mat" )
	Theta1, Theta2 = weight['Theta1'], weight['Theta2']
	nn_params = [ Theta1, Theta2 ]


	nnCostFunction( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda[0])
	# Regularized cost function
	nnCostFunction( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda[1])


def part_2():
	pass






# main function
def main():
	part_1()
	part_2()


# call the main function
if __name__ == '__main__':
	main()


