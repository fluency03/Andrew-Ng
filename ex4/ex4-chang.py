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
	a1 = r_[ ones((1,m)), X.T ]
	z2 = Theta1.dot(a1) 
	a2 = r_[ ones((1,m)), sigmoid(z2) ]
	z3 = Theta2.dot(a2)
	h  = sigmoid( z3 )

	# print shape(h)

	return a1, z2, a2, z3, h


def sigmoidGradient( z ):
	return sigmoid(z) * ( 1-sigmoid(z) )


def paramRollback( nn_params, input_layer_size, hidden_layer_size, num_labels ):
	theta1_elems = ( input_layer_size + 1 ) * hidden_layer_size
	theta1_size  = ( input_layer_size + 1, hidden_layer_size  )
	theta2_size  = ( hidden_layer_size + 1, num_labels )

	# print theta1_elems, theta1_size, theta2_size
	# print shape(nn_params)

	theta1 = nn_params[:theta1_elems].T.reshape( theta1_size ).T	
	theta2 = nn_params[theta1_elems:].T.reshape( theta2_size ).T

	return (theta1, theta2)


def computeCost( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda ):
	Theta1, Theta2 = paramRollback( nn_params, input_layer_size, hidden_layer_size, num_labels )
	# print shape(Theta1), shape(Theta2)
	m, n   = shape(X)
	a1, z2, a2, z3, h = feedForward( Theta1, Theta2, X, m )

	yVec = ( (matlib.repmat(arange(1, num_labels+1), m, 1) == matlib.repmat(y, 1, num_labels)) + 0)

	# cost = sum( - yVec * log(h) - (1 - yVec) * log(1 - h) ) / m
	term1 		= - yVec * log(h.T)
	term2 		= - (1 - yVec) * log(1 - h.T)
	left_term 	= sum(term1 + term2) / m

	Theta1ExcludingBias = Theta1[:,1:]
	Theta2ExcludingBias = Theta2[:,1:]
	right_term 	= ( sum(Theta1ExcludingBias ** 2) + sum(Theta2ExcludingBias ** 2) ) * lamda / (2*m)
	
	J = left_term + right_term
	return J


def nnCostFunction( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda ):
	Theta1 = nn_params[0]
	Theta2 = nn_params[1]
	# print shape(Theta1), shape(Theta2)
	m, n   = shape(X)
	a1, z2, a2, z3, h = feedForward( Theta1, Theta2, X, m )

	yVec = ( (matlib.repmat(arange(1, num_labels+1), m, 1) == matlib.repmat(y, 1, num_labels)) + 0)

	# cost = sum( - yVec * log(h) - (1 - yVec) * log(1 - h) ) / m
	term1 		= - yVec * log(h.T)
	term2 		= - (1 - yVec) * log(1 - h.T)
	left_term 	= sum(term1 + term2) / m

	Theta1ExcludingBias = Theta1[:,1:]
	Theta2ExcludingBias = Theta2[:,1:]
	right_term 	= ( sum(Theta1ExcludingBias ** 2) + sum(Theta2ExcludingBias ** 2) ) * lamda / (2*m)
	
	J = left_term + right_term
	# print J


	D1 = zeros(shape(Theta1))
	D2 = zeros(shape(Theta2))

	sigma3 = h - yVec.T
	sigma2 = Theta2.T.dot(sigma3) * sigmoidGradient( r_[ ones((1,m)), z2 ] )
	sigma2 = sigma2[1:,:]
	# print shape(sigma2)
	delta1 = sigma2.dot( a1.T )
	delta2 = sigma3.dot( a2.T ) 

	D1[:,1:] = delta1[:,1:]/m + (Theta1[:,1:] * lamda / m)
	D2[:,1:] = delta2[:,1:]/m + (Theta2[:,1:] * lamda / m)
	D1[:,0] = delta1[:,0]/m
	D2[:,0] = delta2[:,0]/m 

	# print D1, shape(D1)
	# print D2, shape(D2)

	return J, D1, D2


def randInitializeWeights( L_in, L_out ):
	W = zeros( (L_out, 1 + L_in) )

	epsilon_init = 0.12
	W = random.random((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
	return W


def computeNumericalGradient( theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamda ):
	numgrad = zeros(shape(theta))
	perturb = zeros(shape(theta))
	e = 1e-4

	for p in range( 0, shape(theta)[0] ):
		# Set perturbation vector
		# print p
		perturb[p] = e
		loss1 = computeCost( theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lamda )
		loss2 = computeCost( theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lamda )
		# Compute Numerical Gradient
		numgrad[p] = (loss2 - loss1) / (2*e)
		perturb[p] = 0
	
	return numgrad


def debugInitializeWeights(fan_out, fan_in):
	num_elements = fan_out * (1+fan_in)
	W = array([sin(x) / 10 for x in range(1, num_elements+1)])
	return W.reshape( 1+fan_in, fan_out ).T


def checkNNGradients( lamda=0 ):
	input_layer_size  = 3
	hidden_layer_size = 5
	num_labels 		  = 3
	m 				  = 5

	Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
	Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

	X  = debugInitializeWeights(m, input_layer_size - 1)
	y  = 1 + mod(m, num_labels).T

	nn_params 	= array([theta1.T.reshape(-1).tolist() + theta2.T.reshape(-1).tolist()]).T
	gradient 	= nnCostFunction( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda )[1]
	numgrad 	= computeNumericalGradient( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda )
	diff = linalg.norm( numgrad - gradient ) / (linalg.norm( numgrad + gradient ))
	print diff


def part_1():
	data = scipy.io.loadmat( PATH + "ex4data1.mat" )
	X, y = data['X'], data['y']
	m, n = shape(X)
	# print shape(X)

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
	# part_2_1
	print sigmoidGradient(0)
	print sigmoidGradient( array([1, -0.5, 0, 0.5, 1]) )

	# part_2_2
	theta1 = randInitializeWeights( 400, 25 )
	theta2 = randInitializeWeights( 25, 10 )

	# part_2_3
	data = scipy.io.loadmat( PATH + "ex4data1.mat" )
	X, y = data['X'], data['y']
	m, n = shape(X)

	weight = scipy.io.loadmat( PATH + "ex4weights.mat" )
	Theta1, Theta2 = weight['Theta1'], weight['Theta2']
	nn_params = [ Theta1, Theta2 ]

	input_layer_size  	= 400
	hidden_layer_size 	= 25
	num_labels 			= 10
	lamda 				= 1


	J, D1, D2 = nnCostFunction( nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)

	# unroll theta
	theta = array([D1.reshape(-1).tolist() + D2.reshape(-1).tolist()]).T
	# print shape(theta)

	#--------------------------this part is incredibly slow--------------------------------------------#
	# grad = computeNumericalGradient( theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamda )
	# print "grad is ", grad
	#--------------------------------------------------------------------------------------------------#

	# part_2_4
	






# main function
def main():
	part_1()
	part_2()


# call the main function
if __name__ == '__main__':
	main()


