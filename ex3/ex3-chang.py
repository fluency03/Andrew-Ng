#!/usr/bin/python

#!/usr/bin/python

# import PIL.Image
import scipy.optimize, scipy.special, scipy.io
# import scipy.misc, 
from numpy import *

import pylab
from matplotlib import pyplot, cm
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.mlab as mlab


# Path of the files
PATH = '/home/cliu/Documents/github/Andrew-Ng/ex3/'



def sigmoid( z ):
	return scipy.special.expit(z)
	# return 1.0 / (1.0 + exp( -z ))


def displayData( X, theta=True ):
	width = 20
	rows, cols = 10, 10
	out = zeros(( width * rows, width*cols ))

	rand_indices = random.permutation( 5000 )[0:rows * cols]
	# generate an array of random numbers with max of 5000, and there 100 of them

	counter = 0
	for y in range(0, rows):
		for x in range(0, cols):
			start_x = x * width
			start_y = y * width
			out[start_x:start_x+width, start_y:start_y+width] = X[rand_indices[counter]].reshape(width, width).T
			counter += 1

	img 	= scipy.misc.toimage( out )
	# figure  = pyplot.figure()
	# axes    = figure.add_subplot(1, 1, 1)
	# axes.imshow( img )
	pyplot.imshow(img)

	pyplot.show()


def sigmoid( z ):
	return scipy.special.expit(z)
	# return 1.0 / (1.0 + exp( -z ))


def costFunction( theta, X, Y, lamda):
	m 		   = shape(X)[0]
	hypothesis = sigmoid( dot(X, theta) )
	first 	   = dot( log(hypothesis).T, -Y )
	second 	   = dot( log(1-hypothesis).T, -( 1-Y ))
	reg_term   = dot( theta.T, theta) * (lamda / (2*m))

	return (( first + second )/m + reg_term)


def gradientCost( theta, X, Y, lamda ):
	m = shape(X)[0]
	grad = X.T.dot( sigmoid( X.dot( theta ) ) - Y ) / m
	grad[1:] = grad[1:] + ( (theta[1:] * lamda ) / m )
	return grad


def oneVsALL( X, Y, num_classes, lamda ):
	m, n      = shape(X)
	all_theta = zeros( (num_classes, n+1) )
	X 		  = c_[ones((m, 1)), X]

	for c in range(0, num_classes):
		initial_theta = zeros( (n + 1, 1) ).reshape(-1)
		temp_y 		  = ((Y == (c+1)) + 0).reshape(-1)
		theta 		  = scipy.optimize.fmin_cg( costFunction, fprime=gradientCost, x0=initial_theta, \
												args=(X, temp_y, lamda), maxiter=50, disp=False, full_output=True )
		all_theta[c,:] = theta[0]
		print "%d Cost: %.5f" % (c+1, theta[1])

	return all_theta


def predictOneVsAll( theta, X, Y ):
	m,n = shape( X )
	X 	= c_[ones((m, 1)), X]

	correct = 0
	for i in range(0, m):
		prediction 	= argmax( sigmoid(dot(X[i], theta.T)) ) + 1
		# argmax(): Returns the indices of the maximum values along an axis. 
		actual 		= Y[i]
		# print "prediction = %d actual = %d" % (prediction, actual)
		if actual == prediction:
			correct += 1
	print "Accuracy: %.2f%%" % (correct * 100.0 / m )


def predict( Theta1, Theta2, X ):
	m,n = shape( X )
	# X 	= c_[ones((m, 1)), X]

	a1 = c_[ones((m, 1)), X]
	a2 = c_[ones((m, 1)), sigmoid( dot(a1, Theta1.T) )]
	a3 = sigmoid( dot(a2, Theta2.T) )
	return ( a3 ) 





def part_1_1and2():
	mat 	= scipy.io.loadmat( PATH + "ex3data1.mat" )
	X, Y 	= mat['X'], mat['y']
	print shape(X), shape(Y)

	displayData( X )


def part_1_4():
	mat 		= scipy.io.loadmat( PATH + "ex3data1.mat" )
	X, Y 		= mat['X'], mat['y']
	m, n 	    = shape(X) 
	num_classes = 10
	lamda 		= 0.1

	theta = oneVsALL( X, Y, num_classes, lamda )
	print theta

	predictOneVsAll( theta, X, Y )
	displayData( X, theta )


def part_2_1():
	# Setup the parameters you will use for this exercise
	input_layer_size  = 400  # 20x20 Input Images of Digits
	hidden_layer_size = 25   # 25 hidden units
	num_labels 		  = 10   # 10 labels, from 1 to 10   
							 # (note that we have mapped "0" to label 10)

	# load data
	data = scipy.io.loadmat( PATH + "ex3data1.mat" )
	X, Y = data['X'], data['y']
	m, n = shape(X)

	displayData(X)

	# load weight
	weight 		   = scipy.io.loadmat( PATH + "ex3weights.mat" )
	Theta1, Theta2 = weight['Theta1'], weight['Theta2']
	# print Theta1, Theta2

	prediction = predict(Theta1, Theta2, X)
	print shape(prediction)
	correct = 0
	for i in range(0, m):
		pred   = argmax( prediction[i] ) + 1
		# argmax(): Returns the indices of the maximum values along an axis. 
		actual = Y[i]
		# print "prediction = %d actual = %d" % (pred, actual)
		if actual == pred:
			correct += 1
	print "Accuracy: %.2f%%" % (correct * 100.0 / m )







# main function
def main():
	part_1_1and2()
	part_1_4()
	part_2_1()


# call the main function
if __name__ == '__main__':
	main()


