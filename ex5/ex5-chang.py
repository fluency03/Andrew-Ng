#!/usr/bin/python

import scipy.optimize, scipy.special, scipy.io
from numpy import * 
from numpy import matlib

import pylab
from matplotlib import pyplot, cm
from util import Util

# Path of the files
PATH = '/home/cliu/Documents/github/Andrew-Ng/ex5/'


def plotData( X, y ):
	m, n = shape(X)

	pyplot.plot(X, y, 'ro', markersize=7, linewidth=1.5)
	pyplot.xlabel('Change in water level (x)')
	pyplot.ylabel('Water flowing out of the dam (y)')
	# pyplot.show()


def computeCost( theta, X, y, lamda ):
	theta = theta.reshape( shape(X)[1], 1 )
	m 	  = shape(y)[0]
	J 	  = 0
	grad  = zeros( shape(theta) )

	h = X.dot(theta)
	squaredErrors = (h - y).T.dot(h - y)
	thetaExcludingZero = array( c_[zeros((1, 1)), theta[1:] ].flatten() )
	J = (1.0 / (2 * m)) * sum(squaredErrors) + (lamda / (2 * m)) * sum(thetaExcludingZero.T.dot(thetaExcludingZero))

	return J


def computeGradient( theta, X, y, lamda ):
	theta = theta.reshape( shape(X)[1], 1 )
	m 	  = shape(y)[0]
	J 	  = 0
	grad  = zeros( shape(theta) )

	h = X.dot(theta)
	squaredErrors = (h - y).T.dot(h - y)
	thetaExcludingZero = array( c_[zeros((1, 1)), theta[1:] ].flatten() )
	J = (1.0 / (2 * m)) * sum(squaredErrors) + (lamda / (2 * m)) * sum(thetaExcludingZero.T.dot(thetaExcludingZero))
	grad[:] = (1.0 / m) * (X.T.dot(h - y))[:] + (lamda / m) * thetaExcludingZero[1:]

	return grad.flatten()


def linearRegCostFunction( theta, X, y, lamda ):
	theta = theta.reshape( shape(X)[1], 1 )
	# print theta
	m 	  = shape(y)[0]
	J 	  = 0
	grad  = zeros( shape(theta) )
	# print grad 

	h = X.dot(theta)
	# print h
	squaredErrors = (h - y).T.dot(h - y)
	# print squaredErrors
	thetaExcludingZero = array( c_[zeros((1, 1)), theta[1:] ].flatten() )
	# print shape(thetaExcludingZero)
	J = (1.0 / (2 * m)) * sum(squaredErrors) + (lamda / (2 * m)) * sum(thetaExcludingZero.T.dot(thetaExcludingZero))
	print J
	grad[:] = (1.0 / m) * (X.T.dot(h - y))[:] + (lamda / m) * thetaExcludingZero[1:]
	print grad.flatten()

	return J, grad.flatten()


def trainLinearReg( X, y, lamda, use_scipy=False ):
	theta = zeros( (shape(X)[1], 1) )

	result = scipy.optimize.fmin_cg( computeCost, fprime = computeGradient, x0 = theta, 
									 args = (X, y, lamda), maxiter = 200, disp = True, full_output = True )
	print result[1], result[0]
	return result[1], result[0]











def part_1():
	data = scipy.io.loadmat( PATH + "ex5data1.mat" )
	X, y = data['X'], data['y']
	m, n = shape(X)
	# print shape(X), shape(y)
	X0 	 = X 

	# part_1_1
	plotData(X, y)
	pyplot.show()

	# part_1_2 & part_1_3
	theta = array([1, 1])
	lamda = 1.0
	X 	  = c_[ ones((m, 1)), X ]

	J, grad = linearRegCostFunction( theta, X, y, lamda )

	# part_1_4
	plotData(X0, y)
	
	# X_bias 		= c_[ones(shape(X)), X]
	cost, theta = trainLinearReg( X, y, 0.0 )
	# print theta, shape(theta)

	pyplot.plot( X0, X.dot( theta ), linewidth=2 )
	pyplot.show()




def part_2():
	# Xval, yval 	= mat['Xval'], mat['yval']
	# Xtest, ytest 	= mat['Xtest'], mat['ytest']
	data = scipy.io.loadmat( PATH + "ex5data1.mat" )
	X, y = data['X'], data['y']
	m, n = shape(X)























# main function
def main():
	part_1()
	part_2()





# call the main function
if __name__ == '__main__':
	main()


