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
	# print theta
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


def trainLinearReg( X, y, lamda, use_scipy=True ):
	theta = zeros( (shape(X)[1], 1) )

	result = scipy.optimize.fmin_cg( computeCost, fprime = computeGradient, x0 = theta, 
									 args = (X, y, lamda), maxiter = 200, disp = True, full_output = True )
	# print result[1], result[0]
	return result[1], result[0]


def learningCurve(X, y, Xval, yval, lamda):
	m, n = shape(X)

	error_train = zeros((m, 1))
	error_val   = zeros((m, 1))

	X 	  = c_[ ones((m, 1)), X ]
	mval  = shape(Xval)[0]
	Xval  = c_[ ones((mval, 1)), Xval ]

	for i in range(0, m):
		XSubset 	   = X[0:i+1, :]
		ySubset 	   = y[0:i+1, :]
		# print shape(XSubset)
		cost, theta    = trainLinearReg(XSubset, ySubset, lamda)
		# print(theta)
		error_train[i], grad_train = linearRegCostFunction(theta, XSubset, ySubset, 0)
		error_val[i], grad_val     = linearRegCostFunction(theta, Xval, yval, 0)


	points = array([x for x in range(1, m+1)])

	pyplot.plot( points, error_train, color='b', linewidth=2, label='Train' )
	pyplot.plot( points, error_val, color='g', linewidth=2, label='Cross Validation' )

	pyplot.ylabel('Error')
	pyplot.xlabel('Number of training examples')
	pyplot.ylim([-2, 150])
	pyplot.xlim([0, 13])

	pyplot.legend()
	pyplot.show( block=True )

	# print shape(error_train), shape(error_val)
	return error_train, error_val








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
	data = scipy.io.loadmat( PATH + "ex5data1.mat" )
	X, y = data['X'], data['y']
	m, n = shape(X)

	Xval, yval 	 = data['Xval'], data['yval']
	Xtest, ytest = data['Xtest'], data['ytest']

	lamda = 0.0

	error_train, error_val = learningCurve(X, y, Xval, yval, lamda)


def part_3():
	pass















# main function
def main():
	part_1()
	part_2()
	part_3()




# call the main function
if __name__ == '__main__':
	main()


