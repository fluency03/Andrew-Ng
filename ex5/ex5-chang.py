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
	# theta[0] = 0.0
	# print theta.T
	J = (1.0 / (2 * m)) * (squaredErrors) + (lamda / (2 * m)) * (theta.T.dot(theta))

	return J[0]


def computeGradient( theta, X, y, lamda ):
	theta = theta.reshape( shape(X)[1], 1 )
	m 	  = shape(y)[0]
	J 	  = 0
	grad  = zeros( shape(theta) )

	h = X.dot(theta)
	squaredErrors = (h - y).T.dot(h - y)
	# theta[0] = 0.0
	J = (1.0 / (2 * m)) * (squaredErrors) + (lamda / (2 * m)) * (theta.T.dot(theta))
	# theta[0] = 0.0
	grad = (1.0 / m) * (X.T.dot(h - y)) + (lamda / m) * theta

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
	# print theta
	# theta[0] = 0.0
	# print theta
	J = (1.0 / (2 * m)) * (squaredErrors) + (lamda / (2 * m)) * (theta.T.dot(theta))
	print J[0]
	# print shape(X), shape(h-y), shape(grad), shape(theta)
	# theta[0] = 0.0
	grad = (1.0 / m) * (X.T.dot(h - y)) + (lamda / m) * theta
	print grad.flatten()

	return J[0], grad.flatten()


def trainLinearReg( X, y, lamda, use_scipy=True ):
	# epsilon = 0.9
	theta = random.rand( shape(X)[1], 1 ) #* (2*epsilon)-epsilon # random initialization of theta
	# print theta

	if use_scipy is True:
		result = scipy.optimize.fmin_cg( computeCost, fprime = computeGradient, x0 = theta, 
										 args = (X, y, lamda), maxiter = 200, disp = True, full_output = True )
	else:
		result = Util.fmincg( f=computeCost, fprime=computeGradient, x0=theta, args=(X, y, lamda), maxiter=200 )
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
		error_train[i] = computeCost(theta, XSubset, ySubset, lamda)
		error_val[i]   = computeCost(theta, Xval, yval, lamda)

	# print error_train, error_val

	points = array([x for x in range(1, m+1)])

	# error_train = error_train.flatten()
	# error_val	= error_val.flatten()

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


def polyFeatures(X, p):
	m, n   = shape(X)
	powers = matlib.repmat(range(1, p+1), m, 1)
	Xrep   = matlib.repmat(X, 1, p)
	# print shape(powers), shape(Xrep)
	X_poly = Xrep ** powers
	# print shape(X_poly)
	# print X_poly
	# test   = (ones((12,8))*2) ** powers
	# print test

	return X_poly


def featureNormalize(X):
	# print X
	mu 	   = mean(X, axis=0 )
	# print shape(mu)
	# print mu
	X_norm = X - mu
	# print shape(X_norm)
	# print X_norm


	sigma  = std(X_norm, axis=0, ddof=1)
	# print shape(sigma)
	X_norm = X_norm / sigma


	return X_norm, mu, sigma


def plotFit(min_x, max_x, mu, sigma, theta, p):
	x = arange( min_x - 15, max_x + 25, 0.05).reshape(ceil((40+max_x-min_x)/0.05),1)
	# print shape(x)
	# print (40+max_x-min_x)/0.05

	X_poly = polyFeatures(x, p)
	X_poly = (X_poly - mu) / sigma
	# X_poly = X_poly / sigma
	# print X_poly

	X_poly = c_[ones( (shape(x)[0], 1) ), X_poly]
	# print shape(theta)
	# print shape(X_poly)

	pyplot.plot(x, X_poly.dot(theta), linestyle='--', linewidth=3)






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
	data = scipy.io.loadmat( PATH + "ex5data1.mat" )
	X, y = data['X'], data['y']
	m, n = shape(X)

	Xval, yval 	 = data['Xval'], data['yval']
	Xtest, ytest = data['Xtest'], data['ytest']

	p = 8

	X_poly 			  = polyFeatures(X, p)
	X_norm, mu, sigma = featureNormalize(X_poly)
	X_poly 			  = c_[ones((m, 1)), X_poly]

	X_poly_test = polyFeatures( Xtest, p )
	X_poly_test = X_poly_test - mu
	X_poly_test = X_poly_test / sigma
	X_poly_test = c_[ones(( shape(X_poly_test)[0], 1)), X_poly_test]
	
	X_poly_val = polyFeatures( Xval, p )
	X_poly_val = X_poly_val - mu
	X_poly_val = X_poly_val / sigma
	X_poly_val = c_[ones(( shape(X_poly_val)[0], 1)), X_poly_val]

	print X_poly[0, :]

	# part_3_1
	lamda 		= 0.0
	# print shape(X_poly)
	cost, theta = trainLinearReg(X_poly, y, lamda)
	# print theta
	# print cost

	pyplot.scatter( X, y, marker='x', c='r', s=30, linewidth=2 )
	pyplot.xlim([-80, 80])
	pyplot.ylim([-60, 40])
	pyplot.xlabel('Change in water level(x)')
	pyplot.ylabel('Water flowing out of the dam(y)')

	pyplot.text( -15, 165, 'Lambda = %.1f' %lamda )
	# print mu, sigma
	print theta
	plotFit( min(X), max(X), mu, sigma, theta, p )


	pyplot.show()








# main function
def main():
	part_1()
	part_2()
	part_3()




# call the main function
if __name__ == '__main__':
	main()


