#!/usr/bin/python

'''For scientific computing'''
from numpy import *
import scipy.optimize, scipy.special

'''For plotting'''
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D


# Path of the files
PATH = '/home/cliu/Documents/github/Andrew-Ng/ex2/'


def plot( data ):
	pos  = data[data[:,2] == 1]
	neg  = data[data[:,2] == 0]
	
	pyplot.xlabel("Exam 1 score")
	pyplot.ylabel("Exam 2 score")
	# pyplot.xlim([25, 115])
	# pyplot.ylim([25, 115])

	pyplot.scatter( neg[:, 0], neg[:, 1], c='y', marker='o', s=40, linewidths=1, label="Not admitted" )
	pyplot.scatter( pos[:, 0], pos[:, 1], c='b', marker='+', s=40, linewidths=2, label="Admitted" )
	pyplot.legend() # illustration of different labels mention in the above two lines of codes
					# i.e., label="Not admitted"  and  label="Admitted"


# sigmoid function
def sigmoid( z ):
	return scipy.special.expit(z)
	# or, return 1.0 / (1.0 + exp( -z ))


def costFunction( X, Y, theta ):
	# print X.shape, theta.shape
	hypothesis = sigmoid( dot(X, theta) )
	first 	   = dot( log(hypothesis).T, -Y )
	second 	   = dot( log(1-hypothesis).T, -( 1-Y ))
	m 		   = shape(X)[0]

	return (( first + second )/m).flatten()  
	# if there is no .flatten() , cost is  [[ 0.69314718]]
	# if there is 	 .flatten() , cost is  [ 0.69314718]


def gradientCost( X, Y, theta ):
	m = shape(X)[0]
	return ( dot( X.T, sigmoid( dot(X, theta) ) - Y ) ) / m

def CostGradient( theta, X, Y ):
	cost = costFunction( X, Y, theta )
	grad = gradientCost( X, Y, theta )
	return cost


def findMinTheta( X, Y, theta ):
	# print X.shape, theta.shape
	result = scipy.optimize.fmin( CostGradient, x0=theta, args=(X, Y), maxiter=500, full_output=True )
	# func : callable func(x,*args);    x0 : ndarray
	# args : tuple. Extra arguments passed to func, i.e. f(x,*args).
	# x0 has to be the first argument of function called, i.e., 
	# theta must be at the first position of CostGradient( theta, X, Y )
	return result[0], result[1]


def plotBoundary( data, X, theta ):
	plot( data )
	plot_x = array( [min(X[:,1]), max(X[:,1])] )
	plot_y = (-1./ theta[2]) * (theta[1] * plot_x + theta[0])
	pyplot.plot( plot_x, plot_y )


def part_1_1():
	data = genfromtxt( PATH + "ex2data1.txt", delimiter=',')
	plot( data )
	pyplot.show()


def part_1_2():
	data = genfromtxt( PATH + "ex2data1.txt", delimiter=',')
	m, n = shape(data)[0], shape(data)[1] - 1   # shape(data)[0] is the #row of the data
												# shape(data)[1] is the #column of the data
	# print m, n
	X 	 = c_[ ones((m, 1)), data[:, :n] ]
	Y 	 = data[:, n:n+1]
	theta = zeros( (n+1, 1) ) 

	cost = costFunction(X, Y, theta)
	print "cost is ", cost
	grad = gradientCost( X, Y, theta )
	print "grad is ", grad

	theta, cost = findMinTheta(X, Y, theta)
	print "new cost is ", cost

	plotBoundary( data, X, theta )
	pyplot.show()





# main function
def main():
	part_1_1()
	part_1_2()


# call the main function
if __name__ == '__main__':
	main()