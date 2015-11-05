#!/usr/bin/python

'''For scientific computing'''
from numpy import *
import scipy

'''For plotting'''
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D


# Path of the files
PATH = '/home/cliu/workspace/Andrew Ng/ex1/'


# function to plot the data
def plot(X, Y, color):
	pyplot.plot(X, Y, color, markersize=5 )


def featureNormalize(X):
	mean0 = mean(X, axis=0)
	# print " mean =", mean0
	max0 = amax(X, axis=0)
	# print " max =", max0
	std0 = std(X, axis=0)
	# print " std =", std0
	newX = (X - mean0) / std0

	return newX, mean0, max0, std0


def costFunctionMulti(X, Y, theta, m): 
	term = dot(X, theta) - Y
	# sum( term**2 ) in this case ~= term.T.dot( term )
	return (term.T.dot(term) / (2 * m))[0, 0]


def gradientDescentMulti(X, Y, theta, alpha, iterations, m):
	grad = copy(theta)
	J_history = zeros(( iterations, 1))

	for itr in range(0, iterations):
		cum_sum = X.T.dot(dot(X, grad) - Y)
		grad 	 -= (alpha / m) * cum_sum
		J_history[itr] = costFunctionMulti(X, Y, grad, m )

	return J_history, grad


def normalEqn(X, Y):
	return linalg.inv(X.T.dot( X )).dot( X.T ).dot( Y )


def part_3_1():
	data = genfromtxt( PATH + "ex1data2.txt", delimiter=',')
	X, Y = data[:, 0:2], data[:, 2:3] # x is the first column, y is the second column
	m = len(Y) # the number of data

	X, mean0, max0, std0 = featureNormalize(X)
 
	plot(X[:, 0], Y, 'rx')
	# # pyplot.show(block=True)
	plot(X[:, 1], Y, 'bx')
	pyplot.show(block=True)


def part_3_2():
	data = genfromtxt( PATH + "ex1data2.txt", delimiter=',')
	X, Y = data[:, 0:2], data[:, 2:3] # x is the first column, y is the second column
	m = len(Y) # the number of data

	X, mean0, max0, std0 = featureNormalize(X)

	X 			= c_[ ones((m, 1)), X ] # add intercept to X
	theta 		= zeros( (3, 1) )
	iterations 	= 400
	alphas 		= [0.01, 0.03, 0.1, 0.3, 1.0]

	cost = costFunctionMulti(X, Y, theta, m)
	print cost

	# for each alpha, try to do gradient descent and plot the convergence curve
	for alpha in alphas:
		theta 		= zeros( (3, 1) )
		J_history, theta = gradientDescentMulti(X, Y, theta, alpha, iterations, m)

		# create an array of number of iterations
		number_of_iterations = array( [x for x in range( 1, iterations + 1 )] ).reshape( iterations, 1 )

		pyplot.plot( number_of_iterations, J_history, '-b' )
		pyplot.title( "Alpha = %f" % (alpha) )
		pyplot.xlabel('Number of iterations')
		pyplot.ylabel('Cost J')
		pyplot.xlim( [0, 50] )
		pyplot.show( block=True )

		# 1650 sq feet 3 bedroom house
		test = array([1.0, 1650.0, 3.0])
		# exclude intercept units
		test[1:] = (test[1:] - mean0) / std0
		print test.dot( theta )


def part_3_3():
	data = genfromtxt( PATH + "ex1data2.txt", delimiter=',')	
	X = data[:, 0:2]
	Y = data[:, 2:3]
	m = shape( X )[0]

	X = c_[ ones((m, 1)), X ] # add intercept to X

	theta = normalEqn( X, Y )
	# 1650 sq feet 3 bedroom house
	test = array([1.0, 1650.0, 3.0])
	print "normalEqn is ", test.dot( theta )




# main function
def main():
	part_3_1()
	part_3_2()
	part_3_3()


# call the main function
if __name__ == '__main__':
	main()
