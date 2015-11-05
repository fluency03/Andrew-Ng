#!/usr/bin/python

'''For scientific computing'''
from numpy import *
import scipy

'''For plotting'''
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D


# Path of the files
PATH = '/home/cliu/workspace/Andrew Ng/ex1/'


# generate the matrix - Part 1
def newMatrix():
	A = eye(5)
	print "A = \n", A


def hypothesis(X, theta):
	"""matrix - dot operator"""
	return dot(X, theta)
	# return X.dot(theta)


# cost function
def costFunction(X, Y, theta):
	"""Compute cost in the slower loop method."""
	m = len(Y)  # or m = shape(y)[0], since y is 1D

	overall_sum = 0
	for i in range(0, m):
		overall_sum += (hypothesis(X[i], theta) - Y[i]) ** 2
	# print "The costFunction returns ", overall_sum
	return overall_sum / (2*m)

# cost function using vector calculation
def costFunctionVec(X, Y, theta):
	"""Compute cost in vector calculation."""
	m = len(Y)  # or m = shape(y)[0], since y is 1D

	term = hypothesis(X, theta) - Y
	# sum( term**2 ) in this case ~= term.T.dot( term )
	return (term.T.dot(term) / (2 * m))[0, 0]



# the Gradient Descent
def gradientDescent(X, Y, theta, alpha, iterations):
	"""Gradient descent in loop version"""
	grad = copy(theta) # shadow copy, means only copy the structure of theta but not everything
	m 	 = len(Y) 
	n 	 = shape(X)[1] # the number of column of X

	for itr in range(0, iterations):
		# 
		overall_sum = [0 for x in range(0, n)]
		for j in range(0, n):
			for i in range(0, m):
				overall_sum[j] += (hypothesis(X[i], grad) - Y[i]) * X[i, j]

		# assign new values for each gradient, this should be separate from the loop above
		# in order to achieve simulataneous update effect
		for j in range(0, n):
			grad[j] = grad[j] - (alpha/m) * overall_sum[j]

	return grad


# the Gradient Descent
def gradientDescentVec(X, Y, theta, alpha, iterations):
    """Vectorized gradient descent"""
    grad = copy(theta)
    m 	 = len(Y)

    for itr in range(0, iterations):
        cum_sum = X.T.dot(hypothesis(X, grad) - Y)
        grad 	 -= (alpha / m) * cum_sum

    return grad


# function to plot the data
def plot(X, Y):
	pyplot.plot(X, Y, 'rx', markersize=5 )
	pyplot.ylabel('Profit in $10,000s')
	pyplot.xlabel('Population of City in 10,000s')


# plot the original data
def plotData1():
	data = genfromtxt( PATH + "ex1data1.txt", delimiter=',')
	X, Y = data[:, 0], data[:, 1] # x is the first column, y is the second column
	m = len(Y) # the number of data
	Y 	 = Y.reshape(m, 1)

	plot(X, Y) # plot the data
	pyplot.show(block=True) # Call it in order to show the plot window


def plotData2():
	data = genfromtxt( PATH + "ex1data1.txt", delimiter=',')
	X, Y = data[:, 0], data[:, 1] # x is the first column, y is the second column
	m    = len(Y) # the number of data
	Y 	 = Y.reshape(m, 1)

	# To take into account the intercept term (theta0), we add an additional first column to X 
	# and set it to all ones. This allows us to treat theta0 as simply another 'feature'.
	X 			= c_[ones((m, 1)), X] # Translates slice objects to concatenation along the second axis.
	theta 		= zeros((2, 1)) # initialize fitting parameters
	iterations 	= 1500
	alpha 		= 0.01

	cost 	= costFunctionVec(X, Y, theta)  # should be 32.07
	# cost 	= costFunction(X, Y, theta)  # should be 32.07
	theta 	= gradientDescentVec(X, Y, theta, alpha, iterations)
	# theta 	= gradientDescent(X, Y, theta, alpha, iterations)
	print "The cost is", cost
	print "The theta is ", theta

	# predict1 = dot(array([1, 3.5]), theta)
	# predict2 = dot(array([1, 7]), theta)
	predict1 = dot([1, 3.5], theta)
	predict2 = dot([1, 7], theta)
	print "The predict1 is ", predict1
	print "The predict2 is ", predict2

	plot(X[:, 1], Y)
	pyplot.plot(X[:, 1], dot(X, theta), 'b-')
	pyplot.show(block=True)


def visualization():
	data = genfromtxt( PATH + "ex1data1.txt", delimiter=',')
	X, Y = data[:, 0], data[:, 1] # x is the first column, y is the second column
	m    = len(Y) # the number of data
	Y 	 = Y.reshape(m, 1)
	X 	 = c_[ones((m, 1)), X] # Translates slice objects to concatenation along the second axis.

	theta0_vals = linspace(-10, 10, 100)
	theta1_vals = linspace(-4, 4, 100)

	# initialize J_vals to a matrix of 0's
	J_vals = zeros((len(theta0_vals), len(theta1_vals)), dtype=float64)
	# Fill out J_vals
	for i, v0 in enumerate(theta0_vals):
		for j, v1 in enumerate(theta1_vals):
			theta 		 = array((theta0_vals[i], theta1_vals[j])).reshape(2, 1)
			J_vals[i, j] = costFunctionVec(X, Y, theta)
			# J_vals[i, j] = costFunction(X, Y, theta)

	R, P = meshgrid(theta0_vals, theta1_vals)

	fig = pyplot.figure()
	ax 	= fig.gca(projection='3d')
	ax.plot_surface(R, P, J_vals,
					cmap=cm.jet, linewidth=0.2)
	pyplot.show()


	fig = pyplot.figure()
	# pyplot.contour(R, P, J_vals.T, 15, linewidths=0.5, colors='k')
	# pyplot.contourf(R, P, J_vals.T, 150,                   
					# cmap=pyplot.cm.rainbow, vmax=abs(J_vals).max(), vmin=-abs(J_vals).max())
	pyplot.contourf(R, P, J_vals.T, logspace(-2, 3, 200))
	pyplot.plot(theta[0], theta[1], 'rx', markersize = 10)
	pyplot.show(block=True)




# main function
def main():
	# set_printoptions(precision=6, linewidth=200)
	newMatrix()

	plotData1()
	plotData2()
	visualization()



# call the main function
if __name__ == '__main__':
	main()
