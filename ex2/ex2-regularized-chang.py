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
	
	pyplot.xlabel("Microchip test 1")
	pyplot.ylabel("Microchip test 2")
	# pyplot.xlim([25, 115])
	# pyplot.ylim([25, 115])

	pyplot.scatter( neg[:, 0], neg[:, 1], c='y', marker='o', s=40, linewidths=1, label="y=0" )
	pyplot.scatter( pos[:, 0], pos[:, 1], c='k', marker='+', s=40, linewidths=2, label="y=1" )
	pyplot.legend() # illustration of different labels mention in the above two lines of codes
					# i.e., label="Not admitted"  and  label="Admitted"


def mapFeature( X1, X2 ):
	degrees = 6
	X = ones( (shape(X1)[0], 1) )

	for i in range(1, degrees+1):
		for j in range(0, i+1):
			term1 = X1 ** j
			term2 = X2 ** (i - j)
			term  = (term1 * term2) .reshape( shape(term1)[0], 1 ) 
			# print shape(term)
			X 	  = hstack(( X, term ))
			# print shape(X), shape(term)

	return X


def sigmoid( z ):
	return scipy.special.expit(z)
	# return 1.0 / (1.0 + exp( -z ))


def costFunctionReg( X, Y, theta, lamda):
	m 		   = shape(X)[0]
	hypothesis = sigmoid( dot(X, theta) )
	first 	   = dot( log(hypothesis).T, -Y )
	second 	   = dot( log(1-hypothesis).T, -( 1-Y ))
	reg_term   = dot( theta.T, theta) * (lamda / (2*m))

	return (( first + second )/m + reg_term)
	# if there is no .flatten() , cost is  [[ 0.69314718]]
	# if there is 	 .flatten() , cost is  [ 0.69314718]


def gradientCostReg( X, Y, theta, lamda ):
	m = shape(X)[0]
	grad = X.T.dot( sigmoid( X.dot( theta ) ) - Y ) / m
	grad[1:] = grad[1:] + ( (theta[1:] * lamda ) / m )
	return grad


def CostGradient( theta, X, Y, lamda ):
	cost = costFunctionReg( X, Y, theta, lamda )
	grad = gradientCostReg( X, Y, theta, lamda )
	return cost


def findMinTheta( X, Y, theta, lamda ):
	# print X.shape, theta.shape
	result = scipy.optimize.minimize( CostGradient, x0=theta, args=(X, Y, lamda), method='BFGS', options={"maxiter":1000, "disp":True}  )
	# result = scipy.optimize.fmin( CostGradient, x0=theta, args=(X, Y, lamda), maxiter=500, full_output=True )
	# func : callable func(x,*args);    x0 : ndarray
	# args : tuple. Extra arguments passed to func, i.e. f(x,*args).
	# x0 has to be the first argument of function called, i.e., 
	# theta must be at the first position of CostGradient( theta, X, Y )
	return result.x, result.fun
	# return result[0], result[1]



def part_2_1():
	data = genfromtxt( PATH + "ex2data2.txt", delimiter=',')
	plot( data )
	pyplot.show()


def part_2_2():
	data = genfromtxt( PATH + "ex2data2.txt", delimiter=',')
	X 	 = mapFeature( data[:, 0:1], data[:, 1:2] )
	# print shape(data), shape(data[:, 0:1])
	print X


def part_2_3():
	data   = genfromtxt( PATH + "ex2data2.txt", delimiter=',')
	X 	   = mapFeature( data[:, 0:1], data[:, 1:2] )
	Y	   = data[:,2]
	theta  = zeros( shape(X)[1] )
	lamda  = 1.0

	cost = costFunctionReg(X, Y, theta, lamda)
	print cost

	grad = gradientCostReg(X, Y, theta, lamda)
	print grad

	theta, cost = findMinTheta(X, Y, theta, lamda)
	print "new cost is ", cost


def part_2_4():
	data   = genfromtxt( PATH + "ex2data2.txt", delimiter=',')
	X 	   = mapFeature( data[:, 0:1], data[:, 1:2] )
	Y	   = data[:,2]
	theta  = zeros( shape(X)[1] )
	lamdas = [0.0, 1.0, 10.0, 100.0]

	for lamda in lamdas:
		theta, cost = findMinTheta( X, Y, theta, lamda )

		pyplot.text( 0.15, 1.4, 'Lamda %.1f' % lamda )
		plot( data )

		u = linspace( -1, 1.5, 50 )
		v = linspace( -1, 1.5, 50 )
		z = zeros( (len(u), len(v)) )

		for i in range(0, len(u)): 
			for j in range(0, len(v)):
				mapped = mapFeature( array([u[i]]), array([v[j]]) )
				z[i,j] = mapped.dot( theta )
				# z[i, j] = (mapFeature(array([u[i]]), array([v[j]])).dot(array(theta)))
		z = z.transpose()

		u, v = meshgrid( u, v )	
		pyplot.contour( u, v, z, label='Decision Boundary' )
		# pyplot.legend()

		pyplot.show()








# main function
def main():
	part_2_1()
	part_2_2()
	part_2_3()
	part_2_4()


# call the main function
if __name__ == '__main__':
	main()