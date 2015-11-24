#!/usr/bin/python

import scipy.optimize, scipy.special, scipy.io
from numpy import * 
from numpy import matlib
from sklearn import svm, grid_search

import pylab
from matplotlib import pyplot, cm
from util import Util

# Path of the files
PATH = '/home/cliu/Documents/github/Andrew-Ng/ex6/'



def plotData( data ):
	pos = data[data[:, 2] == 1]
	neg = data[data[:, 2] == 0]

	pyplot.plot( pos[:, 0], pos[:, 1], 'k+',linewidth=1, markersize=7 )
	pyplot.plot( neg[:, 0], neg[:, 1], 'ko', markerfacecolor='y', markersize=7 )


def svmTrain(X, Y, C, kernelFunction, tol, max_passes):
	pass


def visualizeBoundaryLinear(X, y, model):
	w  = model.dual_coef_.dot( model.support_vectors_ ).flatten()
	b  = model.intercept_
	xp = linspace( min(X[:,0]), max(X[:,0]), 100 )
	yp = - (w[0]*xp + b) / w[1]
	plotData( c_[X, y] )

	pyplot.plot(xp, yp, '-b', linewidth=1.0) 


def gaussianKernel(x1, x2, sigma):
	return exp(-sum((x1 - x2) ** 2) / (2.0 * (sigma ** 2)))


def visualizeBoundary(X, y, model):
	x1plot = linspace( min(X[:,0]), max(X[:,0] ), 100)
	x2plot = linspace( min(X[:,1]), max(X[:,1] ), 100)

	X1, X2 = meshgrid(x1plot, x2plot)
	vals = zeros( shape(X1) )

	for i in range(0, shape(X1)[1]):
		this_X 	   = c_[ X1[:, i], X2[:, i] ]
		vals[:, i] = model.predict( this_X )

	pyplot.contour( X1, X2, vals, colors='blue', linewidth=1.0 )


def dataset3Params(X, y, Xval, yval, ver=2):
	steps    = array([ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ])

	minError = float("inf")
	minScore = float("-inf")
	minC     = float("inf")
	minSigma = float("inf")

	rbf_svm  = svm.SVC(kernel='rbf')
	error    = 0.0
	score	 = 0.0
	m_val 	 = shape( Xval )[0] 

	for i in range(0, len(steps)):
		for j in range(0, len(steps)):
			curC = steps[i]
			curSigma = steps[j]

			rbf_svm.set_params( C=curC )
			rbf_svm.set_params( gamma = 1.0 / curSigma )
			rbf_svm.fit( X, y.ravel() )

			# --------------------------------------------------------------

			if ver == 1:
				predictions = rbf_svm.predict(Xval)
				# for i in range( 0, m_val ):
					# prediction_result = rbf_svm.predict( Xval[i] )
					# predictions[i] = prediction_result[0] 

				error = double(predictions != yval.reshape(m_val, 1)).mean()

				if (error < minError) :
					minError = error
					minC 	 = curC
					minSigma = curSigma
			# ---------------------------------------------------------------
			elif ver == 2:
				score = rbf_svm.score( Xval,  yval )

				if (score > minScore) :
					minScore = score
					minC 	 = curC
					minSigma = curSigma

			elif ver == 3:
				gammas 	    = map( lambda x: 1.0 / x, steps )

				parameters 	= {'kernel':('rbf', ), 'C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], 'gamma':map( lambda x: 1.0 / x, steps ) }
				grid 		= grid_search.GridSearchCV( rbf_svm, parameters )
				best 		= grid.fit( X, y.ravel() ).best_params_

				minC 	 = best['C']
				minSigma = 1.0 / best['gamma']

			# --------------------------------------------------------------

	C = minC
	sigma = minSigma
	print C, sigma

	return C, sigma










# ---------------------------------------------------------------------------------

def part_1_1():
	data = scipy.io.loadmat( PATH + "ex6data1.mat" )
	X, y = data['X'], data['y']
	m, n = shape(X)
	# print shape(X)

	# part_1_1
	plotData( c_[X, y] )
	pyplot.show( block=True )

	# linear SVM with C = 1
	linear_svm = svm.SVC(C=1, kernel='linear', tol=1e-3, max_iter=20)
	linear_svm.fit( X, y.ravel() )

	visualizeBoundaryLinear(X, y, linear_svm)
	pyplot.show( block=True )


def part_1_2():
	x1    = array( [1, 2, 1] )
	x2    = array( [0, 4, -1] ) 
	sigma = 2
	sim   = gaussianKernel(x1, x2, sigma)

	print "Gaussian kernel: %f" % sim


	data = scipy.io.loadmat( PATH + "ex6data2.mat" )
	X, y = data['X'], data['y']
	m, n = shape(X)

	plotData( c_[X, y] )
	pyplot.show( block=True )

	sigma   = 0.01
	rbf_svm = svm.SVC(C=1, kernel='rbf', gamma = 1.0 / sigma )
	rbf_svm.fit( X, y.ravel() )

	plotData( c_[X, y] )
	visualizeBoundary( X, y, rbf_svm )
	pyplot.show( block=True ) 


def part_1_3():
	data 	   = scipy.io.loadmat( PATH + "ex6data3.mat" )
	X, y 	   = data['X'], data['y']
	Xval, yval = data['Xval'], data['yval']

	C, sigma   = dataset3Params(X, y, Xval, yval, 1)

	rbf_svm    = svm.SVC(C=C, kernel='rbf', gamma = 1.0 / sigma )
	rbf_svm.fit( X, y.ravel() )

	plotData( c_[X, y] )
	visualizeBoundary( X, y, rbf_svm )
	pyplot.show( block=True ) 













# ---------------------------------------------------------------------------------

# main function
def main():
	part_1_1()
	part_1_2()
	part_1_3()




# call the main function
if __name__ == '__main__':
	main()


