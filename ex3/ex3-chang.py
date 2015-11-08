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


def displayData( X ):
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













def part_1_1and2():
	mat 	= scipy.io.loadmat( PATH + "ex3data1.mat" )
	X, Y 	= mat['X'], mat['y']
	print shape(X), shape(Y)

	displayData( X )



def part_1_3():
	pass





# main function
def main():
	part_1_1and2()
	# part_1_2()


# call the main function
if __name__ == '__main__':
	main()



	
