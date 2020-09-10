# coding=utf-8
import scipy.io as sio
import h5py
import numpy as np
import time
from pytictoc import TicToc
from scipy.linalg import norm
import math
import random

#from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
import pickle
#from itertools import cycle, islice

def binaryconvert(s, num1, num2):
	u = np.empty((num1, num2, n), 'uint8')
	for i in range(n):
		u[:,:,i] = (s >> i) & 1
	return u

'''Bitwise operations'''
def TripleGenerate(numofrow, numofcol, numoftrip):
	triplet = np.random.randint(0, 2, (6, numofrow, numofcol, numoftrip))
	triplet[5] = ((triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]) % 2
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def BitMultiplyMatrix(u1, u2, v1, v2, ta1, tb1, tc1, ta2, tb2, tc2):
	u_a = (u1 - ta1 + u2 - ta2) % 2							# S1 <-> S2
	v_b = (v1 - tb1 + v2 - tb2) % 2
	z1 = (tc1 + u_a * tb1 + v_b * ta1)	% 2					# S1
	z2 = (tc2 + u_a * tb2 + v_b * ta2 + u_a * v_b) % 2		# S2
	return z1, z2

def BitAddition(u1, u2, v1, v2, num1, num2):			# input 10 based numbers
	krange = int(math.log(n, 2))

	t_off_21 = time.time()
	ta1, tb1, tc1, ta2, tb2, tc2 = TripleGenerate(num1, num2, n)
	t_off_22 = time.time()

	t_on_21 = time.time()
	p1 = u1 ^ v1
	p2 = u2 ^ v2
	s1, s2 = BitMultiplyMatrix(u1, u2, v1, v2, ta1, tb1, tc1, ta2, tb2, tc2)
	t_on_22 = time.time()

	t_off_31 = time.time()
	lm = 0
	for k in range(krange):
		lrange = 2**k
		mrange = n // (lrange+1)
		lm += lrange * mrange
	ta11, tb11, tc11, ta12, tb12, tc12 = TripleGenerate(num1, num2, lm)
	ta21, tb21, tc21, ta22, tb22, tc22 = TripleGenerate(num1, num2, lm)
	t_off_32 = time.time()

	c1 = np.empty((num1, num2, n), 'uint8')
	c2 = np.empty((num1, num2, n), 'uint8')
	w1 = np.empty((num1, num2, n), 'uint8')
	w2 = np.empty((num1, num2, n), 'uint8')
	
	t_on_41 = time.time()
	c1[:,:,0] = 0
	c2[:,:,0] = 0
	for i in range(1, n):
		temp1, temp2 = BitMultiplyMatrix(p1[:,:,(i-1)], p2[:,:,(i-1)], c1[:,:,(i-1)], c2[:,:,(i-1)], ta11[:,:,i], tb11[:,:,i], tc11[:,:,i], ta12[:,:,i], tb12[:,:,i], tc12[:,:,i])
		c1[:,:,i] = s1[:,:,(i-1)] ^ temp1
		c2[:,:,i] = s2[:,:,(i-1)] ^ temp2

	w1[:,:,0] = u1[:,:,0] ^ v1[:,:,0]
	w2[:,:,0] = u2[:,:,0] ^ v2[:,:,0]
	w1[:,:,1:n] = u1[:,:,1:n] ^ v1[:,:,1:n] ^ c1[:,:,1:n]
	w2[:,:,1:n] = u2[:,:,1:n] ^ v2[:,:,1:n] ^ c2[:,:,1:n]
	t_on_42 = time.time()

	t_off = t_off_22 - t_off_21 + t_off_32 - t_off_31
	t_on = t_on_22 - t_on_21 + t_on_42 - t_on_41
	return w1[:,:,n-1], w2[:,:,n-1], t_off, t_on

def BitExtractionMatrix(u1, u2, num1 = 1, num2 = 6):     # used in word vector training 
	t_off_11 = time.time()
	r1 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	r2 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	rsum = r1 + r2
	q01 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	q02 = rsum ^ q01
	q1 = binaryconvert(q01, num1, num2)
	q2 = binaryconvert(q02, num1, num2)
	t_off_12 = time.time()
	
	t_on_11 = time.time()
	u1 = u1 - r1
	u2 = u2 - r2
	v0b = u1 + u2
	v01 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	v02 = v0b ^ v01
	v1 = binaryconvert(v01, num1, num2)
	v2 = binaryconvert(v02, num1, num2)
	t_on_12 = time.time()

	w1, w2, t_off_0, t_on_0 = BitAddition(v1, v2, q1, q2, num1, num2)
	t_off = t_off_12 - t_off_11 + t_off_0
	t_on = t_on_12 - t_on_11 + t_on_0
	
	return w1, w2, t_off, t_on

def BitExtractionMatrix2(u1, u2, num1 = 1, num2 = 1):     # used in SVM 
	t_off_11 = time.time()
	r1 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	r2 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	rsum = r1 + r2
	q01 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	q02 = rsum ^ q01
	q1 = binaryconvert(q01, num1, num2)
	q2 = binaryconvert(q02, num1, num2)
	t_off_12 = time.time()
	
	t_on_11 = time.time()
	u1 = u1 - r1
	u2 = u2 - r2
	v0b = u1 + u2
	v01 = np.random.randint(-2**(n-2),2**(n-2),(num1,num2))
	v02 = v0b ^ v01
	v1 = binaryconvert(v01, num1, num2)
	v2 = binaryconvert(v02, num1, num2)
	t_on_12 = time.time()

	w1, w2, t_off_0, t_on_0 = BitAddition(v1, v2, q1, q2, num1, num2)
	t_off = t_off_12 - t_off_11 + t_off_0
	t_on = t_on_12 - t_on_11 + t_on_0
	
	return w1, w2


def TripleGenerate2n(numofrow):
	l = 4
	triplet = np.random.randint(0, 2**(l-2), (6, numofrow))
	#triplet = np.random.randint(0, 1, (6, numofrow))
	triplet[5] = (triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def TripleGenerate2nInner(numofrow):
	l = 3
	triplet = np.random.randint(0, 2**(l-2), (6, numofrow))
	triplet[5] = (triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def MultiplyMatrix2n(u1, u2, v1, v2, ta1, tb1, tc1, ta2, tb2, tc2):
	
	u_a = u1 - ta1 + u2 - ta2							# S1 <-> S2
	v_b = v1 - tb1 + v2 - tb2
	try:
		z1 = tc1 + u_a * tb1 + v_b * ta1					# S1
		z2 = tc2 + u_a * tb2 + v_b * ta2 + u_a * v_b		# S2
	except Exception as detail:
		print 
		print 'MM2N', detail 
		print 
		exit()
	return z1, z2 

def TripleMatrix(numofrow, numofcol):
	triplet = np.random.randint(0, 10, (6, numofrow, numofcol))
	triplet[5] = (triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def MultiplyMatrix(u1, u2, v1, v2):
	dim1 = u1.shape[0]
	dim2 = u1.shape[1]
	toff1 = time.time()
	trip = TripleMatrix(dim1, dim2)
	toff2 = time.time()
	ton1 = time.time()
	z1, z2 = MultiplyMatrix2n(u1, u2, v1, v2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	ton2 = time.time()
	toff = toff2-toff1
	ton = ton2-ton1
	return z1, z2, toff, ton	


def TripleTensor(numoftensor, numofrow, numofcol):
	triplet = np.random.randint(0, 10, (6, numoftensor, numofrow, numofcol))
	triplet[5] = (triplet[0]+triplet[3]) * (triplet[1]+triplet[4]) - triplet[2]
	return triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]

def MultiplyTensor(u1, u2, v1, v2):
	dim0 = u1.shape[0]
	dim1 = u1.shape[1]
	dim2 = u1.shape[2]
	trip = TripleTensor(dim0, dim1, dim2)
	z1, z2 = MultiplyMatrix2n(u1, u2, v1, v2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	return z1, z2

def NumProductss(u1, u2, v1, v2):
	# toff1 = time.time()
	
	trip = TripleGenerate2n(1)

	z1, z2 = MultiplyMatrix2n(u1, u2, v1, v2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])

	return z1, z2

def InnerProductss(a1, a2, b1, b2):  		# 求向量a的各元素的平方和，a=a1+a2
	dim = a1.shape
	# toff1 = time.time()

	trip = TripleGenerate2nInner(dim[0])
	# toff2 = time.time()
	# ton1 = time.time()
	product1, product2 = MultiplyMatrix2n(a1, a2, b1, b2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	# ton2 = time.time()
	# toff = toff2-toff1
	# ton = ton2-ton1
	#print np.sum(product1), np.sum(product2)
	return np.sum(product1), np.sum(product2)

def DotProductss(a1, a2, b1, b2):
	dim = b1.shape
	c1 = np.repeat(a1, len(b1))
	c2 = np.repeat(a2, len(b1))
	toff1 = time.time()
	trip = TripleGenerate2n(len(b1))
	toff2 = time.time()
	ton1 = time.time()
	product1, product2 = MultiplyMatrix2n(c1, c2, b1, b2, trip[0], trip[1], trip[2], trip[3], trip[4], trip[5])
	ton2 = time.time()
	toff = toff2-toff1
	ton = ton2-ton1
	return product1, product2

def MatMulss(A1, A2, B1, B2):
	dim1 = A1.shape[0]
	dim2 = B1.shape[1]
	C1 = np.repeat(A1, dim2, axis=0)
	C2 = np.repeat(A2, dim2, axis=0)
	D1 = np.tile(B1.T, (dim1, 1))
	D2 = np.tile(B2.T, (dim1, 1))
	E1, E2, toff, ton = MultiplyMatrix(C1, C2, D1, D2)
	F1 = np.sum(E1, axis=1)
	F2 = np.sum(E2, axis=1)
	F1 = F1.reshape((dim1,dim2), order='C')
	F2 = F2.reshape((dim1,dim2), order='C')
	return F1, F2

n = 32
