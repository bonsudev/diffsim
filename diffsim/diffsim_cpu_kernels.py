# ###########################################
# Filename: diffsim_cpu_kernels.py
# Calculate diffraction pattern from first principles
# Feature to parameterise displaced atoms (strain).
#
# Author: Marcus Newton
# Version 1.2
# Licence: GNU GPL 3
#
# ###########################################

import numpy
from numba import jit, njit, prange
import math
import numba


@jit(nopython=True, nogil=True)
def dotCPU(array1, array2):
	"""
	Calculate the dot product of two vectors.
	"""
	total = 0
	for i in range(len(array1)):
		total += array1[i]*array2[i]
	return total


@jit(nopython=True, nogil=True)
def crossCPU(array1, array2):
	"""
	Return cross product of two 3D vectors
	"""
	a1 = array1[0]
	a2 = array1[1]
	a3 = array1[2]
	b1 = array2[0]
	b2 = array2[1]
	b3 = array2[2]
	return a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1


@njit
def RXCPU(vector, rvector, angle):
	"""
	Rotate a vector about X axis by angle (rad)
	"""
	rmatrix_x = numpy.zeros((3,3), dtype=numpy.float32)

	rmatrix_x[0,0] = 1.0
	rmatrix_x[0,1] = 0.0
	rmatrix_x[0,2] = 0.0

	rmatrix_x[1,0] = 0.0
	rmatrix_x[1,1] = math.cos(angle)
	rmatrix_x[1,2] = -math.sin(angle)

	rmatrix_x[2,0] = 0.0
	rmatrix_x[2,1] = math.sin(angle)
	rmatrix_x[2,2] = math.cos(angle)

	rvector[0] = dotCPU(rmatrix_x[0,:],vector)
	rvector[1] = dotCPU(rmatrix_x[1,:],vector)
	rvector[2] = dotCPU(rmatrix_x[2,:],vector)


@njit
def RYCPU(vector, rvector, angle):
	"""
	Rotate a vector about Y axis by angle (rad)
	"""
	rmatrix_y = numpy.zeros((3,3), dtype=numpy.float32)

	rmatrix_y[0,0] = math.cos(angle)
	rmatrix_y[0,1] = 0.0
	rmatrix_y[0,2] = math.sin(angle)

	rmatrix_y[1,0] = 0.0
	rmatrix_y[1,1] = 1.0
	rmatrix_y[1,2] = 0.0

	rmatrix_y[2,0] = -math.sin(angle)
	rmatrix_y[2,1] = 0.0
	rmatrix_y[2,2] = math.cos(angle)

	rvector[0] = dotCPU(rmatrix_y[0,:],vector)
	rvector[1] = dotCPU(rmatrix_y[1,:],vector)
	rvector[2] = dotCPU(rmatrix_y[2,:],vector)


@njit
def RZCPU(vector, rvector, angle):
	"""
	Rotate a vector about Z axis by angle (rad)
	"""
	rmatrix_z = numpy.zeros((3,3), dtype=numpy.float32)

	rmatrix_z[0,0] = math.cos(angle)
	rmatrix_z[0,1] = -math.sin(angle)
	rmatrix_z[0,2] = 0.0

	rmatrix_z[1,0] = math.sin(angle)
	rmatrix_z[1,1] = math.cos(angle)
	rmatrix_z[1,2] = 0.0

	rmatrix_z[2,0] = 0.0
	rmatrix_z[2,1] = 0.0
	rmatrix_z[2,2] = 1.0

	rvector[0] = dotCPU(rmatrix_z[0,:],vector)
	rvector[1] = dotCPU(rmatrix_z[1,:],vector)
	rvector[2] = dotCPU(rmatrix_z[2,:],vector)


@jit(nopython=True, nogil=True)
def RotateToZCPU(vector):
	"""
	Rotate to z-axis and return rotation angles required to return back
	"""
	lxy = math.sqrt(vector[0]*vector[0]+vector[1]*vector[1])
	th_x = -math.atan2(lxy,vector[2])
	th_z = -math.atan2(vector[0],vector[1])
	mag = math.sqrt(dotCPU(vector,vector))
	return mag, th_x, th_z


@njit
def CalcKfpCPU(k_i, k_f, alphakf, Q, k_fp_out):
	"""
	Create k_fp vector from k_f and alpha.
	1) find vector k_i * k_f
	2) Rotate to Z and use angle to rotate k_f about z
	3) rotate back
	4) repeat for the other alpha angle
	"""
	knorm = crossCPU(k_i, k_f)
	knormmag, thx, thz = RotateToZCPU(knorm)

	k_fp_1 = numpy.zeros((3,), dtype=numpy.float32)
	k_fp_2 = numpy.zeros((3,), dtype=numpy.float32)
	k_fp_3 = numpy.zeros((3,), dtype=numpy.float32)
	k_fp_4 = numpy.zeros((3,), dtype=numpy.float32)
	k_fp_5 = numpy.zeros((3,), dtype=numpy.float32)

	RZCPU(k_f, k_fp_1, -thz)
	RXCPU(k_fp_1, k_fp_2, -thx)
	RZCPU(k_fp_2, k_fp_3, alphakf[1])
	RXCPU(k_fp_3, k_fp_4, thx)
	RZCPU(k_fp_4, k_fp_5, thz)

	Qmag, thx, thz = RotateToZCPU(Q)
	RZCPU(k_fp_5, k_fp_1, -thz)
	RXCPU(k_fp_1, k_fp_2, -thx)
	RZCPU(k_fp_2, k_fp_3, alphakf[0])
	RXCPU(k_fp_3, k_fp_4, thx)
	RZCPU(k_fp_4, k_fp_5, thz)

	k_fp_out[0] = k_fp_5[0]
	k_fp_out[1] = k_fp_5[1]
	k_fp_out[2] = k_fp_5[2]


@njit(parallel=True)
def CalcRefractionCPU(kf, alat, bins, shapearray, refractionarray):
	x, y, z = shapearray.shape
	alat_inv = numpy.linalg.inv(alat.T)
	kf_lat = numpy.dot(alat_inv, kf)
	kf_norm = kf_lat * (1.0/math.sqrt(dotCPU(kf_lat,kf_lat)))
	p = numpy.zeros((5,), dtype=numpy.double) ## i,j,k, shape bool, count
	pd = numpy.zeros((3,), dtype=numpy.double)
	for idk in prange(z):
		for idj in prange(y):
			for idi in prange(x):
				if shapearray[idi,idj,idk] > 0:
					p[0] = idi
					p[1] = idj
					p[2] = idk
					p[3] = 1.0
					p[4] = 0.0
					while ((p[0] >= 0 and p[0] < x) and (p[1] >= 0 and p[1] < y) and (p[2] >= 0 and p[2] < z) and p[3] > 0):
						pd[0] = (p[0] - idi)*bins[0] * alat[0,0] + (p[1] - idj)*bins[1] * alat[1,0] + (p[2] - idk)*bins[2] * alat[2,0]
						pd[1] = (p[0] - idi)*bins[0] * alat[0,1] + (p[1] - idj)*bins[1] * alat[1,1] + (p[2] - idk)*bins[2] * alat[2,1]
						pd[2] = (p[0] - idi)*bins[0] * alat[0,2] + (p[1] - idj)*bins[1] * alat[1,2] + (p[2] - idk)*bins[2] * alat[2,2]
						refractionarray[idi,idj,idk] = math.sqrt( dotCPU(pd,pd) )
						p[4] += 1.0
						p[0] = math.floor(idi + p[4]*kf_norm[0])
						p[1] = math.floor(idj + p[4]*kf_norm[1])
						p[2] = math.floor(idk + p[4]*kf_norm[2])
						p[3] = shapearray[int(p[0])%x,int(p[1])%y,int(p[2])%z]



@jit(nopython=True, nogil=True)
def CalcDetAlphaCPU(idi, idj, pixelxy, detsize, R):
	"""Calculate K_ip rotation angles for the given (idi, idj) pixel."""
	alphakf_i = math.atan((10.0**(-6))*pixelxy[0] * (idi - detsize[0]/2.0)/R)
	alphakf_j = math.atan((10.0**(-6))*pixelxy[1] * (idj - detsize[1]/2.0)/R)
	return alphakf_i, alphakf_j


@njit(parallel=True)
def CalcPixelScatterCPU(
	atoms, k_f, k_i, Q, atomR,
	exposure, binbits, pixelxy, detsize,
	r0, flux, kmag, R, refidx,
	flatshapearray, flatrefractionarray, gam_out
	):
	"""Calculate the scattering intensity for each pixel.

	Parameters
	----------
	atoms: numpy float array (natoms, 4)
	k_i: numpy float array (nz, 3)
	k_f: numpy float array (nx, ny, nz, 3)
	Q: numpy float array (nz, 3)
	atomR: numpy float array (natoms, nunitcells, 3)
	exposure: float
	binbits: int
	pixelxy: numpy float array (2,)
	detsize: numpy int array (2,)
	r0: float
	flux: float
	kmag: float
	R: float
	refidx: complex float
	flatshapearray: numpy float array (nunitcells,)
	flatrefractionarray,: numpy float array (nunitcells,)

	Return
	------
	gam_out: numpy float array (nx, ny, nz)
	"""
	x, y, z = gam_out.shape
	
	k_fp = numpy.zeros((3,), dtype=numpy.float32)
	
	refidxRe = refidx.real
	refidxIm = refidx.imag
	
	for idk in prange(z):
		for idj in prange(y):
			for idi in prange(x):
				alphakf = CalcDetAlphaCPU(idi, idj, pixelxy, detsize, R)
				CalcKfpCPU(k_i[idk], k_f[idk], alphakf, Q[idk], k_fp)
				q0 = k_fp[0] - k_i[idk,0]
				q1 = k_fp[1] - k_i[idk,1]
				q2 = k_fp[2] - k_i[idk,2]
				expsumRe = 0.0
				expsumIm = 0.0
				n = len(atoms)
				m = len(atomR[0])
				for i in range(1,n,1):
					innersumRe = 0.0
					innersumIm = 0.0
					for j in range(m):
						atomDotProduct = atomR[i,j,0] * q0 + atomR[i,j,1] * q1 + atomR[i,j,2] * q2 
						refracRe = kmag*(refidxRe-1.0)*flatrefractionarray[j]
						refracIm = kmag*refidxIm*flatrefractionarray[j]
						innersumRe += flatshapearray[j] * (math.cos(atomDotProduct+refracRe)*math.cosh(refracIm) - math.cos(atomDotProduct+refracRe)*math.sinh(refracIm))
						innersumIm += flatshapearray[j] * (math.sin(atomDotProduct+refracRe)*math.cosh(refracIm) - math.sin(atomDotProduct+refracRe)*math.sinh(refracIm))
					expsumRe += atoms[i, 0] * innersumRe
					expsumIm += atoms[i, 0] * innersumIm

				gam_out[idi,idj,idk] = exposure*pixelxy[0]*pixelxy[1]*r0*r0*binbits*binbits*\
					flux*(expsumRe*expsumRe+expsumIm*expsumIm)/(R*R)
				


@njit(parallel=True)
def CalcPixelScatterNoAtomFuncCPU(
	k_f, k_i, Q, Rvec,
	exposure, binbits, pixelxy, detsize,
	r0, flux, kmag, R, refidx, S,
	flatshapearray, flatrefractionarray, gam_out 
	):
	"""Calculate the scattering intensity for a pixel, when there are no atom
	displacement functions.

	Parameters
	----------
	k_i: numpy float array (nz, 3)
	k_f: numpy float array (nx, ny, nz, 3)
	Q: numpy float array (nz, 3)
	Rvec: numpy float array (nunitcells, 3)
	exposure: float
	binbits: int
	pixelxy: numpy float array (2,)
	detsize: numpy int array (2,)
	r0: float
	flux: float
	kmag: float
	R: float
	refidx: complex float
	flatshapearray: numpy float array (nunitcells,)
	flatrefractionarray,: numpy float array (nunitcells,)
	S: numpy complex float array (nz,)

	Return
	------
	gam_out: numpy float array (nx, ny, nz)
	"""
	x, y, z = gam_out.shape
	
	k_fp = numpy.zeros((3,), dtype=numpy.float32)
	
	refidxRe = refidx.real
	refidxIm = refidx.imag
	
	for idk in prange(z):
		for idj in prange(y):
			for idi in prange(x):
				alphakf = CalcDetAlphaCPU(idi, idj, pixelxy, detsize, R)
				CalcKfpCPU(k_i[idk], k_f[idk], alphakf, Q[idk], k_fp)
				q0 = k_fp[0] - k_i[idk,0]
				q1 = k_fp[1] - k_i[idk,1]
				q2 = k_fp[2] - k_i[idk,2]
				expsumRe = 0.0
				expsumIm = 0.0
				n = len(flatshapearray)
				for i in range(1,n,1):
					RVecDotProduct = Rvec[i,0]*q0 + Rvec[i,1]*q1 + Rvec[i,2]*q2
					refracRe = kmag*(refidxRe-1.0)*flatrefractionarray[i]
					refracIm = kmag*refidxIm*flatrefractionarray[i]
					expsumRe += flatshapearray[i]*(math.cos(RVecDotProduct+refracRe)*math.cosh(refracIm) - math.cos(RVecDotProduct+refracRe)*math.sinh(refracIm))
					expsumIm += flatshapearray[i]*(math.sin(RVecDotProduct+refracRe)*math.cosh(refracIm) - math.sin(RVecDotProduct+refracRe)*math.sinh(refracIm))

				gam_out[idi,idj,idk] = exposure*pixelxy[0]*pixelxy[1]*r0*r0*binbits*binbits*\
					flux*(S[idk].real*S[idk].real+S[idk].imag*S[idk].imag)*(expsumRe*expsumRe+expsumIm*expsumIm)/(R*R)
				
	

