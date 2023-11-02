# ###########################################
# Filename: diffsim_mpi_kernels.py
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

from mpi4py import MPI

from .diffsim_cpu_kernels import *


def Subdivide(n, shape):
	x,y,z = shape
	div = [1,1,1]
	i = 0
	while i < 6:
		if ((div[0]+1)*div[1]*div[2] < n) and (div[0]+1) <= x:
			div[0] += 1
			i = 0
		if (div[0]*(div[1]+1)*div[2] < n) and (div[1]+1) <= y:
			div[1] += 1
			i = 0
		if (div[0]*div[1]*(div[2]+1) < n) and (div[2]+1) <= z:
			div[2] += 1
			i = 0
		i += 1
	chunks = numpy.zeros((n,6), dtype=numpy.int64)
	ii=0
	xblk = x//div[0]
	yblk = y//div[1]
	zblk = z//div[2]
	for k in range(div[2]+1):
		for j in range(div[1]):
			for i in range(div[0]):
				if ii >= n:
					break
				chunks[ii,:] = [xblk*i, xblk*(i+1), yblk*j, yblk*(j+1), zblk*k, zblk*(k+1)]
				ii += 1
	for ii in range(div[0]*div[1]*div[2],n,1):
		if chunks[ii,1] > x:
			chunks[ii,1] = x
		if chunks[ii,3] > y:
			chunks[ii,3] = y
		if chunks[ii,5] > z:
			chunks[ii,5] = z
	chunks[-1,1] = x
	chunks[-1,3] = y
	chunks[-1,5] = z
	return chunks


@njit(parallel=True)
def CalcPixelScatterMPI(
	block, atoms, k_f, k_i, Q, atomR,
	exposure, binbits, pixelxy, detsize,
	r0, flux, kmag, R, refidx,
	flatshapearray, flatrefractionarray, diffarrayblock
	):
	"""Calculate the scattering intensity for a pixel, when there are no atom
	displacement functions.

	Parameters
	----------
	block: iteration dimensions
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
	diffarrayblock: numpy float array (blkx, blky, blkz)
	"""
	
	k_fp = numpy.zeros((3,), dtype=numpy.float32)
	
	refidxRe = refidx.real
	refidxIm = refidx.imag
	
	for idk in prange(block[4], block[5], 1):
		for idj in prange(block[2], block[3], 1):
			for idi in prange(block[0], block[1], 1):
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
				blki = idi - block[0]
				blkj = idj - block[2]
				blkk = idk - block[4]
				diffarrayblock[blki,blkj,blkk] = exposure*pixelxy[0]*pixelxy[1]*r0*r0*binbits*binbits*\
					flux*(expsumRe*expsumRe+expsumIm*expsumIm)/(R*R)
	##


@njit(parallel=True)
def CalcPixelScatterNoAtomFuncMPI(
	block, k_f, k_i, Q, Rvec,
	exposure, binbits, pixelxy, detsize,
	r0, flux, kmag, R, refidx, S,
	flatshapearray, flatrefractionarray, diffarrayblock
	):
	"""Calculate the scattering intensity for a pixel, when there are no atom
	displacement functions.

	Parameters
	----------
	block: iteration dimensions
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
	diffarrayblock: numpy float array (blkx, blky, blkz)
	"""
	
	k_fp = numpy.zeros((3,), dtype=numpy.float32)
	
	refidxRe = refidx.real
	refidxIm = refidx.imag
	
	for idk in prange(block[4], block[5], 1):
		for idj in prange(block[2], block[3], 1):
			for idi in prange(block[0], block[1], 1):
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
				blki = idi - block[0]
				blkj = idj - block[2]
				blkk = idk - block[4]
				diffarrayblock[blki,blkj,blkk] = exposure*pixelxy[0]*pixelxy[1]*r0*r0*binbits*binbits*\
					flux*(S[idk].real*S[idk].real+S[idk].imag*S[idk].imag)*(expsumRe*expsumRe+expsumIm*expsumIm)/(R*R)
	##
	






