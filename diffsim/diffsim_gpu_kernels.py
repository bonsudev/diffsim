# ###########################################
# Filename: diffsim_gpu_kernels.py
# Calculate diffraction pattern from first principles
# Feature to parameterise displaced atoms (strain).
#
# Author: Marcus Newton
# Version 1.0
# Licence: GNU GPL 3
#
# ###########################################

import numpy
from numba import cuda
import math
import numba


FLOAT = numba.float32


@cuda.jit(device=True)
def dotGPU(array1, array2):
	"""
	Calculate the dot product of two vectors.
	"""
	total = 0
	for i in range(len(array1)):
		total += array1[i]*array2[i]
	return total


@cuda.jit(device=True)
def crossGPU(array1, array2):
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


@cuda.jit(device=True)
def RXGPU(vector, rvector, angle):
	"""
	Rotate a vector about X axis by angle (rad)
	"""
	rmatrix_x = cuda.local.array((3,3), FLOAT)

	rmatrix_x[0,0] = 1.0
	rmatrix_x[0,1] = 0.0
	rmatrix_x[0,2] = 0.0

	rmatrix_x[1,0] = 0.0
	rmatrix_x[1,1] = math.cos(angle)
	rmatrix_x[1,2] = -math.sin(angle)

	rmatrix_x[2,0] = 0.0
	rmatrix_x[2,1] = math.sin(angle)
	rmatrix_x[2,2] = math.cos(angle)

	rvector[0] = dotGPU(rmatrix_x[0,:],vector)
	rvector[1] = dotGPU(rmatrix_x[1,:],vector)
	rvector[2] = dotGPU(rmatrix_x[2,:],vector)


@cuda.jit(device=True)
def RYGPU(vector, rvector, angle):
	"""
	Rotate a vector about Y axis by angle (rad)
	"""
	rmatrix_y = cuda.local.array((3,3), FLOAT)

	rmatrix_y[0,0] = math.cos(angle)
	rmatrix_y[0,1] = 0.0
	rmatrix_y[0,2] = math.sin(angle)

	rmatrix_y[1,0] = 0.0
	rmatrix_y[1,1] = 1.0
	rmatrix_y[1,2] = 0.0

	rmatrix_y[2,0] = -math.sin(angle)
	rmatrix_y[2,1] = 0.0
	rmatrix_y[2,2] = math.cos(angle)

	rvector[0] = dotGPU(rmatrix_y[0,:],vector)
	rvector[1] = dotGPU(rmatrix_y[1,:],vector)
	rvector[2] = dotGPU(rmatrix_y[2,:],vector)


@cuda.jit(device=True)
def RZGPU(vector, rvector, angle):
	"""
	Rotate a vector about Z axis by angle (rad)
	"""
	rmatrix_z = cuda.local.array((3,3), FLOAT)

	rmatrix_z[0,0] = math.cos(angle)
	rmatrix_z[0,1] = -math.sin(angle)
	rmatrix_z[0,2] = 0.0

	rmatrix_z[1,0] = math.sin(angle)
	rmatrix_z[1,1] = math.cos(angle)
	rmatrix_z[1,2] = 0.0

	rmatrix_z[2,0] = 0.0
	rmatrix_z[2,1] = 0.0
	rmatrix_z[2,2] = 1.0

	rvector[0] = dotGPU(rmatrix_z[0,:],vector)
	rvector[1] = dotGPU(rmatrix_z[1,:],vector)
	rvector[2] = dotGPU(rmatrix_z[2,:],vector)


@cuda.jit(device=True)
def RotateToZGPU(vector):
	"""
	Rotate to z-axis and return rotation angles required to return back
	"""
	lxy = math.sqrt(vector[0]*vector[0]+vector[1]*vector[1])
	th_x = -math.atan2(lxy,vector[2])
	th_z = -math.atan2(vector[0],vector[1])
	mag = math.sqrt(dotGPU(vector,vector))
	return mag, th_x, th_z


@cuda.jit(device=True)
def CalcKfpGPU(k_i, k_f, alphakf, Q, k_fp_out):
	"""
	Create k_fp vector from k_f and alpha.
	1) find vector k_i * k_f
	2) Rotate to Z and use angle to rotate k_f about z
	3) rotate back
	4) repeat for the other alpha angle
	"""
	knorm = crossGPU(k_i, k_f)
	knormmag, thx, thz = RotateToZGPU(knorm)

	# Allocate memory for each step of the calculation, to avoid weird issues
	# with re-using the same memory
	k_fp_1 = cuda.local.array((3,), FLOAT)
	k_fp_2 = cuda.local.array((3,), FLOAT)
	k_fp_3 = cuda.local.array((3,), FLOAT)
	k_fp_4 = cuda.local.array((3,), FLOAT)
	k_fp_5 = cuda.local.array((3,), FLOAT)

	RZGPU(k_f, k_fp_1, -thz)
	RXGPU(k_fp_1, k_fp_2, -thx)
	RZGPU(k_fp_2, k_fp_3, alphakf[1])
	RXGPU(k_fp_3, k_fp_4, thx)
	RZGPU(k_fp_4, k_fp_5, thz)

	Qmag, thx, thz = RotateToZGPU(Q)
	RZGPU(k_fp_5, k_fp_1, -thz)
	RXGPU(k_fp_1, k_fp_2, -thx)
	RZGPU(k_fp_2, k_fp_3, alphakf[0])
	RXGPU(k_fp_3, k_fp_4, thx)
	RZGPU(k_fp_4, k_fp_5, thz)

	k_fp_out[0] = k_fp_5[0]
	k_fp_out[1] = k_fp_5[1]
	k_fp_out[2] = k_fp_5[2]


@cuda.jit(device=True)
def CalcDetAlphaGPU(idi, idj, pixelxy, detsize, R):
	"""Calculate K_ip rotation angles for the given (idi, idj) pixel."""
	alphakf_i = math.atan((10.0**(-6))*pixelxy[0] * (idi - detsize[0]/2.0)/R)
	alphakf_j = math.atan((10.0**(-6))*pixelxy[1] * (idj - detsize[1]/2.0)/R)
	return alphakf_i, alphakf_j


@cuda.jit()
def CalcPixelScatterGPU(atoms, k_i, k_f, atomR, Q, exposure, binbits, pixelxy,
												detsize, r0, flux, R, flatshapearray, gam_out):
	"""Calculate the scattering intensity for a pixel.

	Computes the intensity for the pixel given by the ID of the current GPU
	thread, determined by idi, idj and idk.	Units: photons.

	Parameters
	----------
	atoms: numpy float array (natoms, 4)
	k_i: numpy float array (nz, 3)
	k_f: numpy float array (nx, ny, nz, 3)
	atomR: numpy float array (natoms, nunitcells, 3)
	Q: numpy float array (nz, 3)
	exposure: float
	binbits: int
	pixelxy: numpy float array (2,)
	detsize: numpy int array (2,)
	r0: float
	flux: float
	R: float
	flatshapearray: numpy float array (nunitcells,)

	Return
	------
	gam_out: numpy float array (nx, ny, nz)
	"""
	idi = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
	idj = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
	idk = cuda.blockIdx.z*cuda.blockDim.z + cuda.threadIdx.z

	k_fp = cuda.local.array((3,), FLOAT)
	alphakf = CalcDetAlphaGPU(idi, idj, pixelxy, detsize, R)
	CalcKfpGPU(k_i[idk], k_f[idk], alphakf, Q[idk], k_fp)

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
			atomDotProduct = atomR[i,j,0] * q0 + \
					atomR[i,j,1] * q1 + \
					atomR[i,j,2] * q2
			innersumRe += flatshapearray[j] * math.cos(atomDotProduct)
			innersumIm += flatshapearray[j] * math.sin(atomDotProduct)
		expsumRe += atoms[i, 0] * innersumRe
		expsumIm += atoms[i, 0] * innersumIm

	gam_out[idi,idj,idk] = exposure*pixelxy[0]*pixelxy[1]*r0*r0*binbits*binbits*\
		flux*(expsumRe*expsumRe+expsumIm*expsumIm)/(R*R)


@cuda.jit
def CalcPixelScatterNoAtomFuncGPU(k_f, k_i, Q, Rvec, exposure, binbits, pixelxy,
									detsize, r0, flux, R, flatshapearray, S, gam_out):
	"""Calculate the scattering intensity for a pixel, when there are no atom
	displacement functions.

	Computes the intensity for the pixel given by the ID of the current GPU
	thread, determined by idi, idj and idk.	Units: photons.

	Parameters
	----------
	atoms: numpy float array (natoms, 4)
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
	R: float
	flatshapearray: numpy float array (nunitcells,)
	S: numpy complex float array (nz,)

	Return
	------
	gam_out: numpy float array (nx, ny, nz)
	"""
	idi = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
	idj = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
	idk = cuda.blockIdx.z*cuda.blockDim.z + cuda.threadIdx.z

	k_fp = cuda.local.array(3, FLOAT)
	alphakf = CalcDetAlphaGPU(idi, idj, pixelxy, detsize, R)
	CalcKfpGPU(k_i[idk], k_f[idk], alphakf, Q[idk], k_fp)

	q0 = k_fp[0] - k_i[idk,0]
	q1 = k_fp[1] - k_i[idk,1]
	q2 = k_fp[2] - k_i[idk,2]
	expsumRe = 0.0
	expsumIm = 0.0
	n = len(flatshapearray)
	for i in range(1,n,1):
		RVecDotProduct = Rvec[i,0]*q0+\
				Rvec[i,1]*q1+\
				Rvec[i,2]*q2
		expsumRe += flatshapearray[i]*math.cos(RVecDotProduct)
		expsumIm += flatshapearray[i]*math.sin(RVecDotProduct)

	gam_out[idi,idj,idk] = exposure*pixelxy[0]*pixelxy[1]*r0*r0*binbits*binbits*\
		flux*(S[idk].real*S[idk].real+S[idk].imag*S[idk].imag)*(expsumRe*expsumRe+expsumIm*expsumIm)/(R*R)


