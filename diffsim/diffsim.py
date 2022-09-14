# ###########################################
# Filename: diffsim.py
# Calculate diffraction pattern from first principles
# Feature to parameterise displaced atoms (strain).
#
# Author: Marcus Newton
# Version 1.0
# Licence: GNU GPL 3
#
# ###########################################

import numpy
from math import sqrt, atan, atan2, asin, sin, cos, fabs, ceil
from mpi4py import MPI
import threading
from time import strftime
from inspect import getsource, signature
from . import diffsim_gpu_kernels
from os import sched_getaffinity

class DiffSim():
	"""
	Simulate Bragg coherent diffraction	from first principles in the kinematic
	approximation. The Euler coordinate system is used. Single precision is the
	default.
	"""
	def __init__(self):
		comm = MPI.COMM_WORLD
		self.mpi_size = comm.Get_size()
		self.mpi_rank = comm.Get_rank()
		self.float = numpy.float32
		self.float16 = numpy.float16
		self.int = numpy.int32
		self.shortint = numpy.int8
		self.complex = numpy.complex64
		self.pi = numpy.pi
		self.rad2deg = 180.0/self.pi
		self.h = 6.62607004 * 10.0**(-34)
		self.e = 1.60217662 * 10.0**(-19)
		self.c = 299792458.0
		self.r0 = 2.81794032 * 10.0**(-15)
		self.n0 = 1.0
		self.waveln = 0.0 # (nm)
		self.energy = 0.0 # (keV)
		self.k = 0.0 # (1/nm)
		self.theta = 0.0 #(rad)
		self.dtheta = 0.0 #(rad)
		self.thetamax = 0.0 #(rad)
		self.resolution = 0.0 # nm
		self.beta = 2.0 # oversampling ratio
		self.R = 0.0 # sample detector distance
		self.alattice = numpy.zeros((3,3), dtype=self.float)
		self.glattice = numpy.zeros((3,3), dtype=self.float)
		self.bin = 10*numpy.ones((3), dtype=self.int)
		self.binbits = 1
		self.Q = numpy.zeros((3), dtype=self.float)
		self.Qhat = numpy.zeros((3), dtype=self.float)
		self.Qmag = 0.0
		self.S = 0.0
		self.flux = 10.0**8 # photons/um^2/s 
		self.exposure = 1.0 # (s)
		self.Isc = 0.0
		self.k_i = numpy.zeros((3), dtype=self.float)
		self.k_f = numpy.zeros((3), dtype=self.float)
		self.k_fp = numpy.zeros((3), dtype=self.float)
		self.hkl = numpy.zeros((3), dtype=self.int)
		self.atoms = numpy.zeros((1,4), dtype=self.float)
		self.atomfunctions = []
		self.pixelxy = numpy.zeros((2), dtype=self.float)
		self.detsize = numpy.zeros((2), dtype=self.int)
		self.alphakf = numpy.zeros((2), dtype=self.float)
		self.shapearray = None # Object shape function
		self.flatshapearray = None
		self.ijks = None
		self.Rvec = None
		self.diffarray = None # Diffraction pattern
		self.diffarrayblock = None # Diffraction pattern block
		self.object = None
		self.flatobject = None
		self.coordarray = None
		self.meta_name = "" # Simulation name
		self.datestr = ""
	def SetLatticeVector(self, index, vector):
		"""
		Set a unit cell lattice vector.
		Index: in [0,1,2] .
		vector: numpy double array
		of length 3.
		units: nm
		"""
		self.alattice[index,:] = vector[:]
	def SetMillerIndices(self, indices):
		"""
		Set miller indicies.
		numpy int array of length 3.
		"""
		self.hkl[:] = indices[:]
	def AddAtom(self, factor, x,y,z):
		"""
		Add atom to unit cell using
		Wyckoff internal coordinates.
		factor: scattering factor of atom
		x,y,z: fractions of each unit cell
		direction
		"""
		self.atoms = numpy.append(self.atoms, [[factor,x,y,z]], axis=0).astype(self.float)
	def AddAtomFunction(self, index, function):
		"""
		Displacement function for each
		atom in the unit cell as a function
		of position (i.e. the strain field).
		index: atom index (1,2,3, ....). not zero!
		function: function operating on atom
		The input to each function is an
		N * 3 (i,j,k) array.
		The output should be an N * 3 array
		of Wyckoff coordinates for atom at
		index.
		"""
		self.atomfunctions.append([index,function])
	def SetSuperCell(self, x,y,z):
		self.SetUnitCellBin(x,y,z)
	def SetUnitCellBin(self, x,y,z):
		"""
		Bin unit cells to improve speed of compuation.
		Resolution is unaffected provided the bins are
		smaller than that due to geometric constraints.
		"""
		self.bin[0] = self.int(x)
		self.bin[1] = self.int(y)
		self.bin[2] = self.int(z)
		self.binbits = self.int(self.bin[0]*self.bin[1]*self.bin[2])
	def BinArray(self, arobj):
		amp = self.bin[0]*self.bin[1]*self.bin[2]
		shp = arobj.shape
		nx = (shp[0]+self.bin[0] -1)//self.bin[0]
		ny = (shp[1]+self.bin[1] -1)//self.bin[1]
		nz = (shp[2]+self.bin[2] -1)//self.bin[2]
		arraybin = numpy.zeros((nx,ny,nz), dtype=self.float, order='C')
		for k in range(self.bin[2]):
			for j in range(self.bin[1]):
				for i in range(self.bin[0]):
					subshp = arobj[i::self.bin[0],j::self.bin[1],k::self.bin[2]].shape
					arraybin[(nx-subshp[0]):nx,(ny-subshp[1]):ny,(nz-subshp[2]):nz] +=\
					arobj[i::self.bin[0],j::self.bin[1],k::self.bin[2]]
		return (1.0/amp)*arraybin

	def SetShapeArray(self, filename):
		"""
		Set the shape function array.
		uint8 numpy array:
		0 - no unit cell
		> 0 - unit cell
		dtype: numpy.uint8
		"""
		self.shapearray = numpy.ascontiguousarray(numpy.load(filename), dtype=numpy.uint8)
		shape = self.shapearray.shape
		n = numpy.prod(self.shapearray.shape)
		self.flatshapearray = numpy.ascontiguousarray(self.shapearray.reshape((n)), dtype=numpy.uint8)
		self.object = numpy.zeros(shape, dtype=numpy.cdouble)
		self.flatobject = self.object.reshape((n))
		if len(self.atomfunctions) > 0:
			self.ijks = numpy.ascontiguousarray(numpy.transpose(numpy.unravel_index(numpy.arange(n), shape))*self.bin, dtype=numpy.int32)
			self.Rvec = numpy.ascontiguousarray(numpy.dot(self.ijks,self.alattice), dtype=self.float)
		else:
			self.Rvec = numpy.ascontiguousarray(numpy.dot(numpy.transpose(numpy.unravel_index(numpy.arange(n), shape))*self.bin,self.alattice), dtype=self.float)

	def _SetShapeArrayFromNormals(self, arraysize, normals):
		"""
		Create a shape function array
		of 3D size 'arraysize' with facets
		defined by normal vectors in
		the (n,6) list 'normals'.
		0-2 are the vector start coordinates
		3-5 are the vector end coordinates
		"""
		arobj = numpy.zeros(arraysize, dtype=self.shortint)
		shape = arobj.shape
		n = numpy.prod(arobj.shape)
		if n < numpy.iinfo(numpy.uint32).max:
			xyzs = numpy.transpose(numpy.unravel_index(numpy.arange(n, dtype=numpy.uint32), shape))
		else:
			xyzs = numpy.transpose(numpy.unravel_index(numpy.arange(n, dtype=numpy.uint64), shape))
		flatarobj = arobj.reshape((n))
		surfaces = numpy.array(normals)
		init = surfaces[0,0:3]
		term = surfaces[0,3:6]
		norm = term - init
		norm = norm * (1.0/numpy.sqrt(numpy.dot(norm,norm)))
		inout = numpy.dot((xyzs - init),norm) < 0
		for i in range(1,len(surfaces),1):
			init = surfaces[i,0:3]
			term = surfaces[i,3:6]
			norm = term - init
			norm = norm * (1.0/numpy.sqrt(numpy.dot(norm,norm)))
			inout *= numpy.dot((xyzs - init),norm) < 0
		flatarobj[inout] = 1
		#self.shapearray = arobj.astype(numpy.uint8)
		#self.flatshapearray = flatarobj
		#self.ijks = xyzs
		#self.Rvec = numpy.dot(self.ijks,self.alattice)
		self.shapearray = numpy.ascontiguousarray(self.BinArray(arobj), dtype=numpy.uint8)
		del arobj
		del flatarobj
		del xyzs
		del inout
		shape = self.shapearray.shape
		n = numpy.prod(self.shapearray.shape)
		self.flatshapearray = numpy.ascontiguousarray(self.shapearray.reshape((n)), dtype=numpy.uint8)
		self.object = numpy.zeros(shape, dtype=numpy.cdouble)
		self.flatobject = self.object.reshape((n))
		if len(self.atomfunctions) > 0:
			self.ijks = numpy.ascontiguousarray(numpy.transpose(numpy.unravel_index(numpy.arange(n), shape))*self.bin, dtype=numpy.int32)
			self.Rvec = numpy.ascontiguousarray(numpy.dot(self.ijks,self.alattice), dtype=self.float)
		else:
			self.Rvec = numpy.ascontiguousarray(numpy.dot(numpy.transpose(numpy.unravel_index(numpy.arange(n), shape))*self.bin,self.alattice), dtype=self.float)
	def SetShapeArrayFromNormals(self, arraysize, normals):
		# arobj = numpy.zeros(arraysize, dtype=self.float)
		shp = arraysize
		nx = (shp[0]+self.bin[0] -1)//self.bin[0]
		ny = (shp[1]+self.bin[1] -1)//self.bin[1]
		nz = (shp[2]+self.bin[2] -1)//self.bin[2]
		npoints = nx*ny*nz
		n = self.mpi_size
		ni = self.mpi_rank
		comm = MPI.COMM_WORLD
		# if ni == 0:
		# 	self._SetShapeArrayFromNormals(arraysize, normals)
		# else:
		# 	self.shapearray = numpy.empty((nx,ny,nz), dtype=numpy.uint8, order='C')
		# comm.Bcast(self.shapearray, root=0)
		self._SetShapeArrayFromNormals(arraysize, normals)
		if ni != 0:
			self.flatshapearray = numpy.empty(npoints, dtype=numpy.uint8, order='C')
		comm.Bcast(self.flatshapearray, root=0)
		if len(self.atomfunctions) > 0:
			if ni != 0:
				self.ijks = numpy.empty([npoints,3], dtype=numpy.int32, order='C')
			comm.Bcast(self.ijks, root=0)
		if ni != 0:
			self.Rvec = numpy.empty([npoints,3], dtype=self.float, order='C')
		comm.Bcast(self.Rvec, root=0)
	def SaveShapeArray(self, name):
		if self.mpi_rank == 0:
			numpy.save(name, self.shapearray)
	def SetDetPixelSize(self, dx, dy):
		"""
		Set the detector pixel size
		in the X and Y directions
		units: microns
		"""
		self.pixelxy[0] = dx
		self.pixelxy[1] = dy
	def SetDetSize(self, DN1, DN2):
		"""
		Set the number of detector
		pixels in the X and Y directions.
		"""
		self.detsize[0] = self.int(DN1)
		self.detsize[1] = self.int(DN2)
	def SetRecipLattice(self):
		"""
		Set Reciprocal lattice vectors
		from lattice vectors.
		"""
		gv = numpy.dot(self.alattice[0,:], numpy.cross(self.alattice[1,:], self.alattice[2,:]))
		self.glattice[0,:] = 2.0*self.pi*numpy.cross(self.alattice[1,:], self.alattice[2,:]) / gv
		self.glattice[1,:] = 2.0*self.pi*numpy.cross(self.alattice[2,:], self.alattice[0,:]) / gv
		self.glattice[2,:] = 2.0*self.pi*numpy.cross(self.alattice[0,:], self.alattice[1,:]) / gv
	def SetWavelength(self, waveln):
		"""
		Set Wavelength in nm
		"""
		self.waveln = waveln
		self.energy = (10.0**6)*self.h*self.c/(self.waveln*self.e)
		self.k = 2.0*self.pi/self.waveln
	def SetEnergy(self, energy):
		"""
		Set energy in keV
		"""
		self.energy = energy
		self.waveln = (10.0**6)*self.h*self.c/(self.energy*self.e)
		self.k = 2.0*self.pi/self.waveln
	def SetFlux(self, flux):
		"""
		Takes a flux in photos/m^2/s and sets it in photons/um^2/s
		"""
		self.flux = self.float(flux*(10.0**(-12)))
	def SetBeta(self, beta):
		"""
		Set oversampling ratio
		"""
		self.beta = beta
	def SetQ(self):
		"""
		Set Q vector
		"""
		self.Q[0] = self.hkl[0]*self.glattice[0,0] + self.hkl[1]*self.glattice[1,0] + self.hkl[2]*self.glattice[2,0]
		self.Q[1] = self.hkl[0]*self.glattice[0,1] + self.hkl[1]*self.glattice[1,1] + self.hkl[2]*self.glattice[2,1]
		self.Q[2] = self.hkl[0]*self.glattice[0,2] + self.hkl[1]*self.glattice[1,2] + self.hkl[2]*self.glattice[2,2]
		self.Qmag = sqrt(numpy.dot(self.Q,self.Q))
		self.Qhat[:] = self.Q / self.Qmag
	def SetDTheta(self):
		alatticesq = numpy.square(self.alattice)
		a = sqrt(numpy.sum(alatticesq[0,:]))
		b = sqrt(numpy.sum(alatticesq[1,:]))
		c = sqrt(numpy.sum(alatticesq[2,:]))
		a_av = (a+b+c)/3.0
		self.dtheta = self.pi / (self.beta * self.Qmag *\
							((self.binbits*numpy.sum(self.shapearray))**(1/3))*\
							a_av)
	def ScaleVector(self, vector, scalefactor):
		"""
		Scale a vector by scalefactor
		"""
		vmag = sqrt(numpy.dot(vector,vector))
		vector = vector * (scalefactor/vmag)
		return vector
	def RX(self, vector, axis, angle):
		"""
		Rotate a vector about axis by angle (rad)
		"""
		rmatrix_x = numpy.array(((1.0,0.0,0.0),(0.0,cos(angle),-sin(angle)),(0.0,sin(angle),cos(angle))),dtype=self.float)
		rmatrix_y = numpy.array(((cos(angle),0.0,sin(angle)),(0.0,1.0,0.0),(-sin(angle),0.0,cos(angle))),dtype=self.float)
		rmatrix_z = numpy.array(((cos(angle),-sin(angle),0.0),(sin(angle),cos(angle),0.0),(0.0,0.0,1.0)),dtype=self.float)
		if axis == 0:
			rvector = numpy.dot(rmatrix_x,vector)
		if axis == 1:
			rvector = numpy.dot(rmatrix_y,vector)
		if axis == 2:
			rvector = numpy.dot(rmatrix_z,vector)
		return rvector
	def RotateToZ(self, vector):
		"""
		Rotate to z-axis and return rotation
		angles requires to return back
		"""
		lxy = sqrt(vector[0]*vector[0]+vector[1]*vector[1])
		th_x = -atan2(lxy,vector[2])
		th_z = -atan2(vector[0],vector[1])
		mag = sqrt(numpy.dot(vector,vector))
		return mag, th_x, th_z
	def SetKiKf(self):
		"""
		Calculate k vector components by rotating Q to
		z-axis and using |Q| = 2 |k| sin(theta) and align
		k vectors along x-axis. Then rotate both back.
		"""
		Qmag, thx, thz = self.RotateToZ(self.Q)
		self.theta = asin(Qmag/(2.0*self.k))
		self.k_i[0] = self.k*cos(self.theta)
		self.k_i[1] = 0.0
		self.k_i[2] = -self.k*sin(self.theta)
		self.k_f[0] = self.k*cos(self.theta)
		self.k_f[1] = 0.0
		self.k_f[2] = self.k*sin(self.theta)
		nk_i = self.RX(self.k_i, 0, thx)
		self.k_i = self.RX(nk_i, 2, thz)
		nk_f = self.RX(self.k_f, 0, thx)
		self.k_f = self.RX(nk_f, 2, thz)
		self.k_fp[:] = self.k_f[:]
	def SetR(self):
		"""
		Set sample detector distance (m)
		"""
		alatticesq = numpy.square(self.alattice)
		a = sqrt(numpy.sum(alatticesq[0,:]))
		b = sqrt(numpy.sum(alatticesq[1,:]))
		c = sqrt(numpy.sum(alatticesq[2,:]))
		a_av = (a+b+c)/3.0
		self.R = self.float(0.5*(self.pixelxy[0]+self.pixelxy[1])*self.beta*\
							self.k*((self.binbits*numpy.sum(self.shapearray))**(1/3))*\
							a_av*10.0**(-6))
	def CalcDetAlpha(self, i, j):
		"""
		Returns K_ip rotation angles.
		"""
		alphakf_i = 0.0
		alphakf_j = 0.0
		if i < self.detsize[0] and j < self.detsize[1]:
			alphakf_i = atan((10.0**(-6))*self.pixelxy[0] * (i - self.detsize[0]/2.0)/self.R)
			alphakf_j = atan((10.0**(-6))*self.pixelxy[1] * (j - self.detsize[1]/2.0)/self.R)
		return alphakf_i,alphakf_j
	def SetDetAlpha(self, i, j):
		"""
		Set K_ip rotation angles.
		"""
		self.alphakf[0],self.alphakf[1] = self.CalcDetAlpha(i,j)
	def CalcKfp(self, alphakf):
		"""
		Creat k_fp vector from k_f and alpha.
		1) find vector k_i * k_f.
		2) Rotate to Z and use angle to rotate k_f
		   about z
		3) rotate back.
		4) repeat for the other alpha angle.
		"""
		knorm = numpy.cross(self.k_i, self.k_f)
		knormmag, thx, thz = self.RotateToZ(knorm)
		k_f1 = self.RX(self.k_f, 2, -thz)
		k_f2 = self.RX(k_f1, 0, -thx)
		k_f3 = self.RX(k_f2, 2, alphakf[1])
		k_f4 = self.RX(k_f3, 0, thx)
		k_f5 = self.RX(k_f4, 2, thz)
		Qmag, thx, thz = self.RotateToZ(self.Q)
		k_f1 = self.RX(k_f5, 2, -thz)
		k_f2 = self.RX(k_f1, 0, -thx)
		k_f3 = self.RX(k_f2, 2, alphakf[0])
		k_f4 = self.RX(k_f3, 0, thx)
		return self.RX(k_f4, 2, thz)
	def RotateKiKfQ(self, theta, k_i, k_f, Q):
		"""
		Rotate ki and kf by amount theta
		"""
		knorm = numpy.cross(k_i, k_f)
		knormmag, thx, thz = self.RotateToZ(knorm)
		k_i1 = self.RX(k_i, 2, -thz)
		k_i2 = self.RX(k_i1, 0, -thx)
		k_f1 = self.RX(k_f, 2, -thz)
		k_f2 = self.RX(k_f1, 0, -thx)
		Q1 = self.RX(Q, 2, -thz)
		Q2 = self.RX(Q1, 0, -thx)
		k_i3 = self.RX(k_i2, 2, theta)
		k_f3 = self.RX(k_f2, 2, theta)
		Q3 = self.RX(Q2, 2, theta)
		k_i4 = self.RX(k_i3, 0, thx)
		self.k_i[:] = self.RX(k_i4, 2, thz)
		k_f4 = self.RX(k_f3, 0, thx)
		self.k_f[:] = self.RX(k_f4, 2, thz)
		Q4 = self.RX(Q3, 0, thx)
		self.Q[:] = self.RX(Q4, 2, thz)
		self.Qmag = sqrt(numpy.dot(self.Q,self.Q))
		self.Qhat[:] = self.Q / self.Qmag
	def SetKfp(self):
		"""
		Set  k_fp vector
		"""
		self.k_fp[:] = self.CalcKfp()
	def SetS(self):
		"""
		Calculate structure factor S(Q).
		This is only used for simple
		calculations where S(Q) is a
		constant
		"""
		n = len(self.atoms)
		real = 0.0
		imag = 0.0
		for i in range(1,n,1):
			f = self.atoms[i,0]
			r = numpy.sum(numpy.multiply(self.alattice.T,\
											self.atoms[i,1:]).T, axis=0)
			real += f * cos(numpy.dot(self.Q, r))
			imag += f * sin(numpy.dot(self.Q, r))
		self.S = real + imag*1j
	def SetExposureTime(self, expo):
		"""
		Set the exposure time for each
		frame in seconds.
		"""
		self.exposure = expo
	def SetIntensity(self):
		"""
		Scattering intensity at centroid.
		"""
		N = numpy.sum(self.shapearray)*self.binbits
		S = numpy.abs(self.S)
		self.Isc = self.exposure*self.r0*self.r0*self.flux*(10.0**12)*S*S*N*N/(self.R*self.R)
	def SetThetaMax(self, max=None):
		"""
		Set rocking curve width.
		"""
		alatticesq = numpy.square(self.alattice)
		a = sqrt(numpy.sum(alatticesq[0,:]))
		b = sqrt(numpy.sum(alatticesq[1,:]))
		c = sqrt(numpy.sum(alatticesq[2,:]))
		a_av = (a+b+c)/3.0
		wf = 2.0*self.pi/(a_av*(self.binbits*numpy.sum(self.shapearray))**(1/3))
		qmax = wf*sqrt(fabs((self.Isc*self.pixelxy[0]*self.pixelxy[1]*(10.0**(-12)))/self.n0 - 1.0))/self.pi
		if max is None:
			self.thetamax = 2.0*qmax/self.Qmag
			self.resolution = 2.0*self.pi/qmax
		else:
			self.thetamax = max
			qmax = self.Qmag*self.thetamax/2.0
			self.resolution = 2.0*self.pi/qmax
	def CalcAtomDisplacement(self):
		"""Calculate the displacement in position of all unit cells
		"""

		n = len(self.atoms)
		atomR = numpy.zeros((n,len(self.Rvec),3), dtype=self.float)
		for i in range(1,n,1):
			atomR[i,:] = self.Rvec + numpy.sum(numpy.multiply(self.alattice.T, self.atoms[i,1:]).T, axis=0)

			for f in self.atomfunctions:
				if f[0] == i:
					atomR[i,:] += numpy.dot(f[1](self.ijks), self.alattice)

		return atomR
	def CalcPixelScatter(self, k_fp, atomR):
		"""
		Calculate scattering intensity into a single pixel.
		Units: photons.
		@param k_fp numpy array of size 3
		@param self.k_i numpy array of size 3
		@param self.Rvec numpy array of size (3, ?)
		"""
		q = k_fp - self.k_i
		expsum = 0.0
		if len(self.atomfunctions) > 0:
			n = len(self.atoms)
			for i in range(1,n,1):
				expsum += self.atoms[i,0] * numpy.sum(self.flatshapearray * numpy.exp(1j*numpy.dot(atomR[i],q)))

			gam = self.exposure*self.pixelxy[0]*self.pixelxy[1]*self.r0*self.r0*\
						self.binbits * self.binbits *\
						self.flux*(expsum.real*expsum.real + expsum.imag*expsum.imag)/(self.R*self.R)
		else:
			expsum = numpy.sum(self.flatshapearray * numpy.exp(1j*numpy.dot(self.Rvec,q)))
			gam = self.exposure*self.pixelxy[0]*self.pixelxy[1]*self.r0*self.r0*\
						self.flux*(self.S.real*self.S.real+self.S.imag*self.S.imag)*\
						self.binbits * self.binbits *\
						(expsum.real*expsum.real + expsum.imag*expsum.imag)/(self.R*self.R)
		return gam
	def Prepare(self):
		"""
		Set up paramaters.
		The order here is important.
		"""
		self.datestr = strftime("%Y-%m-%d_%H.%M.%S")
		self.SetRecipLattice()
		self.SetQ()
		self.SetKiKf()
		self.SetR()
		self.SetDTheta()
		self.SetS()
		self.SetIntensity()
		self.SetThetaMax()
	def Subdivide(self, n, shape):
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
	def CalcObject(self):
		if len(self.atomfunctions) > 0:
			n = len(self.atoms)
			for i in range(1,n,1):
				if i == 1:
					atomR = self.Rvec + numpy.sum(numpy.multiply(self.alattice.T,\
												self.atoms[i,1:]).T, axis=0)
				else:
					atomR[:] = self.Rvec + numpy.sum(numpy.multiply(self.alattice.T,\
												self.atoms[i,1:]).T, axis=0)
				for f in self.atomfunctions:
					if f[0] == i:
						atomR[:] += numpy.dot(f[1](self.ijks),self.alattice)
				self.flatobject[:] += self.flatshapearray * numpy.exp(1j*numpy.dot(atomR,self.Q))
		else:
			self.object[:] = self.shapearray[:]
	def CalcObjectCoords(self):
		latt = self.alattice
		shp = self.object.shape
		mx = self.bin[0]
		my = self.bin[1]
		mz = self.bin[2]
		self.coordarray = numpy.zeros((self.object.size,3), dtype=numpy.double)
		X = numpy.array([[float(x)*mx, float(y)*my, float(z)*mz] for z in range(shp[2]) for y in range(shp[1]) for x in range(shp[0])], dtype=numpy.double)
		self.coordarray[:] = numpy.dot(X,latt)
	def _CalcThread(self, block, atomR, idk):
		for idj in range(block[2], block[3], 1):
			for idi in range(block[0], block[1], 1):
				alphakf = self.CalcDetAlpha(idi,idj)
				k_fp = self.CalcKfp(alphakf) # and this
				self.diffarray[idi,idj,idk] = self.CalcPixelScatter(k_fp, atomR) # and this
	def CalcRockingCurveThreads(self, nthreads=None):
		ki = self.k_i.copy()
		kf = self.k_f.copy()
		Q = self.Q.copy()
		steps = int(self.thetamax/self.dtheta)
		# #
		x,y,z = self.detsize[0],self.detsize[1],steps
		self.diffarray = numpy.zeros((self.detsize[0],self.detsize[1],steps), dtype=self.float)
		# Compute the atom displacements
		atomR = self.CalcAtomDisplacement()
		# threads
		if nthreads == None:
			n = len(sched_getaffinity(0))
		else:
			n = nthreads
		blocks = self.Subdivide(n, [x,y,1])
		# 
		for i in range(steps):
			thetarock = i*self.dtheta - self.thetamax/2.0
			self.RotateKiKfQ(thetarock, ki, kf, Q)
			self.SetS()
			threads = []
			for t in range(n):
				block = blocks[t]
				thread = threading.Thread(target=self._CalcThread, args=(block, atomR, i))
				thread.start()
				threads.append(thread)
			for thread in threads:
				if thread.is_alive():
					thread.join()
		##
	def CalcRockingCurveGPU(self):
		ki = self.k_i.copy()
		kf = self.k_f.copy()
		Q = self.Q.copy()
		steps = int(self.thetamax/self.dtheta)
		##
		x,y,z = self.detsize[0],self.detsize[1],steps
		self.diffarray = numpy.zeros((self.detsize[0],self.detsize[1],steps), dtype=self.float)

		# Arrays for pre-calculated quanities, to be passed to the GPU
		k_i_array = numpy.ones((z,3), dtype=self.float)
		k_f_array = numpy.ones((z, 3), dtype=self.float)
		Q_array = numpy.ones((z, 3), dtype=self.float)
		s_array = numpy.ones(z, dtype=self.complex)

		# Calculate Q, k_i and k_f for each (z) pixel, to be passed to the GPU
		for idk in range(z):
			thetarock = idk*self.dtheta - self.thetamax/2.0
			self.RotateKiKfQ(thetarock, ki, kf, Q)
			self.SetS()
			Q_array[idk] = self.Q
			s_array[idk] = self.S
			k_i_array[idk] = self.k_i
			k_f_array[idk] = self.k_f

		# Compute the atom displacements
		atomR = self.CalcAtomDisplacement()

		# Compute the intensity for each (x, y, z) pixel on the GPU in one call
		if len(self.atomfunctions) > 0:
			diffsim_gpu_kernels.CalcPixelScatterGPU[(ceil(x/64),y,z),(64,1,1)](
				self.atoms, k_i_array, k_f_array, atomR, Q_array, self.exposure, self.binbits, self.pixelxy,
				self.detsize, self.r0, self.flux, self.R, self.flatshapearray, self.diffarray
			)
		else:
			diffsim_gpu_kernels.CalcPixelScatterNoAtomFuncGPU[(ceil(x/64),y,z),(64,1,1)](
				k_f_array, k_i_array, Q_array, self.Rvec, self.exposure, self.binbits, self.pixelxy, self.detsize,
				self.r0, self.flux, self.R, self.flatshapearray, s_array, self.diffarray
			)
	def CalcRockingCurveMPI(self):
		ki = self.k_i.copy()
		kf = self.k_f.copy()
		Q = self.Q.copy()
		steps = int(self.thetamax/self.dtheta)
		##
		x,y,z = self.detsize[0],self.detsize[1],steps
		n = self.mpi_size
		ni = self.mpi_rank
		blocks = self.Subdivide(n, [x,y,z])
		block = blocks[ni]
		blkx = block[1] - block[0]
		blky = block[3] - block[2]
		blkz = block[5] - block[4]
		if ni == 0:
			self.diffarray = numpy.zeros((self.detsize[0],self.detsize[1],steps), dtype=self.float)
		self.diffarrayblock = numpy.zeros((blkx,blky,blkz), dtype=self.float)

		# Compute the atom displacements
		atomR = self.CalcAtomDisplacement()

		for idk in range(block[4], block[5], 1):
			thetarock = idk*self.dtheta - self.thetamax/2.0
			self.RotateKiKfQ(thetarock, ki, kf, Q)
			self.SetS()

			for idj in range(block[2], block[3], 1):
				for idi in range(block[0], block[1], 1):
					alphakf = self.CalcDetAlpha(idi,idj)
					k_fp = self.CalcKfp(alphakf)
					blki = idi - block[0]
					blkj = idj - block[2]
					blkk = idk - block[4]
					self.diffarrayblock[blki,blkj,blkk] = self.CalcPixelScatter(k_fp, atomR)
		##
		comm = MPI.COMM_WORLD
		if ni > 0:
			data1 = self.diffarrayblock
			data = numpy.ascontiguousarray(data1, dtype=self.float)
			comm.Send(data, dest=0, tag=13)
		##
		if ni == 0:
			self.diffarray[block[0]:block[1],block[2]:block[3],block[4]:block[5]] = self.diffarrayblock[:]
			for i in range(1,n,1):
				block = blocks[i]
				blkx = block[1] - block[0]
				blky = block[3] - block[2]
				blkz = block[5] - block[4]
				data1 = numpy.zeros((blkx,blky,blkz), dtype=self.float)
				data = numpy.ascontiguousarray(data1, dtype=self.float)
				comm.Recv(data, source=i, tag=13)
				self.diffarray[block[0]:block[1],block[2]:block[3],block[4]:block[5]] = data[:]

	def SaveParameters(self):
		if self.mpi_rank == 0:
			params = ""
			params += "Name: %s \n"%self.meta_name
			params += "Beam flux density: %1.5e (1/m2) \n"%(self.flux*(10.0**12))
			params += "Wavelength: %2.6f (nm) \n" %self.waveln
			params += "Energy: %2.6f (keV) \n" %self.energy
			params += "k magnitude: %2.6f (1/nm) \n" %self.k
			params += "HKL: (%d %d %d) \n" %(self.hkl[0],self.hkl[1],self.hkl[2])
			params += "Q vector: %2.6f %2.6f %2.6f (1/nm) \n" %( self.Q[0],self.Q[1],self.Q[2] )
			params += "ki vector: %2.6f %2.6f %2.6f (1/nm) \n" %( self.k_i[0],self.k_i[1],self.k_i[2] )
			params += "kf vector: %2.6f %2.6f %2.6f (1/nm) \n" %( self.k_f[0],self.k_f[1],self.k_f[2] )
			params += "Bragg angle: %2.6f (Deg) \n" %(self.theta*self.rad2deg)
			params += "2Bragg angle: %2.6f (Deg) \n" %(2.0*self.theta*self.rad2deg)
			params += "Number of unit cells: %d \n" %(numpy.sum(self.shapearray)*self.binbits)
			params += "Object supercell size: %d x %d x %d (unit cells) \n" %(self.bin[0],self.bin[1],self.bin[2])
			params += "Photon count at centre pixel: %1.4e (1/m2) \n" %self.Isc
			params += "Exposure time: %.2f (s) \n" %self.exposure
			params += "Oversampling Ratio: %2.2f \n" %self.beta
			params += "Detector size x: %d \n" %(self.detsize[0])
			params += "Detector size y: %d \n" %(self.detsize[1])
			params += "Detector pixel size x: %.0f (microns) \n" %(self.pixelxy[0])
			params += "Detector pixel size y: %.0f (microns) \n" %(self.pixelxy[1])
			params += "Detector distance: %1.4f (m) \n" %(self.R)
			params += "Rocking curve increment: %1.6f (Deg) \n" %(self.dtheta*self.rad2deg)
			params += "Rocking curve width max: %1.6f (Deg) \n" %(self.thetamax*self.rad2deg)
			params += "Rocking curve steps: %d \n" %(int(self.thetamax/self.dtheta))
			params += "Resolution: %.2f (nm) \n" %self.resolution
			params += "CPU processes: %d \n" %self.mpi_size
			if len(self.atomfunctions) > 0:
				params += "========== Atom Functions ========== \n"
			for f in self.atomfunctions:
				params += "Atom index: %d \n" %f[0]
				params += "%s \n" %(getsource(f[1]))
				params += "\n"
			f = open('DiffSim_'+self.meta_name+'_'+self.datestr+'.txt', "w")
			f.write(params)
			f.close()
	def SaveObject(self, name=None):
		if name is None:
			name = "Object_"+self.meta_name+("_[%d%d%d]_"%(self.hkl[0],self.hkl[1],self.hkl[2]))+self.datestr+'.npy'
		numpy.save(name, self.object)
	def SaveObjectCoords(self, name=None):
		if name is None:
			name = "ObjectCoords_"+self.meta_name+("_[%d%d%d]_"%(self.hkl[0],self.hkl[1],self.hkl[2]))+self.datestr+'.npy'
		numpy.save(name, self.coordarray)
	def SaveDiffraction(self, name=None):
		if self.mpi_rank == 0:
			if name is None:
				name = "Diff_"+self.meta_name+("_[%d%d%d]_"%(self.hkl[0],self.hkl[1],self.hkl[2]))+self.datestr+'.npy'
			numpy.save(name, self.diffarray)



if __name__=="__main__":
	import numpy
	from math import sqrt
	#from diffsim import DiffSim
	d = DiffSim()
	d.meta_name = "ZnO"
	# Wurtzite structure
	# unit cell in nm
	crystal_a = 0.325 # ZnO
	crystal_b = 0.325 # ZnO
	crystal_c = 0.520 # ZnO
	d.SetLatticeVector(0,[0.5*crystal_a,-0.5*sqrt(3)*crystal_a,0.0])
	d.SetLatticeVector(1,[0.5*crystal_b,0.5*sqrt(3)*crystal_b,0.0])
	d.SetLatticeVector(2,[0.0,0.0,crystal_c])
	# Reflection
	d.SetMillerIndices([0,0,2])
	# Atoms in unit cell and scattering factors
	d.AddAtom(24.649, 0,0,0) # Zn
	d.AddAtom(24.649, 1/3,1/3,1/2) # Zn
	d.AddAtom(5.793, 0,0,3/8) # O
	d.AddAtom(5.793, 1/3,1/3,7/8) # O
	# Atom functions for displacement from
	# equilibrium.
	# Create functions here that will operate
	# on specific atomis in each unit cell.
	# The input is always a numpy array of
	# all indices (N by 3) of each unit cell.
	# Use these indicies to parameterise
	# displacements in Wyckoff units.
	# Use this feature to create a ferroelectric
	# displacement, for example.
	def Atom1Function(ijks):
		return 0.025*ijks
	# Add function for specific atom number:
	d.AddAtomFunction(1, Atom1Function)
	# Supercell
	# Set supercell size in object before setting
	# shape array
	d.SetSuperCell(10,10,10)
	# Shape function array
	# # load if you have it already
	#d.SetShapeArray("shape.npy")
	# hexagonal prism
	X=350
	Y=350
	Z=550
	dX = 30
	X2=X/2 - dX
	Y2=Y/2 - dX
	Z2=Z/2 - dX
	s3 = sqrt(3)
	d.SetShapeArrayFromNormals([X,Y,Z],\
		[[dX,Y2,Z2,dX-1,Y2,Z2],\
		[X-dX,Y2,Z2,X,Y2,Z2],\
		[X2/2+X/2,s3*Y2/2+Y/2,Z2,X2/1.9+X/2,s3*Y2/1.9+Y/2,Z2],\
		[X2/2+X/2,-s3*Y2/2+Y/2,Z2,X2/1.9+X/2,-s3*Y2/1.9+Y/2,Z2],\
		[-X2/2+X/2,s3*Y2/2+Y/2,Z2,-X2/1.9+X/2,s3*Y2/1.9+Y/2,Z2],\
		[-X2/2+X/2,-s3*Y2/2+Y/2,Z2,-X2/1.9+X/2,-s3*Y2/1.9+Y/2,Z2],\
		[X2,Y2,dX,X2,Y2,dX-1],\
		[X2,Y2,Z-dX,X2,Y2,Z],\
		])
	# 
	d.SaveShapeArray("shape.npy")
	# detector
	d.SetDetPixelSize(50,50)
	d.SetDetSize(128,128)
	# Wavelength / energy (keV)
	d.SetEnergy(9.0)
	# Flux
	d.SetFlux(10.0**20)
	# Exposure time (s)
	# This will directly influence
	# the rocking curve width
	d.SetExposureTime(100)
	# Oversampling ratio
	d.SetBeta(1.2)
	# prepare
	d.Prepare()
	# override rocking curve
	# if needed:
	# # increment (rad)
	# d.dtheta = 0.0001 #(rad)
	# # width (rad)
	# d.SetThetaMax(max= 0.04) #(rad)
	# Save parameters
	d.SaveParameters()
	# Object
	d.CalcObject()
	d.CalcObjectCoords()
	# Rocking curve
	# d.CalcRockingCurveThreads()
	# d.CalcRockingCurveMPI()
	d.CalcRockingCurveGPU()
	# Save result
	d.SaveObject()
	d.SaveObjectCoords()
	d.SaveDiffraction()




