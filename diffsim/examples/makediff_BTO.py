# #######################################
# filename: makediff_BTO.py
# 
# Example script showing how to use diffsim_mpi.py
# 
# #######################################

import numpy
from math import sqrt
from diffsim import DiffSim
from scipy import signal

d = DiffSim()
d.meta_name = "BTO_SquareWave"
# Tetragonal structure 
# unit cell in nm
crystal_a = 0.39925
crystal_b = 0.39925
crystal_c = 0.40373
# distort along x axis
d.SetLatticeVector(0,[crystal_c,0.0,0.0])
d.SetLatticeVector(1,[0.0,crystal_b,0.0])
d.SetLatticeVector(2,[0.0,0.0,crystal_a])
# Reflection
d.SetMillerIndices([1,0,1])
# Atoms in unit cell and scattering factors
d.AddAtom(55.7107, 0,0,0) # Ba
d.AddAtom(22.3470, 0.5,0.5,0.5) # Ti
d.AddAtom(8.04313, 0.5,0.5,0.0) # O
d.AddAtom(8.04313, 0.5,0.0,0.5) # O
d.AddAtom(8.04313, 0,0.5,0.5) # O

def Atom2Function(ijks): #atom function for Ti i.e atom number 2
	X= 400
	Y= 400
	Z= 400
	u = 0.01 * signal.square((2.0* numpy.pi/Z) * ijks)
	u[:,0] = u[:,2]
	u[:,1] = 0
	u[:,2] = 0
	return u

def Atom3Function(ijks):
	X=400
	Y=400
	Z=400
	u=-0.022 * signal.square((2.0 * numpy.pi/Z) * ijks)
	u[:,0] = u[:,2]
	u[:,1] = 0
	u[:,2] = 0
	return u

def Atom45Function(ijks):
	X=400
	Y=400
	Z=400
	u = -0.014 * signal.square((2.0*numpy.pi/Z)* ijks)
	u[:,0] = u[:,2]
	u[:,1] = 0
	u[:,2] = 0
	return u

d.AddAtomFunction(2, Atom2Function)
d.AddAtomFunction(3, Atom3Function)
d.AddAtomFunction(4, Atom45Function)
d.AddAtomFunction(5, Atom45Function)
#
d.SetSuperCell(5,5,10)
#d.SetShapeArray("shape.npy")
d.SetShapeArrayFromNormals([400,400,400],\
	[[5,200,200,4,200,200],\
	[395,200,200,396,200,200],\
	[200,5,200,200,4,200],\
	[200,395,200,200,396,200],\
	[200,200,5,200,200,4],\
	[200,200,395,200,200,396],\
	])
#d.SaveShapeArray("shape.npy")
d.SetDetPixelSize(50,50)
d.SetDetSize(256,256)
d.SetEnergy(9.0)
d.SetFlux(10**20)
d.SetExposureTime(100)
d.SetBeta(1.5)
d.Prepare()
d.SaveParameters()
d.CalcObject()
d.CalcObjectCoords()
d.SaveObject()
d.SaveObjectCoords()
d.CalcRockingCurveGPU()
#d.CalcRockingCurveThreads()
#d.CalcRockingCurveMPI()
d.SaveDiffraction()

