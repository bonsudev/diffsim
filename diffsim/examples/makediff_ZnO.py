# #######################################
# filename: makediff_ZnO.py
# 
# Example script showing how to use diffsim.py
# 
# #######################################

import numpy
from math import sqrt
from diffsim import DiffSim

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
d.SetMillerIndices([1,0,1])
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
def Atom1Function(ijks):
	X=350
	Y=350
	Z=450
	u = 0.025*numpy.sin((2.0*numpy.pi/Z)*ijks)
	u[:,0] = u[:,2]
	u[:,1] = u[:,2]
	u[:,2] = 0
	return u
# Add function for specific atom number:
d.AddAtomFunction(1, Atom1Function)
#d.AddAtomFunction(2, Atom1Function)
#d.AddAtomFunction(3, Atom1Function)
#d.AddAtomFunction(4, Atom1Function)
# Supercell zie
d.SetSuperCell(16,16,16)
# Shape function array
# # load if you have it already
#d.SetShapeArray("shape.npy") 
# hexagonal prism
X=350
Y=350
Z=450
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
d.SaveShapeArray("shape161616.npy")
# detector
d.SetDetPixelSize(50,50)
d.SetDetSize(128,128)
# Wavelength / energy (keV)
d.SetEnergy(9.0)
# Flux
d.SetFlux(10**20)
# Exposure time (s)
# This will directly influence
# the rocking curve width
d.SetExposureTime(100)
# Oversampling ratio
d.SetBeta(1.5)
# prepare
d.Prepare()
# override rocking curve
# if needed:
# # increment (rad)
# d.dtheta = 0.0001 #(rad)
# # width (rad)
d.SetThetaMax(max=0.0967) #(rad)
# Save parameters
d.SaveParameters()
# Rocking curve
d.CalcRockingCurveGPU()
#d.CalcRockingCurveThreads()
#d.CalcRockingCurveMPI()
# Save result
d.SaveDiffraction()




