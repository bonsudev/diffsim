[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Python Versions](https://img.shields.io/badge/Python-3brightgreen) ![PyPI](https://img.shields.io/pypi/v/diffsim) ![PyPI - Wheel](https://img.shields.io/pypi/wheel/diffsim)

# DiffSim

## Introduction

DiffSim provides an atomistic approach to efficiently simulate Bragg coherent x-ray diffraction imaging (BCDI) diffraction patterns by factorising and eliminating certain redundancies in the conventional approach. The method used is able to reduce the computation time by several orders of magnitude without compromising the recovered phase information and therefore enables feasible atomistic simulations on nanoscale crystals with arbitrary lattice distortions.

Please cite the following article when using Diffsim in published work:

[Ahmed Mokhtar, David Serban and Marcus Newton, Simulation of Bragg coherent diffraction imaging, J. Phys. Commun. Volume 6, 055003 (2022)](https://doi.org/10.1088/2399-6528/ac6ab0)


## Installation

DiffSim requires:
* Python >= 3.7
* Numba >= 0.58.1
* NumPy >= 1.25.0
* scipy
* mpi4py


Installation via pip:
```
$ pip install diffsim
```


## Reporting Bugs

Please send any bugs, problems, and proposals to: Bonsu.Devel@gmail.com
or visit: http://github.com/bonsudev/diffsim


## Library Usage and Examples

Example scripts are available in the examples folder.


## Version History

#### Version 1.2 🗓 ️(02/11/2023)

	🔧 Minor bug fixes.


#### Version 1.1 🗓 ️(29/10/2023)

	✨ Refactored code for speed improvements.
	✨ Added attenuation and refraction corrections.
	🔧 Minor bug fixes.


#### Version 1.0 🗓 ️(14/09/2022)

	✨ First major release. 


## Licence

GNU GPLv3



