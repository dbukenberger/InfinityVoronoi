<img src='infinityVoronoi.svg' align="right" width="256" height="256">

# InfinityVoronoi

### Constructing L∞ Voronoi Diagrams in 2D and 3D
[![PDF](https://img.shields.io/badge/PDF-green)](https://ls7-gv.cs.tu-dortmund.de/downloads/publications/2022/sgp22.pdf)
[![CGF Paper](https://img.shields.io/badge/DOI-10.1111%2Fcgf%2E14609-blue)](https://doi.org/10.1111/cgf.14609)

Our implementation generates Voronoi Diagrams for a given set of sites with parameterized orientation and anisotropy fields.
This is currently only the 2D version of the algorithm, the 3D code will follow after some more refactoring.

## Dependencies
Most used libraries are already included in the Python Standard Library (developed and tested with Python 3.8.10).

The only **actually required** library is [NumPy](https://github.com/numpy/numpy).

**Optionally**, results can be plotted with [matplotlib](https://github.com/matplotlib/matplotlib/) and progress bars shown in the shell are realized with [tqdm](https://github.com/tqdm/tqdm).

## Run Examples
In the main directory, just run `python runExamples2D.py`.
This file contains examples for two L∞ Voronoi Diagrams and a Lloyd relaxation with different orientation and anisotropy fields each.
Results are stored as `.obj` files, plots in any format supported by the matplotlib.

## Citation
You can cite the paper with:
```
@Article{bukenberger2022constructing,
	author    = {Bukenberger, Dennis R. and Buchin, Kevin and Botsch, Mario},
	journal   = {Computer Graphics Forum},
	title     = {{Constructing $L_\infty$ Voronoi Diagrams in 2D and 3D}},
	year      = {2022},
	issn      = {1467-8659},
	pages     = {135--147},
	volume    = {41.5},
	doi       = {10.1111/cgf.14609},
	publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
}
```