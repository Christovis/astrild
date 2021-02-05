GalaxyHaloConnection
==============================
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

astrild is a specialized python package for cosmological simulations based on [RAMSES](https://bitbucket.org/rteyssie/ramses/wiki/Home). This contains extensions such as [ECOSMOG](https://arxiv.org/abs/1110.1379), to study the effects of novel theories that modify GR on large scales and late times. In addition it is possible to explore arrays of simulations in parallel as it is needed for ray-tracing simulations based on [Ray-Ramses](https://arxiv.org/pdf/1601.02012.pdf).

The functionality of the package includes:

* analysis of matter, fifth-force, and halo power- and bispectra, supported by [DTFE](https://www.astro.rug.nl/~voronoi/DTFE/dtfe.html) and [halotools](https://github.com/astropy/halotools).
* halo statistics such as halo mass function, correlation functions, mean pairwise velocities, redshift space clustering, and concentration mass relation
* weak lensing statistics for several types of voids
* ISW and RS analysis

