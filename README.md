# BOPcat: Bond-Order Potential Construction and Testing

BOPcat is a python package for the automated parametrization of tight-binding models and analytic bond-order potentials. 
It drives the BOPfox calculator to calculate energies, forces and stresses which are used to optimize the parameters.
Details of the methodology and application are given in the original publication (A. Ladines, T. Hammerschmidt, R. Drautz, ["BOPcat software package for the construction and testing of tight-binding models and analytic bond-order potentials"](https://www.sciencedirect.com/science/article/abs/pii/S0927025619307542), Comp. Mat. Sci. 173 (2020) 109455) ([preprint](https://arxiv.org/abs/1907.12254)).

# Requirements:

[numpy](http://www.numpy.org/), [scipy](http://www.scipy.org/), [matplotlib](http://www.matplotlib.org/), [spglib](https://atztogo.github.io/spglib/), [ase](https://wiki.fysik.dtu.dk/ase/)

# Installation:

1. Get a copy from git repository

   $ git clone https://github.com/ICAMS/BOPcat.git

2. Run installation script

   $ python setup.py install

Alternatively, specify path to BOPcat source files, 
i.e. append the following to your ``.bashrc``:
    
    ``export PYTHONPATH=<path to bopcat>:/bopcat:$PYTHONPATH``
    ``export PYTHONPATH=<path to bopcat>:$PYTHONPATH``

3. Run test installation script

   $ python test_install.py

4. Run examples in ``examples/``

# Usage:

1. Write a main script to specify parameterization process. Use the tools of the package. See manual at ``docs/build/``

2. Write an input controls script to specify variables.

3. Run the script.

   $ python main.py input.py

# License

BOPcat is available under GPL v3.

# Citation

If you find BOPcat useful please consider citing:

A. Ladines, T. Hammerschmidt, R. Drautz, 
["BOPcat software package for the construction and testing of tight-binding models and analytic bond-order potentials"](https://www.sciencedirect.com/science/article/abs/pii/S0927025619307542) Comp. Mat. Sci. 173 (2020) 109455

