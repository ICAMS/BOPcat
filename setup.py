#!/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
import sys
import os

main_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(main_dir, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

sys.path.insert(0, os.path.join(main_dir, "bopcat"))

packages = ['bopcat','bopcat.structmap']

package_dir = {'bopcat':'bopcat'}

package_data = {'bopcat':['logo.gif','structmap/cluster.f90','structmap/cluster.so'
                         ,'structmap/atoms.bx'
                         ,'structmap/bonds.bx','structmap/models.bx'
                         ]
               }
setup(
    name='bopcat',
    version='1.0',
    description='BOPcat is a python package for automatic parametrization of bond-order potentials',
    long_description=long_description, 
    url='https://github.com/ICAMS/BOPcat',
    author='Alvin Noe Ladines',
    author_email='ladinesalvinnoe@gmail.com',
    license='License :: GNU-GPL',
    classifiers=['Development Status :: 5 - Production/Stable',
'Topic :: Scientific/Engineering :: Physics ',
'License :: OSI Approved :: GNU General Public License (GPL)',
'Programming Language :: Python :: 3.7',],
    keywords=['bond-order potentials', 'parameterization', 'modelling'],
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    include_package_data=True,
    install_requires=['numpy','scipy','matplotlib','spglib','ase'],
)
