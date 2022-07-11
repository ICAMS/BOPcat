#!/usr/bin/env python

# ASE interface to BOPfox based on system call
# Based on Rye Terell's TSASE interface

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import subprocess
import os
import tempfile
import shutil
import numpy as np
from ase.calculators.general import Calculator
import ase.io.bopfox as bopio
from time import gmtime, strftime
import gzip
import warnings


def bopfox_keys():
    """
    BOPcat tries to update from modules.fp. It it fails, e.g. when it does
    not find the path to bopfox/src the following will be used.
    """
    skeys = ['version', 'forces', 'task', 'model', 'screeningversion'
        , 'jijversion', 'bandwidth', 'bopkernel', 'terminator'
        , 'efermimixer', 'scfmixer', 'magconfig', 'scftraj'
        , 'cutoffversion', 'ecutoffversion', 'strucfile', 'modelfile'
        , 'tbkpointmesh', 'tbintegration', 'tbsolver', 'tbkpointfile'
        , 'rxkernel', 'rxtype', 'mdkernel', 'mdthermostat', 'mdbarostat'
        , 'polytype', 'polyoutputlevel', 'struktype']
    bkeys = ['screening', 'orthogonal', 'globalbandwidth', 'scfreusehii'
        , 'scfrigidhiishift', 'scffixmagcom', 'scffixmagdiff', 'scfdamping'
        , 'scfsaveonsite', 'scfrestart', 'printdos', 'dosvsn', 'partialdos'
        , 'fastbop', 'verbose', 'verbosetime', 'verboseatom', 'printsc'
        , 'printbonds', 'neighbourhistogram', 'printbetaparameters'
        , 'printoverparameters', 'printrepparameters', 'printham', 'printxi'
        , 'printxi_normed', 'printmu', 'printmu2', 'printmu_averaged'
        , 'printmu_normed', 'printanbn', 'printanhxbnhx', 'printa_infb_inf'
        , 'printsigma', 'printbo', 'printbotilde', 'printefermi', 'printefermifinal'
        , 'printonsitelevels', 'printcharges', 'printmagmom', 'printenergies'
        , 'printforces', 'printstress', 'printtorques', 'printcfg', 'rxconstraint'
        , 'cgfinish', 'bgdiagco', 'mdfixcom', 'mdrestart', 'mdvrescale'
        , 'mddohistogram', 'cvanalyze', 'rdfanalyze', 'msdanalyze', 'polyanalyze'
        , 'strukanalyze', 'cvianalyze']
    ikeys = ['moments', 'numfdisp', 'nexpmoments', 'efermisteps', 'scfsteps'
        , 'scffirenmin', 'dosgrid', 'tbnbandstotal', 'tbnsmear', 'ioutput'
        , 'openmp_nthreads', 'rxmaxsteps', 'rxsaveframes', 'cgiprint1', 'cgiprint2'
        , 'cgmethod', 'cgiflag', 'cgirest', 'bgiprint1', 'bgiprint2', 'bgiflag'
        , 'mdsteps', 'mdequilibrationsteps', 'mdhistosize', 'mdchainlength1'
        , 'mdchainlength2', 'mdsaveframes', 'rdfupdate', 'rdfgridpoints'
        , 'msdinterval', 'polycorrlevel', 'strukcorrlevel', 'polygap', 'cviupdate']
    fkeys = ['numfinc', 'rskin', 'rthickskin', 'global_a_inf', 'global_b_inf'
        , 'maxpathradius', 'alphaweight', 'efermitol', 'scftol', 'scflinearmixpara'
        , 'scfbroydenmixpara', 'scfonsitemddtsq', 'scfonsitemdmass'
        , 'scfonsitemddamping', 'scffirefinc', 'scffirefdec', 'scffirefalpha'
        , 'scffirealphastart', 'scffiredtinitial', 'scffiredtmax', 'screenp1'
        , 'screenp2', 'scfdampinglimit', 'ecut', 'decut', 'tbdsmear', 'rxsteplimit'
        , 'rxeconv', 'rxfconv', 'dndamping', 'dnrecmass', 'dnrecmasscell', 'cgeps'
        , 'bgleps', 'bgltol', 'fidt', 'mdtimestep', 'mdinitialt', 'mdtemperature'
        , 'mdvolume', 'mdcoupling', 'mdpressure', 'mdpistonmass', 'rdflimit'
        , 'polyweight']
    return skeys, bkeys, ikeys, fkeys


def which(program):
    from distutils.spawn import find_executable
    return find_executable(program)


def find_bopfox_source():
    envs = os.environ
    bopfox = None
    if 'BOPFOX' in envs:
        envs = envs['BOPFOX'].split(':')
        for env in envs:
            if 'bopfox' in env:
                bopfox = env
                break
    if bopfox is None:
        raise RuntimeError('Specify path to BOPFOX.')
    return bopfox


def update_bopfox_keys():
    """
    Extracts all keywords in bopfox from modules.fp
    """
    import pickle
    temp = which("bopfox").strip()
    temp = temp.split('/')
    path2bopfox = ''
    for i in range(1, len(temp) - 1):
        path2bopfox += ('/' + temp[i])

    if not os.path.isdir(path2bopfox):
        raise ValueError("Provided path; %s does not exist." % path2bopfox)
    bopfox_keys_s = []
    bopfox_keys_b = []
    bopfox_keys_i = []
    bopfox_keys_f = []
    l = []
    for fn in ['modules.fp', 'relax_init.fp', 'md_init.fp', 'analyseatom_init.fp']:
        if os.path.isfile('%s/%s' % (path2bopfox, fn)):
            f = open('%s/%s' % (path2bopfox, fn))
        else:
            try:
                path2bopfox = find_bopfox_source().rstrip('/bopfox')
                f = open('%s/%s' % (path2bopfox, fn))
            except:
                print(("Cannot find %s in %s" % (fn, path2bopfox)))
                print("Will use known keys. May result in bopfox error.")
                return bopfox_keys()
                # raise ValueError("Cannot upload keys")
        l += f.readlines()
        f.close()
    for i in range(len(l)):
        if 'type(input_c)::' in l[i]:
            try:
                name = l[i].split("'")[1].strip()
            except:
                name = l[i].split("=")[0].split('::')[1].strip().lower()
            bopfox_keys_s.append(name)
        if 'type(input_l)::' in l[i]:
            try:
                name = l[i].split("'")[1].strip()
            except:
                name = l[i].split("=")[0].split('::')[1].strip().lower()
            bopfox_keys_b.append(name)
        if 'type(input_i)::' in l[i]:
            try:
                name = l[i].split("'")[1].strip()
            except:
                name = l[i].split("=")[0].split('::')[1].strip().lower()
            bopfox_keys_i.append(name)
        if 'type(input_r)::' in l[i]:
            try:
                name = l[i].split("'")[1].strip()
            except:
                name = l[i].split("=")[0].split('::')[1].strip().lower()
            bopfox_keys_f.append(name)
    with open('bopfox_keys.bx', 'wb') as f:
        pickle.dump((bopfox_keys_s, bopfox_keys_b, bopfox_keys_i
                     , bopfox_keys_f), f)
    return bopfox_keys_s, bopfox_keys_b, bopfox_keys_i, bopfox_keys_f


try:
    import pickle

    with open('bopfox_keys.bx', 'rb') as f:
        temp = pickle.load(f)
    str_keys = temp[0]
    boo_keys = temp[1]
    int_keys = temp[2]
    flo_keys = temp[3]
    f.close()
except:
    str_keys, boo_keys, int_keys, flo_keys = update_bopfox_keys()

lis_keys = ['tbkpointfolding'
    , 'mpi_pbc'
    , 'mpi_grid'
            ]


class BOPfox(Calculator):
    """
    Defines an ASE interface to BOPfox.
    
    In order to define a calculator, the user should provide a modelsbx,
    or atomsbx and bondsbx. These can either be objects (see :mod:`modelsbx`,
    :mod:`atomsbx` and :mod:`bondsbx`), filenames or paths. In addition, bopfox
    input controls (infox parameters) can be optionally provided, otherwise the
    calculator is expecting an infox.bx file on current working directory. 
    The calculator has other attributes on top
    of those of ASE such as :func:`get_moments`

    .. todo:: extend ASE Atoms object to call BOPfox native functions.

    :Parameters:
    
        - *atomsbx*: str or instance of atomsbx

            used to create atoms.bx 

            ``None``: will use modelsbx

        - *bondssbx*: str or instance of bondsbx

            used to create bonds.bx 

            ``None``: will use modelsbx

        - *modelsbx*: str or instance of modelsbx

            used to create models.bx 

            ``None``: will use modelsbx

        - *bopfox*: str 

            bopfox executable

        - *savelog*: bool

            logical flag to save log file for further anaylis 

        - *mem_limit*: float

            sets memory limit for bopfox calculation

        - *ignore_errors*: bool
 
            logical flag to skip bopfox errors

        - *root_tmp_folder*: str
       
            directory where a tmp folder will be generated
 
        - *debug*: bool

            logical flag for debugging purposes, will not delete temporary
            folders

        - *kpoints*: list

            coordinates of kpoints to be used for tight-binding band structure
            calculation

        - *kwargs*: 
 
            optional keyword arguments to be written in infox, if not provided
            will use infox.bx  
            
    """

    def __init__(self, restart=None, track_output=False,
                 atoms=None,
                 atomsbx=None,
                 bondsbx=None,
                 infoxbx='infox.bx',
                 modelsbx='models.bx',
                 bopfox='bopfox',
                 savelog=False,
                 mem_limit=None,
                 ignore_errors=True,
                 debug=False,
                 kpoints=None,
                 root_tmp_folder='/tmp',
                 magconfig=None,
                 **kwargs):
        self.name = 'BOPfox'
        self.restart = restart
        self.track_output = track_output
        self.positions = None
        self.u = None  # potential energy
        self.f = None  # forces
        self.s = None  # stress
        self.stresses = None
        self.magnetic_moments = None
        self.moments = None
        self.an = None
        self.anhx = None
        self.a_inf = None
        self.bn = None
        self.bnhx = None
        self.b_inf = None
        self.charges = None
        self.contributions_energy = None
        self.contributions_forces = None
        self.contributions_stresses = None
        self.eigenvalues = None
        self.orbital_character = None
        self.fermi_energy = None
        self.dos = None
        self.dos_dict = None
        self.bonds = None
        self.dists = None
        self.neighbors = None
        self.atoms = atoms
        self.magconfig = magconfig
        self.atomsbx = atomsbx
        self.bondsbx = bondsbx
        self.modelsbx = modelsbx
        self.infoxbx = infoxbx
        self.bopfox = bopfox
        self.savelog = savelog
        self.mem_limit = mem_limit
        self.ignore_errors = ignore_errors
        self.root_tmp_folder = root_tmp_folder
        self.debug = debug
        self.bopfox_error = False
        self.kpoints = kpoints
        self.str_params = {}
        self.boo_params = {}
        self.int_params = {}
        self.flo_params = {}
        self.lis_params = {}
        for key in str_keys:
            self.str_params[key] = None
        for key in boo_keys:
            self.boo_params[key] = None
        for key in int_keys:
            self.int_params[key] = None
        for key in flo_keys:
            self.flo_params[key] = None
        for key in lis_keys:
            self.lis_params[key] = None
        self.str_params['strucfile'] = 'struc.bx'
        self.str_params['task'] = 'energy'
        self.relaxed_atoms = None
        self.onsite = None
        self.set(**kwargs)

    def set(self, **kwargs):
        self.infox_from_file = True
        for key in kwargs:
            self.infox_from_file = False
            if key.lower() in self.str_params:
                self.str_params[key.lower()] = kwargs[key]
            elif key.lower() in self.boo_params:
                self.boo_params[key.lower()] = kwargs[key]
            elif key.lower() in self.int_params:
                self.int_params[key.lower()] = kwargs[key]
            elif key.lower() in self.flo_params:
                self.flo_params[key.lower()] = kwargs[key]
            elif key.lower() in self.lis_params:
                self.lis_params[key.lower()] = kwargs[key]
            elif key.lower() == 'kpoints':
                self.kpoints = kwargs[key]
            elif key == "bopfox":
                self.bopfox = kwargs[key]
            elif key == "debug":
                self.debug = kwargs[key]
            elif key == "root_tmp_folder":
                self.root_tmp_folder = kwargs[key]
            elif key.lower() == 'ini_magmoms':
                pass
            else:
                raise TypeError('Unrecognized parameter: ' + key)
        if self.lis_params['tbkpointfolding'] is None:
            self.lis_params['tbkpointfolding'] = [10, 10, 10]
        self.set_keys_from_modelsbx()

    def set_keys_from_modelsbx(self):
        if self.modelsbx is not None:
            if not isinstance(self.modelsbx, str):
                infox_parameters = self.modelsbx.infox_parameters
                for key, val in list(infox_parameters.items()):
                    if isinstance(val, bool):
                        self.boo_params[key] = val
                    elif isinstance(val, str):
                        self.str_params[key] = val
                    elif isinstance(val, int):
                        self.int_params[key] = val
                    elif isinstance(val, list):
                        self.lis_params[key] = val

    def write_infox(self, **kwargs):
        """ 
        Writes infox.bx from input parameters, atomsbx and bondsbx
        compatible.
        """
        if self.get_spin_polarized():
            self.str_params['magconfig'] = 'cm'
        f = open('infox.bx', 'w')
        f.write('/infox.bx written in ASE\n')
        f.write('/------------------------------------------------------\n')
        for key, val in list(self.str_params.items()):
            if val is not None:
                f.write('{0:30} = {1}\n'.format(key, val))
        f.write('/------------------------------------------------------\n')
        for key, val in list(self.boo_params.items()):
            if val is not None:
                if val:
                    f.write('{0:30} = {1}\n'.format(key, 'T'))
                else:
                    f.write('{0:30} = {1}\n'.format(key, 'F'))
        f.write('/------------------------------------------------------\n')
        for key, val in list(self.int_params.items()):
            if val is not None:
                f.write('{0:30} = {1}\n'.format(key, val))
        f.write('/------------------------------------------------------\n')
        for key, val in list(self.flo_params.items()):
            if val is not None:
                f.write('{0:30} = {1:10.20f}\n'.format(key, val))
        for key, val in list(self.lis_params.items()):
            if val is not None:
                f.write('{0:30} ='.format(key))
                for i in range(len(val)):
                    f.write('{0:4} '.format(val[i]))
                f.write('\n')
        f.close()

    def read_modelsbx(self, key, vtype=str):
        warnings.warn('Use inspect_modelsbx', DeprecationWarning)
        return self.inspect_modelsbx(key, vtype)

    def inspect_modelsbx(self, key, vtype=str):
        """
        Checks the value of key in modelsbx.        
        """
        val = None
        l = open('models.bx').readlines()
        comments = '/!#$%*@'
        for i in range(len(l)):
            s = l[i].split()
            if len(s) < 2 or s[0][0] in comments:
                continue
            if s[0].lower() == key.lower():
                val = vtype(s[-1])
                break
        return val

    def write_infox_new(self, **kwargs):
        """ 
        Writes infox.bx from input parameters, models.bx compatible.
        """
        # infox_keys = ['strucfile','modelfile','model','task','tbkpointfile'
        #             ,'tbkpointmesh','scfsteps','tbkpointfolding','magconfig'
        #             ]
        not_infox_keys = ['version', 'eamversion', 'atomsversion', 'bondsversion'
            , 'repversion', 'screening', 'screeningversion', 'jijversion'
            , 'moments', 'global_a_inf', 'global_b_inf', 'bandwidth'
            , 'globalbandwidth', 'terminator', 'nexpmoments', 'bopkernel'
            , 'orthogonal']
        if 'model' in self.str_params:
            if self.str_params['model'] is None:
                if isinstance(self.modelsbx, str):
                    self.str_params['model'] = self.inspect_modelsbx('model'
                                                                     , str)
                else:
                    self.str_params['model'] = self.modelsbx.model
        # for key in self.boo_params.keys():
        #    if 'print' in key:
        #        infox_keys.append(key)
        if self.get_spin_polarized():
            self.str_params['magconfig'] = 'cm'
        f = open('infox.bx', 'w')
        f.write('/infox.bx written in ASE\n')
        f.write('/------------------------------------------------------\n')
        for key, val in list(self.str_params.items()):
            if key.lower() not in not_infox_keys and val is not None:
                f.write('{0:30} = {1}\n'.format(key, val))
        f.write('/------------------------------------------------------\n')
        for key, val in list(self.boo_params.items()):
            if key.lower() not in not_infox_keys and val is not None:
                if val:
                    f.write('{0:30} = {1}\n'.format(key, 'T'))
                else:
                    f.write('{0:30} = {1}\n'.format(key, 'F'))
        f.write('/------------------------------------------------------\n')
        for key, val in list(self.int_params.items()):
            if key.lower() not in not_infox_keys and val is not None:
                f.write('{0:30} = {1}\n'.format(key, val))
        f.write('/------------------------------------------------------\n')
        for key, val in list(self.flo_params.items()):
            if key.lower() not in not_infox_keys and val is not None:
                f.write('{0:30} = {1:10.20f}\n'.format(key, val))
        for key, val in list(self.lis_params.items()):
            if key.lower() not in not_infox_keys and val is not None:
                f.write('{0:30} ='.format(key))
                for i in range(len(val)):
                    f.write('{0:4} '.format(val[i]))
                f.write('\n')
        f.close()

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()
        self.u = None  # potential energy
        self.f = None  # forces
        self.s = None  # stress
        self.stresses = None
        self.magnetic_moments = None
        self.moments = None
        self.charges = None
        self.contributions_energy = None
        self.contributions_forces = None
        self.contributions_stresses = None
        self.eigenvalues = None
        self.orbital_character = None
        self.fermi_energy = None
        self.dos = None
        self.bonds = None
        self.dists = None

    def set_modelsbx(self, modelsbx):
        self.modelsbx = modelsbx
        self.u = None
        self.f = None
        self.s = None
        self.stresses = None
        self.magnetic_moments = None
        self.moments = None
        self.charges = None
        self.contributions_energy = None
        self.contributions_forces = None
        self.contributions_stresses = None
        self.eigenvalues = None
        self.orbital_character = None
        self.fermi_energy = None
        self.dos = None
        self.bonds = None
        self.dists = None
        self.set_keys_from_modelsbx()

    def copy(self):
        return self

    def get_atoms(self):
        atoms = self.atoms.copy()
        atoms.set_calculator(self)
        return atoms

    def get_name(self):
        """Returns the name of the calculator (string).  """
        return self.name

    def get_mode(self):
        """Returns the  calculation mode (string).  """
        if self.modelsbx is None:
            return self.str_params['version']
        else:
            ver = 'bop'
            if isinstance(self.modelsbx, str):
                self.inspect_modelsbx('version', str)
            else:
                infox = self.modelsbx.infox_parameters
                if 'version' in infox:
                    ver = infox['version'].lower()
            return ver

    def get_version(self):
        warnings.warn('Use get_mode', DeprecationWarning)
        return self.get_mode()

    def get_potential_energy(self, atoms=None):
        if self.calculation_required(atoms, "energy"):
            self.atoms = atoms.copy()
            self.calculate()
        return self.u

    def get_forces(self, atoms=None):
        if self.calculation_required(atoms, "forces"):
            self.atoms = atoms.copy()
            self.calculate()
        return self.f.copy()

    def get_stress(self, atoms):
        if self.calculation_required(atoms, "stress"):
            self.atoms = atoms.copy()
            self.calculate()
        return self.s.copy()

    def get_stresses(self, atoms):
        if self.calculation_required(atoms, "stress"):
            self.atoms = atoms.copy()
            self.calculate()
        return self.stresses.copy()

    def get_spin_polarized(self):
        if not hasattr(self, 'spinpol'):
            self.spinpol = self.atoms.get_initial_magnetic_moments().any()
        return self.spinpol

    def get_magnetic_moment(self, atoms):
        self.update(atoms, "energy")
        return self.magnetic_moment

    def get_magnetic_moments(self, atoms):
        self.update(atoms, "energy")
        return self.magnetic_moments

    def update(self, atoms, quantity):
        if self.calculation_required(atoms, quantity):
            self.calculate()

    def calculation_required(self, atoms, quantities):
        if atoms != self.atoms or self.atoms is None:
            return True
        if self.f is None or self.u is None or self.s is None or atoms is None:
            return True
        return False

    def get_relaxed_atoms(self):
        return self.relaxed_atoms

    def get_contributions(self, key):
        warnings.warn('Use get_contributions_energy', DeprecationWarning)
        return self.get_contributions_energy(key)

    def get_contributions_energy(self, key):
        key = key.lower()
        rep_keys = ['rep%d' % i for i in range(100)]
        ene_keys = ['binding', 'bond', 'prom', 'exchange', 'ionic', 'coulomb'
                       , 'env', 'pair', 'rep'] + rep_keys
        if self.contributions_energy is None:
            self.contributions_energy = {}
            for key in ene_keys:
                contrib = {}
                for i in range(len(self.atoms) + 1):
                    contrib[i] = 9E99
                self.contributions_energy[key] = contrib
        if key in ene_keys:
            return self.contributions_energy[key]
        else:
            raise Exception('Unrecognized key: %s' % key)

    def get_contributions_forces(self, key):
        key = key.lower()
        rep_keys = ['rep%d' % i for i in range(100)]
        f_keys = ['analytic', 'eam', 'coulomb', 'total'] + rep_keys
        if self.contributions_forces is None:
            self.contributions_forces = {}
            for key in f_keys:
                contrib = {}
                for i in range(len(self.atoms) + 1):
                    contrib[i] = np.ones(3) * 1E99
                self.contributions_forces[key] = contrib
        if key.lower() in f_keys:
            return self.contributions_forces[key]
        else:
            raise Exception('Unrecognized key: %s' % key)

    def get_contributions_stresses(self, key):
        key = key.lower()
        rep_keys = ['rep%d' % i for i in range(100)]
        s_keys = ['bond', 'total'] + rep_keys
        if self.contributions_stresses is None:
            self.contributions_stresses = {}
            for key in s_keys:
                contrib = {}
                for i in range(len(self.atoms) + 1):
                    contrib[i] = np.ones(6) * 1E99
                self.contributions_stresses[key] = contrib
        if key.lower() in s_keys:
            return self.contributions_stresses[key]
        else:
            raise Exception('Unrecognized key: %s' % key)

    def get_moments(self, atom_index=0, moment=2):
        """Returns the moments.
        
        If the atom index is 0, then average will be returned
        If the atom index is NOT an integer, it will return all moments 
        
        """
        if self.moments != None:
            if (type(atom_index) == int):
                # If atom_index is an integer, return the moments of this atom
                if type(moment) == int:
                    # If moment is an integer, return just the moment-th moment
                    return self.moments[atom_index][moment - 1]
                else:
                    # If moment was set to anything else, return all moments
                    return self.moments[atom_index]
            else:
                # If atom index was set to anything else return a list of the 
                # moments
                n_atoms = len(self.moments) - 1
                output = [None] * n_atoms
                for i_atom in range(n_atoms):
                    if type(moment) == int:
                        # If moment is an integer, return just the moment-th 
                        # moment
                        output[i_atom] = self.moments[i_atom + 1][moment + 1]
                    else:
                        # If moment was set to anything else, return all 
                        # moments
                        output[i_atom] = self.moments[i_atom + 1]
                return output
        else:
            print('return None')
            return self.moments

    def get_anbn(self, atom_index=0, moment=2):
        if self.an != None and self.bn != None:
            if (type(atom_index) == int):
                # If atom_index is an integer, return anbn of this atom
                if type(moment) == int:
                    # If moment is an integer, return just the moment-th moment
                    return self.an[atom_index][moment - 1], \
                           self.bn[atom_index][moment - 1]
                else:
                    # If moment was set to anything else, return all moments
                    return self.an[atom_index], self.bn[atom_index]
            else:
                # If atom index was set to anything else return a list of the 
                # moments
                n_atoms = len(self.an) - 1
                output_an = [None] * n_atoms
                output_bn = [None] * n_atoms
                for i_atom in range(n_atoms):
                    if type(moment) == int:
                        # If moment is an integer, return just the moment-th 
                        # moment
                        output_an[i_atom] = self.an[i_atom][moment + 1]
                        output_bn[i_atom] = self.bn[i_atom][moment + 1]
                    else:
                        # If moment was set to anything else, return all 
                        # moments
                        output_an[i_atom] = self.an[i_atom]
                        output_bn[i_atom] = self.bn[i_atom]
                return output_an, output_bn
        else:
            print('return None, None')
            return self.an, self.bn

    def get_charges(self, atom_index=0):
        if self.charges != None:
            return self.charges[atom_index]
        else:
            return self.charges

    def get_eigenvalues(self):
        return self.eigenvalues

    def get_orbital_character(self):
        return self.orbital_character

    def get_dos(self):
        return self.dos

    def get_fermi_energy(self):
        return self.fermi_energy

    # def get_bonds(self):
    #     return self.bonds, self.dists

    def get_bonds(self, i_atom='all'):
        if type(i_atom) == int:
            return self.bonds[i_atom], self.dists[i_atom], self.neighbors[i_atom]
        else:
            return self.bonds, self.dists, self.neighbors

    def _do_moments(self):
        pm = ['printmu', 'printmu_averaged']
        out = False
        for i in range(len(pm)):
            if pm[i] in self.boo_params:
                if self.boo_params[pm[i]] == True:
                    out = True
                    break
        return out

    def _do_anbn(self):
        pm = ['printanbn']
        out = False
        for i in range(len(pm)):
            if pm[i] in self.boo_params:
                if self.boo_params[pm[i]] == True:
                    out = True
                    break
        return out

    def _do_anhxbnhx(self):
        pm = ['printanhxbnhx']
        out = False
        for i in range(len(pm)):
            if pm[i] in self.boo_params:
                if self.boo_params[pm[i]] == True:
                    out = True
                    break
        return out

    def _do_a_inf_b_inf(self):
        pm = ['printa_infb_inf']
        out = False
        for i in range(len(pm)):
            if pm[i] in self.boo_params:
                if self.boo_params[pm[i]] == True:
                    out = True
                    break
        return out

    def write_onsites(self):
        sym = self.atoms.get_chemical_symbols()
        onsites = self.onsite
        nsp = 1
        if self.get_spin_polarized():
            nsp = 2
        if onsites is None:
            if 'onsites' in self.atoms.info:
                onsites = self.atoms.info['onsites']
                onsites = np.reshape(onsites, (len(self.atoms), nsp))
            else:
                return
        attype = []
        with open('onsite.dat', 'w') as f:
            for i in range(len(sym)):
                if sym[i] not in attype:
                    attype.append(sym[i])
                at = attype.index(sym[i]) + 1
                f.write('%10d  %2d %s ' % (i + 1, at, sym[i]))
                for n in range(nsp):
                    f.write('%20.10f ' % (onsites[i][n]))
                f.write('\n')

    def calculate(self, *args):
        """
        Sets up a temporary foler for perfoming a BOPfox calculation. Then 
        writes the corresponding files and performs a calculation. 
        Cleans up afterwards.       
        """
        curdir = os.getcwd()
        bfdir = tempfile.mkdtemp(dir=self.root_tmp_folder)
        os.chdir(bfdir)
        if self.str_params['tbkpointmesh'] == 'path':
            if self.kpoints is None:
                tbkpointfile = 'kpoints.dat'
                if 'tbkpointfile' in self.str_params:
                    tbkpointfile = self.str_params['tbkpointfile']
                os.system("mv %s/%s %s/" % (curdir, tbkpointfile, bfdir))
            else:
                bopio.write_kpoints(self.kpoints)
        if len(args) > 0:
            self.set_atoms(args[0])
        if self.modelsbx is None:
            if isinstance(self.atomsbx, str):
                shutil.copyfile(os.path.join(curdir, self.atomsbx)
                                , os.path.join(bfdir, "atoms.bx"))
            else:
                from ase.io.bopfox import write_atomsbx
                write_atomsbx(self.atomsbx)
            if isinstance(self.bondsbx, str):
                shutil.copyfile(os.path.join(curdir, self.bondsbx)
                                , os.path.join(bfdir, "bonds.bx"))
            else:
                from ase.io.bopfox import write_bondsbx
                write_bondsbx(self.bondsbx)
            if self.infox_from_file:
                shutil.copyfile(os.path.join(curdir, self.infoxbx)
                                , os.path.join(bfdir, "infox.bx"))
            else:
                self.write_infox()
        else:
            if isinstance(self.modelsbx, str):
                shutil.copyfile(os.path.join(curdir, self.modelsbx)
                                , os.path.join(bfdir, "models.bx"))
            else:
                self.modelsbx.write()
            if self.infox_from_file:
                shutil.copyfile(os.path.join(curdir, self.infoxbx)
                                , os.path.join(bfdir, "infox.bx"))
            else:
                self.write_infox_new()
        bopio.write_strucbx(self.atoms, filename=self.str_params['strucfile'])
        if 'scfrestart' in self.boo_params:
            if self.boo_params['scfrestart'] == True:
                if isinstance(self.onsite, str):
                    shutil.copyfile(os.path.join(curdir, self.onsite)
                                    , os.path.join(bfdir, "onsite.bx"))
                else:
                    self.write_onsites()
        try:
            if type(self.mem_limit) == int:
                cmd = 'ulimit -v ' + str(self.mem_limit) + '; ' + self.bopfox
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            else:
                cmd = self.bopfox
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output = process.communicate()[0]
            with open('log.bx', 'wb') as p1:
                p1.write(output)
        except KeyboardInterrupt:
            del process
            exit()
        try:
            self.u, self.f, self.s, self.stresses = bopio.read_ufs()
            if self.u == 9.99E99:
                self.bopfox_error = True
        except UnboundLocalError:
            self.u = 0.
            self.f = np.zeros((self.atoms.get_number_of_atoms(), 3))
            self.s = np.zeros((6))
            self.bopfox_error = True
        except:
            self.u = None
            self.f = None
            self.s = None
            self.bopfox_error = True
        if not self.bopfox_error:
            self.bopfox_error = bopio.bopfox_error(filename='log.bx')
        # exits here if bopfox error detected
        if self.bopfox_error and self.ignore_errors == False:
            print('Warning: BOPfox calculation failed.')
            print(('Check log file in ', bfdir))
            print((self.moments))
            os.chdir(curdir)
            return
        try:
            self.fermi_energy = bopio.read_fermi(filename='log.bx')
        except UnboundLocalError:
            self.fermi_energy = 0.
        except:
            self.fermi_energy = None
        try:
            self.magnetic_moments = bopio.get_magnetic_moments(len(self.atoms)
                                                               , filename='log.bx')
        except:
            pass
        # also extracts relaxed atoms
        if self.str_params['task'].lower() == 'relax':
            if bopio.is_converged(filename='log.bx') != True:
                print('Warning: Convergence criterion not reached!')
            f_s = self.str_params['strucfile'].split('.')
            fn = ''
            for i in range(len(f_s) - 1):
                fn += (f_s[i] + '.')
            fn = fn + 'final.bx'
            self.relaxed_atoms = bopio.read_strucbx(fn)
        # and contributions to energy
        self.contributions_energy = \
            bopio.contributions_energy(filename='log.bx')
        # and contributions to forces
        self.contributions_forces = \
            bopio.contributions_forces(filename='log.bx')
        # and contributions to stresses
        self.contributions_stresses = \
            bopio.contributions_stresses(filename='log.bx')
        # and charges
        try:
            self.charges = bopio.get_charges(filename='log.bx')
        except:
            pass
        if self.get_mode() == 'bop':
            try:
                if self._do_moments():
                    self.moments = bopio.get_moments(filename='log.bx')
            except:
                self.moments = None
            try:
                if self._do_anbn():
                    self.an, self.bn = bopio.get_anbn(filename='log.bx')
            except:
                self.an, self.bn = None, None
            # Added the get anhxbnhx for on 05/12/2017 to
            # as part of the overhaul of the bopfox io
            try:
                if self._do_anhxbnhx():
                    # print("Getting anhxnbhx")
                    self.anhx, self.bnhx = bopio.get_anhxbnhx(filename='log.bx')
                # print("GOT THEM")
            except:
                self.anhx, self.bnhx = None, None

            try:
                if self._do_a_inf_b_inf():
                    # print("GETTING A INF ETC")
                    self.a_inf = bopio.get_a_inf_b_inf(filename='log.bx')
                    self.b_inf = self.a_inf
            except:
                self.a_inf, self.b_inf = None, None

        else:
            self.moments = None
            # self.moments_atom = None
        if 'tbkpointmesh' in self.str_params and \
                self.str_params['tbkpointmesh'] == 'path':
            self.eigenvalues = bopio.get_eigenvalues()
            self.orbital_character = bopio.get_orbital_character()
        elif 'printdos' in self.boo_params and \
                self.boo_params['printdos'] == True:
            self.dos = bopio.get_dos()
            self.dos_dict = bopio.get_dos(get_dict=True)
        if self.savelog:
            timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
            self.logfilename = 'bopfoxASE_' + timestamp + 'gmt.out'
            logpath = os.path.join(curdir, self.logfilename)
            shutil.copyfile('log.bx', logpath)
            f_in = open(logpath, 'rb')
            f_out = gzip.open(logpath + '.gz', 'wb')
            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            os.remove(logpath)
        if 'scfsaveonsite' in self.boo_params and \
                self.boo_params['scfsaveonsite'] == True:
            try:
                subprocess.call(['cp', 'onsite.dat', curdir])
            except:
                print("Error saving onsite file.")
            try:
                self.onsite = bopio.get_onsites()
                self.atoms.info['onsites'] = self.onsite
            except:
                pass
        if 'printbonds' in self.boo_params and \
                self.boo_params['printbonds'] == True:
            # try:
            self.bonds, self.dists, self.neighbors \
                = bopio.get_bonds(filename='log.bx')
            # except:
            #     self.bonds = None
            #     self.dists = None
            #     self.neighbors = None
        os.chdir(curdir)
        if self.debug:
            print(('Output files in ', bfdir))
        else:
            shutil.rmtree(bfdir)
        # raise

    def get_onsite_levels(self):
        return self.onsite

    def clean(self):
        return
