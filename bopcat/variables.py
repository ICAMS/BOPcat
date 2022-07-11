# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

"""
Contains all global variables in BOPcat
"""
import os


def periodic_table():
    return ['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au'
        , 'B', 'Ba', 'Be', 'Bh', 'Bi', 'Bk', 'Br'
        , 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Cn', 'Co', 'Cr'
        , 'Cs', 'Cu'
        , 'Db', 'Ds', 'Dy'
        , 'Er', 'Es', 'Eu'
        , 'F', 'Fe', 'Fm', 'Fr'
        , 'Ga', 'Gd', 'Ge'
        , 'H', 'He', 'Hf', 'Hg', 'Ho', 'Hs'
        , 'I', 'In', 'Ir'
        , 'K', 'Kr'
        , 'La', 'Li', 'Lr', 'Lu'
        , 'Md', 'Mg', 'Mn', 'Mo', 'Mt'
        , 'N', 'Na', 'Nb', 'Nd', 'Ne', 'Ni', 'No', 'Np'
        , 'O', 'Os'
        , 'P', 'Pa', 'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu'
        , 'Ra', 'Rb', 'Re', 'Rf', 'Rg', 'Rh', 'Rn', 'Ru'
        , 'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr'
        , 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm'
        , 'U', 'V', 'W', 'Xe', 'Y', 'Yb', 'Zn', 'Zr']


def atomic_properties():
    atprop = {
        'Ag': {'mass': 107.868, 'valence': {'s': 1, 'p': 0, 'd': 10}},
        'Al': {'mass': 26.982, 'valence': {'s': 2, 'p': 1, 'd': 0}},
        'Ar': {'mass': 39.948, 'valence': {'s': 2, 'p': 6, 'd': 0}},
        'Au': {'mass': 196.966, 'valence': {'s': 1, 'p': 0, 'd': 10}},
        'Ba': {'mass': 137.330, 'valence': {'s': 2, 'p': 6, 'd': 0}},
        'Be': {'mass': 9.012, 'valence': {'s': 2, 'p': 0, 'd': 0}},
        'C': {'mass': 12.011, 'valence': {'s': 2, 'p': 2, 'd': 0}},
        'Ca': {'mass': 40.078, 'valence': {'s': 2, 'p': 6, 'd': 0}},
        'Cd': {'mass': 112.410, 'valence': {'s': 2, 'p': 0, 'd': 10}},
        'Co': {'mass': 58.933, 'valence': {'s': 1, 'p': 0, 'd': 8}},
        'Cr': {'mass': 51.996, 'valence': {'s': 1, 'p': 0, 'd': 5}},
        'Cs': {'mass': 132.905, 'valence': {'s': 1, 'p': 6, 'd': 0}},
        'Cu': {'mass': 63.546, 'valence': {'s': 0, 'p': 1, 'd': 10}},
        'Fe': {'mass': 55.847, 'valence': {'s': 1.2, 'p': 0, 'd': 6.8}
            , 'jii': 1, 'free_atom_energy': 0.0, 'stoner': 0.80},
        'Ge': {'mass': 72.610, 'valence': {'s': 2, 'p': 2, 'd': 0}},
        'H': {'mass': 1.0079, 'valence': {'s': 1, 'p': 0, 'd': 0}},
        'He': {'mass': 4.002602, 'valence': {'s': 2, 'p': 0, 'd': 0}},
        'Hf': {'mass': 178.490, 'valence': {'s': 2, 'p': 0, 'd': 2}},
        'Hg': {'mass': 200.590, 'valence': {'s': 2, 'p': 0, 'd': 10}},
        'Ir': {'mass': 192.220, 'valence': {'s': 1, 'p': 0, 'd': 8}},
        'K': {'mass': 39.098, 'valence': {'s': 1, 'p': 0, 'd': 5}},
        'Kr': {'mass': 83.800, 'valence': {'s': 2, 'p': 6, 'd': 0}},
        'Li': {'mass': 6.941, 'valence': {'s': 1, 'p': 0, 'd': 0}},
        'Mg': {'mass': 24.305, 'valence': {'s': 2, 'p': 0, 'd': 0}},
        'Mn': {'mass': 54.938, 'valence': {'s': 2, 'p': 0, 'd': 5}},
        'Mo': {'mass': 95.940, 'valence': {'s': 1, 'p': 0, 'd': 5}},
        'Na': {'mass': 22.990, 'valence': {'s': 1, 'p': 0, 'd': 0}},
        'Ni': {'mass': 58.690, 'valence': {'s': 2, 'p': 0, 'd': 8}},
        'Nb': {'mass': 92.906, 'valence': {'s': 1, 'p': 0, 'd': 4}
            , 'jii': 4, 'free_atom_energy': 0.0},
        'Ne': {'mass': 20.180, 'valence': {'s': 2, 'p': 6, 'd': 0}},
        'Os': {'mass': 190.200, 'valence': {'s': 2, 'p': 6, 'd': 0}},
        'Pd': {'mass': 106.420, 'valence': {'s': 1, 'p': 0, 'd': 9}},
        'Pt': {'mass': 195.080, 'valence': {'s': 1, 'p': 0, 'd': 9}},
        'Rb': {'mass': 85.468, 'valence': {'s': 1, 'p': 0, 'd': 5}},
        'Re': {'mass': 186.207, 'valence': {'s': 2, 'p': 0, 'd': 5}
            , 'jii': 6, 'free_atom_energy': 0.0},
        'Rh': {'mass': 102.906, 'valence': {'s': 1, 'p': 0, 'd': 8}},
        'Ru': {'mass': 101.070, 'valence': {'s': 1, 'p': 0, 'd': 7}},
        'Sc': {'mass': 44.956, 'valence': {'s': 2, 'p': 6, 'd': 1}},
        'Se': {'mass': 78.960, 'valence': {'s': 2, 'p': 4, 'd': 0}},
        'Si': {'mass': 28.085, 'valence': {'s': 2, 'p': 2, 'd': 0}},
        'Sr': {'mass': 87.620, 'valence': {'s': 4, 'p': 4, 'd': 0}},
        'Ta': {'mass': 180.948, 'valence': {'s': 2, 'p': 0, 'd': 3}},
        'Tc': {'mass': 98., 'valence': {'s': 2, 'p': 5, 'd': 0}},
        'Te': {'mass': 127.600, 'valence': {'s': 2, 'p': 4, 'd': 0}},
        'Th': {'mass': 232.038, 'valence': {'s': 0, 'p': 0, 'd': 10}},
        'Ti': {'mass': 47.880, 'valence': {'s': 1, 'p': 0, 'd': 3}},
        'V': {'mass': 50.941, 'valence': {'s': 1, 'p': 0, 'd': 4}},
        'W': {'mass': 183.850, 'valence': {'s': 2, 'p': 0, 'd': 3}},
        'Xe': {'mass': 131.290, 'valence': {'s': 2, 'p': 6, 'd': 0}},
        'Y': {'mass': 88.906, 'valence': {'s': 2, 'p': 6, 'd': 1}},
        'Zn': {'mass': 65.390, 'valence': {'s': 0, 'p': 2, 'd': 10}},
        'Zr': {'mass': 91.224, 'valence': {'s': 2, 'p': 6, 'd': 2}}
    }
    return atprop


def data_comments():
    return ['#', '/', '!']


def required_data_fields(dtype=0):
    if dtype == 0 or dtype.lower() == 'dft':
        required = {'code': 999, 'basis_set': 9, 'xc_functional': 999
            , 'pseudopotential': 99, 'spin': 1
            , 'stoichiometry': '*', 'system_type': 9
            , 'calculation_type': 99
            , 'calculation_order': 9.9, 'space_group': 999}
    else:
        raise NotImplementedError
    return required


def data_keys():
    return {
        "code": 999
        , "basis_set": 9
        , "xc_functional": 999
        , "pseudopotential": 99
        , "hubbard": 0.0
        , "lr_correction": 0
        , "relativistic": 0
        , "valency": {}
        , "number_of_valence_electrons": {}
        , "nlc_correction": 0
        , "encut": 0.0
        , "deltak": 0.0
        , "encut_ok": 0
        , "deltak_ok": 0
        , "spin": 1
        , "data_type": 0
        , "author": 0
        , "stoichiometry": '*'
        , "system_type": 0
        , "calculation_type": 0
        , "calculation_order": 0.0
        , "crystal_type ": 0
        , "space_group": 0
        , "fermi_level": 0.0
        , "coord_k": 'c'
        , 'number_of_k_points': 0
        , "k_points": []
        , 'number_of_bands': 0
        , "eigenvalues": []
        , "occupation": []
        , "orbital_character": []
        , "strucname": '*'
        , 'vacancy_energy': 0.0
        , 'phonons': []
        , 'stresses': []
        , 'structuremap_coordinates': []
        , 'structdb_id': -1
        , 'structdb_NAME': ''
        , 'kmesh': []
        , 'weight': 0.0
        , 'onsites': []
        , 'initial_magnetic_moments': []
    }


def data_key_code():
    return {'vasp': 7
        , 'pbe': 201
        , 'paw': 30
        , 'bulk': 0, 'cluster': 1, 'defect': 2, 'surface': 3, 'interface': 4
        , 'relaxation': 0, 'volume': 1, 'elastic': 2, 'phonon': 3, 'transformation': 4
        , 'unrelaxed': 0, 'relaxed': 1
        , 'unrelax': 0, 'relax_ion': 1, 'relax_volume': 2, 'relax_all': 3
            }


def available_calculators():
    return ['bopfox']


def minimum_distance_isolated_atom():
    return 7.5


def homedirectory():
    src = os.path.abspath(__file__)
    return os.path.dirname(os.path.split(src)[0])


def pathtobetas():
    return homedirectory() + '/betas'


def pathtoonsites():
    return homedirectory() + '/onsites'


def repulsive_keys_bopfox():
    repkeys = ['rep2', 'rep1', 'rep3'
        , 'pairrepulsion', 'eamattraction', 'eamcoupling']
    return repkeys


def beta_names():
    return ['sssigma', 'spsigma', 'sdsigma', 'pssigma'
        , 'ppsigma', 'pppi', 'pdsigma', 'pdpi', 'dssigma'
        , 'dpsigma', 'dppi', 'ddsigma', 'ddpi', 'dddelta']


def bopfox_keys():
    """
    BOPcat tries to update from modules.fp. It it fails, e.g. when it does
    not find the path to bopfox/src the following will be used.
    """
    skeys = ['version', 'forces', 'task', 'model', 'eamversion'
        , 'atomsversion', 'bondsversion', 'repversion'
        , 'screeningversion', 'jijversion', 'bandwidth', 'bopkernel'
        , 'terminator', 'efermimixer', 'scfmixer', 'magconfig', 'scftraj'
        , 'cutoffversion', 'ecutoffversion', 'cutoff2version'
        , 'strucfile', 'modelfile', 'tbkpointmesh', 'tbintegration'
        , 'tbsolver', 'tbkpointfile']
    bkeys = ['screening', 'globalbandwidth', 'scfreusehii', 'scfrigidhiishift'
        , 'scffixmagcom', 'scffixmagdiff', 'scfdamping', 'scfsaveonsite'
        , 'scfrestart', 'printdos', 'dosvsn', 'partialdos', 'fastbop'
        , 'verbose', 'printbonds', 'neighbourhistogram', 'printxi'
        , 'printxi_normed', 'printmu', 'printmu2', 'printmu_averaged'
        , 'printmu_turchi', 'printmu_normed', 'printanbn', 'printanhxbnhx'
        , 'printa_infb_inf', 'printgammadelta', 'printsigma', 'printbo'
        , 'printbotilde', 'printcosphif', 'printdo', 'printtilde'
        , 'printeam', 'printforces', 'printefermi', 'printuatomic'
        , 'printtb', 'printham', 'printsc', 'printtsse', 'printstress'
        , 'printtorque']
    ikeys = ['moments', 'numfdisp', 'nexpmoments', 'efermisteps', 'scfsteps'
        , 'dosgrid', 'tbnbandstotal', 'tbnsmear', 'ioutput']
    fkeys = ['numfinc', 'rskin', 'rthickskin', 'global_a_inf', 'global_b_inf'
        , 'alphaweight', 'efermitol', 'scftol', 'scfmixpara'
        , 'dampinglimit', 'rcut', 'dcut', 'r2cut', 'd2cut', 'ecut'
        , 'decut', 'tbdsmear']
    return skeys, bkeys, ikeys, fkeys


def fit_weights(name='energy'):
    weights = {}
    weights['energy'] = 1.0
    weights['forces'] = 1 / 10.
    weights['vacancy_energy'] = 1 / 10.
    weights['stress'] = 1 / 100.
    weights['eigenvalues'] = 1.0
    return weights[name]


def bopcat_version():
    cwd = os.getcwd()
    os.chdir(homedirectory() + '/bopcat')
    l = open('__init__.py').readlines()
    for i in range(len(l)):
        if '__version__' in l[i]:
            ver = l[i].split('=')[1].strip().strip("'")
    os.chdir(cwd)
    return ver


def is_boplib():
    from ase.calculators import bopfox
    out = False
    if 'ctypes' in dir(bopfox):
        out = True
    return out
