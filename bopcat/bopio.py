#!/usr/bin/env python

# Functionality for reading and writing an ASE atoms object in BOPfox *.bx format.

# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import os
import numpy as np
from ase import Atoms


def write_strucbx(atoms, coord='cartesian', filename='struc.bx'):
    """
    Writes out bopfox structure file. 
    If filename is None, returns the structure file as a list
    """
    file_content = []
    file_content.append("StrucName = struc\n")
    file_content.append("aLat = 1.0\n")
    file_content.append("a1 = % .12f   % .12f   % .12f\n" % (atoms.cell[0][0],
                                                             atoms.cell[0][1],
                                                             atoms.cell[0][2]))
    file_content.append("a2 = % .12f   % .12f   % .12f\n" % (atoms.cell[1][0],
                                                             atoms.cell[1][1],
                                                             atoms.cell[1][2]))
    file_content.append("a3 = % .12f   % .12f   % .12f\n" % (atoms.cell[2][0],
                                                             atoms.cell[2][1],
                                                             atoms.cell[2][2]))
    if coord.lower()[0] == 'c':
        file_content.append("coord = cartesian\n")
        r = atoms.get_positions()
    elif coord.lower()[0] == 'd':
        file_content.append("coord = direct\n")
        r = atoms.get_scaled_positions()
    else:
        raise ValueError("Input coord %s incorrect." % coord)
    symbols = atoms.get_chemical_symbols()
    tags = atoms.get_tags()
    for i in range(len(atoms)):
        if (tags[i] == 0):
            fix = "F F F"  # false
        elif (tags[i] == 1):
            fix = "T T T"  # true
        file_content.append("%s % .12f % .12f % .12f \n" % (symbols[i]
                                                            , r[i][0], r[i][1], r[i][2]))
    if atoms.has('magmoms') or atoms.has('initial_magmoms'):
        magmoms = atoms.get_initial_magnetic_moments()
        file_content.append("magnetisation = true\n")
        # check shape of magmoms!
        if magmoms.shape == (atoms.get_number_of_atoms(), 3):
            for i in range(len(atoms)):
                mm = atoms[i].magmom
                # check tag of atom to identify d, p or s shell character
                file_content.append("%3.4f " % mm[0])
                file_content.append("%3.4f " % mm[1])
                file_content.append("%3.4f \n" % mm[2])
                # file_content.append("0 0 0 0 0 0 / %3d\n" % (i+1))
        elif magmoms.shape == (atoms.get_number_of_atoms(),):
            for i in range(len(atoms)):
                mm = atoms[i].magmom
                file_content.append("0 0 ")
                file_content.append("%3.4f \n" % mm)
                # file_content.append("0 0 0 0 0 0 / %3d\n" % (i+1))
        else:
            raise Exception
    file_content.append("\n")

    if (filename == None):
        return file_content
    else:
        fout = open(filename, 'w')
        for line in file_content:
            fout.write(line)
        fout.close()


def read_ufs(filename='log.bx'):
    """
    Reads energy, forces and stress from the log.bx file
    """
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    u = 1.99E99  # for fitting purposes sometimes guess parameters
    # lead to crashing of BOPfox
    s = np.ones(6) * 9.99E99
    temp = 0
    for line in log:
        if not line.find('init: N(total)') == -1:
            N = int(float(line.split()[4]))
            f = np.zeros((N, 3))
            stresses = np.zeros((N, 6))
        elif not line.find('U_binding/atom') == -1:
            try:
                u = float(line.split()[1]) * N
            except ValueError:
                u = 9.99E99
        elif not line.find('FBOP (total   )') == -1:
            try:
                i = int(float(line.split()[3]))
                f[i - 1] = [float(x) for x in line.split()[4:7]]
                temp += 1
                assert (np.shape(f[i - 1]) == (3,))
            except:
                f[temp] = np.ones(3) * 9.99E99
                temp += 1
        elif not line.find('stress (') == -1:
            try:
                atom_number = int(line.split()[3])
                stresses[atom_number - 1, :] = \
                    np.array([float(x) for x in line.split()[4:10]])
                assert (np.shape(stresses[atom_number - 1]) == (6,))
            except:
                stresses = np.ones((N, 6)) * 9.99E99
        elif not line.find('sum(stress)/volume') == -1:
            try:
                s = np.array([float(x) for x in line.split()[1:]])
                assert (np.shape(s) == (6,))
            except:
                s = np.ones(6) * 9.99E99
        elif not line.find('unnorm. stress') == -1:
            try:
                atom_number = int(line.split()[2])
                stresses[atom_number - 1, :] = \
                    np.array([float(x) for x in line.split()[3:9]])
                assert (np.shape(stresses[atom_number - 1]) == (6,))
            except:
                stresses = np.ones((N, 6)) * 9.99E99
        elif not line.find('global stress/volume') == -1:
            # depends on the BOPfox version
            # old version:
            try:
                j = 0
                for nextline in log:
                    if j >= 3: break
                    s[j] = [float(x) for x in nextline.split()]
                    j += 1
                assert (np.shape(s) == (6,))
            except:
                s = np.ones(6) * 9.99E99
        elif not line.find('unnormalized stress on atom') == -1:
            try:
                i = int(float(line.split()[4][:-1]))
                j = 0
                for nextline in log:
                    if j >= 3: break
                    stresses[i - 1, j] = [float(x) for x in nextline.split()]
                    j += 1
                assert (np.shape(stresses[i - 1]) == (6,))
            except:
                stresses = np.ones((N, 6)) * 9.99E99
    log.close()
    return u, f, s, np.array(stresses)


def check_bonds(bopbonds):
    if bopbonds.version is None:
        raise ValueError('Version should be set to proceed.')
    if bopbonds.bond is None:
        raise ValueError('Bond should be set to proceed.')
    if bopbonds.valence is None:
        raise ValueError('Valence should be set to proceed.')
    else:
        for i in range(2):
            if bopbonds.valence[i] == 's':
                if bopbonds.sssigma is None:
                    raise ValueError('sssigma should be set to proceed.')
            elif bopbonds.valence[i] == 'sp':
                if bopbonds.sssigma is None:
                    raise ValueError('sssigma should be set to proceed.')
                if bopbonds.ppsigma is None:
                    raise ValueError('ppsigma should be set to proceed.')
                if bopbonds.pppi is None:
                    raise ValueError('pppi should be set to proceed.')
                if bopbonds.spsigma is None:
                    raise ValueError('spsigma should be set to proceed.')
                if bopbonds.pssigma is None:
                    raise ValueError('pssigma should be set to proceed.')
            elif bopbonds.valence[i] == 'd':
                if bopbonds.ddsigma is None:
                    raise ValueError('ddsigma should be set to proceed.')
                if bopbonds.ddpi is None:
                    raise ValueError('ddpi should be set to proceed.')
                if bopbonds.dddelta is None:
                    raise ValueError('dddelta should be set to proceed.')
            if bopbonds.valence[0] == bopbonds.valence[1]:
                break


def write_bondsbx(bopbonds, filename='bonds.bx', update=False):
    import warnings
    warnings.warn('Use write function of bondsbx oject', DeprecationWarning)
    # Check if necessary parameters were defined otherwise cannot
    # write bonds.bx
    if update == True:
        from .bopmodel import read_bondsbx
        bonds = read_bondsbx(filename=filename)
        for i in range(len(bopbonds)):
            check_bonds(bopbonds[i])
            for j in range(len(bonds)):
                old_bonds = [bonds[j].bond[0], bonds[j].bond[1]]
                new_bonds = [bopbonds[i].bond[0], bopbonds[i].bond[1]]
                if bonds[j].get_version() == bopbonds[i].get_version() and \
                        old_bonds == new_bonds:
                    bonds[j] = bopbonds[i]
    else:
        bonds = bopbonds
    versions = []
    for i in range(len(bonds)):
        version = bonds[i].get_version()
        if version not in versions:
            versions.append(version)
    f = open(filename, 'w')
    f.write('/bonds.bx written in ASE\n')
    dash = '/-----------------------------------------------------------\n'
    for i in range(len(versions)):
        marker = 0
        for j in range(len(bonds)):
            if bonds[j].get_version() == versions[i]:
                if marker == 0:
                    f.write(dash)
                    f.write('Version = %s\n' % versions[i])
                    f.write(dash)
                    f.write('\n')
                f.write('bond           =   %s %s \n' % (bonds[j].bond[0]
                                                         , bonds[j].bond[1]))
                f.write('valence        =   %s %s \n' % (bonds[j].valence[0]
                                                         , bonds[j].valence[1]))
                f.write('scaling        =%10.5f \n' % bonds[j].scaling[0])
                for key, list_val in list(bonds[j].get_bondspar().items()):
                    if list_val is not None:
                        if isinstance(list_val, list):
                            func = bonds[j].get_version()
                            para = list(list_val)
                            f.write('{0:14} = '.format(key))
                        else:
                            func = list_val.get_name(environment='bopfox-ham')
                            para = list_val.get_numbers()
                            f.write('{0:14} = {1:15}'.format(key, func))
                            # f.write('{0:14} = '.format(key))
                        for val in para:
                            f.write('%10.5f ' % val)
                        f.write('\n')
                for key, list_val in list(bonds[j].get_overlappar().items()):
                    if list_val is not None:
                        if isinstance(list_val, list):
                            func = bonds[j].get_version()
                            para = list(list_val)
                            f.write('{0:14} = '.format(key))
                        else:
                            func = list_val.get_name(environment='bopfox-ham')
                            para = list_val.get_numbers()
                            f.write('{0:14} = {1:15}'.format(key, func))
                            # f.write('{0:14} = '.format(key))
                        for val in para:
                            f.write('%10.5f ' % val)
                        f.write('\n')
                for key, list_val in list(bonds[j].get_repetal().items()):
                    if list_val is not None:
                        if isinstance(list_val, list):
                            func = bonds[j].get_version()
                            para = list(list_val)
                            f.write('{0:14} = '.format(key))
                        else:
                            func = list_val.get_name(environment='bopfox-rep')
                            para = list_val.get_numbers()
                            f.write('{0:14} = {1:15}'.format(key, func))
                            # f.write('{0:14} = '.format(key))
                        for val in para:
                            f.write('%10.5f ' % val)
                        f.write('\n')
                marker += 1
                f.write('\n')
        f.write('\n')
    f.close()


def write_atomsbx(bopatoms, filename='atoms.bx', update=False):
    # Warning: The keyword update does not necessarily mean that atomsbx
    # is updated. It just means that succeeding entries will not
    # have the header 'Version'. To update an atomsbx file, use
    # update_atomsbx
    import warnings
    warnings.warn('Use write function of bondsbx oject', DeprecationWarning)
    if update == True:
        from .bopmodel import read_atomsbx
        atoms = read_atomsbx(filename=filename)
        for i in range(len(bopatoms)):
            for j in range(len(atoms)):
                if atoms[j].get_version() == bopatoms[i].get_version() and \
                        atoms[j].get_atom() == bopatoms[i].get_atom():
                    atoms[j] = bopatoms[i]
                    break
    else:
        atoms = bopatoms
    versions = []
    for i in range(len(atoms)):
        version = atoms[i].get_version()
        if version not in versions:
            versions.append(version)
    f = open(filename, 'w')
    f.write('/atoms.bx written in ASE\n')
    dash = '/-----------------------------------------------------------\n'
    for i in range(len(versions)):
        marker = 0
        for j in range(len(atoms)):
            if atoms[j].get_version() == versions[i]:
                if marker == 0:
                    f.write(dash)
                    f.write('Version = %s\n' % versions[i])
                    f.write(dash)
                    f.write('\n')
                f.write('Atom             = %s\n' % atoms[j].get_atom())
                f.write('Mass             = %s\n' % atoms[j].get_mass())
                for key, list_val in list(atoms[j].get_atomspar().items()):
                    if key in ['valenceorbitals', 'norbscreen'] \
                            and list_val is not None:
                        f.write('{0:16} = {1:10}\n'.format(key
                                                           , int(list_val[0])))
                    elif key in ['onsitelevels'] \
                            and list_val is not None:
                        f.write('{0:16} =  '.format(key))
                        for v in range(len(list_val)):
                            f.write('%10.5f ' % list_val[v])
                        f.write('\n')
                    elif list_val is not None and type(list_val) is list:
                        f.write('{0:16} = '.format(key))
                        for val in list_val:
                            f.write('%10.5f ' % val)
                        f.write('\n')
                    elif list_val is not None and type(list_val) is not list:
                        f.write('{0:16} = {1:10}\n'.format(key, list_val))
                f.write('\n')
                marker += 1
                f.write('\n')
    f.close()


def write_kpoints(path_k, filename='kpoints.dat'):
    """ 
    Writes a k-point file for use in TB calculations
    """
    path_k, Nk = path_k
    f = open(filename, 'w')
    for i in range(len(path_k) - 1):
        for j in range(3):
            f.write('%3.5f  ' % path_k[i][j])
        f.write('\n')
        f.write('%d \n' % Nk)
    for j in range(3):
        f.write('%3.5f  ' % path_k[-1][j])
    f.close()


def read_strucbx(filename='struc.final.bx'):
    """
    Converts a bopfox structure file into an ASE Atoms object.
    """
    import string
    elements = ['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au'
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
    if (type(filename) == str):
        f = open(filename)
        l = f.readlines()
        f.close()
    elif (type(filename) == list or type(filename) == np.ndarray):
        l = filename
    cell = np.zeros([3, 3])
    pos = []
    ele = []
    magmoms = []
    tags = []
    chars = string.punctuation
    for i in range(len(l)):
        if len(l[i].split()) > 1:
            if l[i].strip()[0][0] not in chars:
                l_s = l[i].split()
                if l_s[0].lower() == 'alat':
                    alat = float(l_s[2])
                elif l_s[0].lower() == 'a1':
                    cell[0] = l_s[2:5]
                elif l_s[0].lower() == 'a2':
                    cell[1] = l_s[2:5]
                elif l_s[0].lower() == 'a3':
                    cell[2] = l_s[2:5]
                elif l_s[0].lower() == 'coord':
                    coord = l_s[2]
                elif l_s[0].lower() == 'strucname':
                    struc = l_s[2]
                elif l_s[0].lower() == 'magnetisation':
                    if l[i].split('=')[1].lower().strip() == 'true':
                        for j in range(i + 1, len(l)):
                            l_s = l[j].split()
                            if len(l_s) > 8 and \
                                    l[j].strip()[0][0] not in chars:
                                try:
                                    mom = np.array(l_s[:9], dtype=float)
                                    magmoms.append(mom[np.nonzero(mom)[0][0]])
                                except:
                                    print('Error reading magnetic moments.')
                            elif len(l_s) > 2 and \
                                    l[j].strip()[0][0] not in chars:
                                try:
                                    mom = np.array(l_s[:3], dtype=float)
                                    magmoms.append(mom[np.nonzero(mom)[0][0]])
                                except:
                                    print('Error reading magnetic moments.')
                    break
                else:
                    try:
                        if l_s[0] in elements:
                            ele.append(l_s[0])
                        else:
                            # try to decipher it
                            if l_s[0:2] in elements:
                                ele.append(l_s[0:2])
                            elif l_s[0][0] in elements:
                                ele.append(l_s[0][0])
                            else:
                                err = 'Tried but cannot determine symbols!'
                                raise RuntimeError
                        pos.append([float(l_s[1]), float(l_s[2]), float(l_s[3])])
                        if len(l_s) > 4:
                            if 'T' in l_s[4]:
                                tags.append(1)
                            else:
                                tags.append(0)
                    except:
                        pass

    pos = np.array(pos)
    if coord[0].lower() == 'd':
        atom = Atoms(symbols=ele, cell=cell * alat, scaled_positions=pos
                     , pbc=[True, True, True])
    elif coord[0].lower() == 'c':
        atom = Atoms(symbols=ele, cell=cell * alat, positions=pos
                     , pbc=[True, True, True])
    else:
        raise Exception('Unrecognized coordinate type!')
    # add tags and magnetic moments
    if tags != []:
        try:
            atom.set_tags(tags)
        except:
            print('Error setting tags')
    if magmoms != []:
        try:
            atom.set_initial_magnetic_moments(magmoms)
        except:
            print('Error setting magnetic moments')
    return atom


def is_converged(filename='log.bx'):
    """
    Checks if BOPfox calculation is properly converged
    """
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    conv = False
    for i in range(len(l)):
        if l[i].find('convergence limit reached') != -1:
            conv = True
            break
    return conv


def bopfox_error(filename='log.bx'):
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    err = True
    for i in range(len(l)):
        if 'BOPfox calculation is finished!' in l[i]:
            err = False
    return err


def contributions_energy(filename='log.bx'):
    """ 
    Returns the contributions to the energy from a log file as a dictionary
    """
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    rep_keys = ['rep%d' % i for i in range(100)]
    ene_keys = ['binding', 'bond', 'prom', 'exchange', 'ionic', 'coulomb'
                   , 'env', 'pair', 'rep'] + rep_keys
    out = {}
    for key in ene_keys:
        out[key] = {}
    success = False
    for i in range(len(l)):
        if 'U_' not in l[i]:
            continue
        split = l[i].split()
        if len(split) < 2:
            continue
        key = split[0].lower()
        key = key.split('u_')[1].strip()
        if '/atom' in key:
            key = key.split('/atom')[0].strip()
            try:
                N = max(out[key].keys())
                out[key][0] = float(split[-1]) * N
            except:
                out[key][0] = 9.99E99
        elif key in ene_keys:
            try:
                out[key][int(split[-3])] = float(split[-1])
            except:
                out[key][int(split[-3])] = 9.99E99
            success = True
    # 
    if not success:
        out = None
    return out


def contributions_forces(filename='log.bx'):
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    rep_keys = ['rep%d' % i for i in range(100)]
    f_keys = ['analytic', 'eam', 'coulomb', 'total'] + rep_keys
    out = {}
    for key in f_keys:
        out[key] = {}
    success = False
    for i in range(len(l)):
        if 'FBOP' not in l[i]:
            continue
        split = l[i].split()
        if len(split) < 2:
            continue
        key = split[1].lower().strip('(').strip().strip(')').strip()
        if key in f_keys:
            try:
                out[key][int(split[-7])] = np.array([float(split[k]) \
                                                     for k in [-6, -5, -4]])
            except:
                # out[key][int(split[-7])] = [9.99E99]*3
                success = False
                break
            success = True
    for key, val in list(out.items()):
        if len(val) > 0:
            fsum = np.zeros(3)
            for i in list(val.keys()):
                fsum += val[i]
            out[key][0] = fsum
    if not success:
        out = None
    return out


def contributions_stresses(filename='log.bx'):
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    rep_keys = ['rep%d' % i for i in range(100)]
    s_keys = ['bond', 'total'] + rep_keys
    out = {}
    for key in s_keys:
        out[key] = {}
    success = False
    for i in range(len(l)):
        if 'stress' not in l[i]:
            continue
        split = l[i].split()
        if len(split) < 2:
            continue
        key = split[1].lower().strip('(').strip().strip(')').strip()
        if key in s_keys:
            try:
                out[key][int(split[-7])] = np.array([float(split[k]) \
                                                     for k in [-6, -5, -4, -3, -2, -1]])
            except:
                # out[key][int(split[-7])] = [9.99E99]*3
                success = False
                break
            success = True
    for key, val in list(out.items()):
        if len(val) > 0:
            ssum = np.zeros(6)
            for i in list(val.keys()):
                ssum += val[i]
            out[key][0] = ssum
    if not success:
        out = None
    return out


def get_magnetic_moments(N, filename='log.bx'):
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    magmoms = np.zeros(N)
    for i in range(len(l)):
        if 'Mag_mom' in l[i]:
            split = l[i].split()
            magmoms[int(split[5]) - 1] = float(split[-1])
    return magmoms


def get_onsites(filename='onsite.dat'):
    with open(filename)  as f:
        l = f.readlines()
    # onsite = np.array([float(l[i].split()[-1]) for i in range(len(l))])
    onsites = []
    if len(l[0].split()) == 4:
        onsites = np.array([[float(l[i].split()[-1])] for i in range(len(l))])
    else:
        onsites = np.array([[float(l[i].split()[-1])
                                , float(l[i].split()[-2])] for i in range(len(l))])
    return onsites


def get_moments(filename='log.bx'):
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    moments_dic = {}
    n_atoms = 0

    # Collect final moments in a dictionary
    for i in range(len(l)):
        if 'Avg.Mu ' in l[i]:
            split = l[i].replace("-------->", "").split()
            orbital = split[7][0]
            atom = split[4][:-1]
            if n_atoms < int(atom):
                n_atoms = int(atom)
            temp = []
            # Keep 6 for downwards compability, can be changed to 9 from
            # rev 161
            for num in split[6:]:
                try:
                    num = float(num)
                    temp.append(num)
                except:
                    pass
            moments_dic[atom + ' ' + orbital] = temp
        if 'Avg.Mu/atom' in l[i]:
            split = l[i].replace("-------->", "").split()
            temp = []
            for num in split[1:]:
                try:
                    num = float(num)
                    temp.append(num)
                except:
                    pass
                moments_dic['0 ave'] = temp

    moments = [[] for i in range(n_atoms + 1)]

    n_orb = {'ave': 1, 's': 1, 'p': 3, 'd': 5, 'f': 7}
    atom_numbers = []
    # Collect all atom indices
    for key in list(moments_dic.keys()):
        atom = int(key.split()[0])
        if not (atom in atom_numbers):
            atom_numbers.append(atom)
    orbitals = [[] for i in range(len(atom_numbers))]
    # Collect orbitals for each atom
    for key in list(moments_dic.keys()):
        for atom_i, atom in enumerate(atom_numbers):
            temp_atom = int(key.split()[0])
            if atom == temp_atom:
                orbital = key.split()[1]
                if not (orbital in orbitals[atom_i]):
                    orbitals[atom_i].append(orbital)
    # Calculate total number of orbitals for each atom
    n_orbs_tot = [0 for i in range(n_atoms + 1)]
    for atom_i in range(len(atom_numbers)):
        for orbital in orbitals[atom_i]:
            n_orbs_tot[atom_i] += n_orb[orbital]
    # Summarize moments in a list
    for key in list(moments_dic.keys()):
        atom = int(key.split()[0])
        orbital = key.split()[1]
        if moments[atom] == []:
            for i_mom in range(len(moments_dic[key])):
                moments[atom].append(moments_dic[key][i_mom]
                                     * n_orb[orbital])
        else:
            for i_mom in range(len(moments[atom])):
                moments[atom][i_mom] += moments_dic[key][i_mom] * \
                                        n_orb[orbital]

    # average all moments over the number of orbitals
    for atom in range(len(moments)):
        atom_i = np.argwhere(np.array(atom_numbers) == atom)[0][0]
        for i_mom in range(len(moments[atom])):
            moments[atom][i_mom] = moments[atom][i_mom] \
                                   / n_orbs_tot[atom_i]

    return moments


def get_anbn(filename='log.bx'):
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    an_dic = {}
    bn_dic = {}
    n_atoms = 0

    # Collect final an in a dictionary
    for i in range(len(l)):
        if l[i].split("(")[0].replace(" ", '') == "an" or \
                l[i].split("(")[0].replace(" ", '') == "bn":
            # Prepare the string for further evaluation 
            # by stripping it down
            split = l[i].replace("-------->", "") \
                .replace("=", '').replace("atom", '')
            split = split.replace("(", '').replace(")", '')
            split = split.replace("orbital", '').replace("orb", '') \
                .replace(",", '').split()
            orbital = split[2]
            atom = split[1]
            if n_atoms < int(atom):
                n_atoms = int(atom)
            temp = []
            for num in split[3:]:

                try:
                    num = float(num)
                    temp.append(num)
                except:
                    pass
            if l[i].split("(")[0].replace(" ", '') == "an":
                an_dic[atom + ' ' + orbital] = temp
            if l[i].split("(")[0].replace(" ", '') == "bn":
                bn_dic[atom + ' ' + orbital] = temp

    an = [[] for i in range(n_atoms)]
    bn = [[] for i in range(n_atoms)]

    n_orb = {'ave': 1, 's': 1, 'p': 3, 'd': 5, 'f': 7}
    atom_numbers = []
    # Collect all atom indices
    for key in list(an_dic.keys()):
        atom = int(key.split()[0])
        if not (atom in atom_numbers):
            atom_numbers.append(atom)
    orbitals = [[] for i in range(len(atom_numbers))]
    # Collect orbitals for each atom
    for key in list(an_dic.keys()):
        for atom_i, atom in enumerate(atom_numbers):
            temp_atom = int(key.split()[0])
            if atom == temp_atom:
                orbital = key.split()[1]
                if not (orbital in orbitals[atom_i]):
                    orbitals[atom_i].append(orbital)
    # Calculate total number of orbitals for each atom
    n_orbs_tot = [0 for i in range(n_atoms)]
    for atom_i in range(len(atom_numbers)):
        for orbital in orbitals[atom_i]:
            n_orbs_tot[atom_i] += n_orb[orbital]
    # Summarize recursion coefficients in a list
    for key in list(an_dic.keys()):
        atom = int(key.split()[0]) - 1
        orbital = key.split()[1]
        if an[atom] == []:
            for i_mom in range(len(an_dic[key])):
                an[atom].append(an_dic[key][i_mom]
                                * n_orb[orbital])
        else:
            for i_mom in range(len(an[atom])):
                an[atom][i_mom] += an_dic[key][i_mom] * \
                                   n_orb[orbital]

    # average all an over the number of orbitals
    for atom in range(1, len(an) + 1):
        atom_i = np.argwhere(np.array(atom_numbers) == atom)[0][0]
        for i_mom in range(len(an[atom - 1])):
            an[atom - 1][i_mom] = an[atom - 1][i_mom] \
                                  / n_orbs_tot[atom_i]

    atom_numbers = []
    # Collect all atom indices
    for key in list(bn_dic.keys()):
        atom = int(key.split()[0])
        if not (atom in atom_numbers):
            atom_numbers.append(atom)
    orbitals = [[] for i in range(len(atom_numbers))]
    # Collect orbitals for each atom
    for key in list(bn_dic.keys()):
        for atom_i, atom in enumerate(atom_numbers):
            temp_atom = int(key.split()[0])
            if atom == temp_atom:
                orbital = key.split()[1]
                if not (orbital in orbitals[atom_i]):
                    orbitals[atom_i].append(orbital)
    # Calculate total number of orbitals for each atom
    n_orbs_tot = [0 for i in range(n_atoms)]
    for atom_i in range(len(atom_numbers)):
        for orbital in orbitals[atom_i]:
            n_orbs_tot[atom_i] += n_orb[orbital]
    # Summarize recursion coefficients in a list
    for key in list(bn_dic.keys()):
        atom = int(key.split()[0]) - 1
        orbital = key.split()[1]
        if bn[atom] == []:
            for i_mom in range(len(bn_dic[key])):
                bn[atom].append(bn_dic[key][i_mom]
                                * n_orb[orbital])
        else:
            for i_mom in range(len(bn[atom])):
                bn[atom][i_mom] += bn_dic[key][i_mom] * \
                                   n_orb[orbital]

    # average all bn over the number of orbitals
    for atom in range(1, len(bn) + 1):
        atom_i = np.argwhere(np.array(atom_numbers) == atom)[0][0]
        for i_mom in range(len(bn[atom - 1])):
            bn[atom - 1][i_mom] = bn[atom - 1][i_mom] \
                                  / n_orbs_tot[atom_i]

    return an, bn


def get_anhxbnhx(filename='log.bx'):
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    an_dic = {}
    bn_dic = {}
    n_atoms = 0

    # Collect final an in a dictionary
    for i in range(len(l)):
        # print(l[i].split("(")[0])
        # print("an" in l[i].split("(")[0] or "bn" in l[i].split("(")[0])
        # print(' an  (' in l[i] or ' bn  (' in l[i])
        # if ' an  (' in l[i] or ' bn  (' in l[i]:
        if l[i].split("(")[0].replace(" ", '') == "anhx" or l[i].split("(")[0].replace(" ", '') == "bnhx":

            # Prepare the string for further evaluation by stripping it down

            split = l[i].replace("-------->", "").replace("=", '').replace("atom", '')
            split = split.replace("(", '').replace(")", '')
            split = split.replace("orbital", '').replace("orb", '').replace(",", '').split()
            orbital = split[2]
            atom = split[1]
            if n_atoms < int(atom):
                n_atoms = int(atom)
            temp = []
            for num in split[3:]:

                try:
                    num = float(num)
                    temp.append(num)
                except:
                    pass
            if l[i].split("(")[0].replace(" ", '') == "anhx":
                an_dic[atom + ' ' + orbital] = temp
            if l[i].split("(")[0].replace(" ", '') == "bnhx":
                bn_dic[atom + ' ' + orbital] = temp

    an = [[] for i in range(n_atoms)]
    bn = [[] for i in range(n_atoms)]

    n_orb = {'ave': 1, 's': 1, 'p': 3, 'd': 5, 'f': 7}
    atom_numbers = []
    # Collect all atom indices
    for key in list(an_dic.keys()):
        atom = int(key.split()[0])
        if not (atom in atom_numbers):
            atom_numbers.append(atom)
    orbitals = [[] for i in range(len(atom_numbers))]
    # Collect orbitals for each atom
    for key in list(an_dic.keys()):
        for atom_i, atom in enumerate(atom_numbers):
            temp_atom = int(key.split()[0])
            if atom == temp_atom:
                orbital = key.split()[1]
                if not (orbital in orbitals[atom_i]):
                    orbitals[atom_i].append(orbital)
    # Calculate total number of orbitals for each atom
    n_orbs_tot = [0 for i in range(n_atoms)]
    for atom_i in range(len(atom_numbers)):
        for orbital in orbitals[atom_i]:
            n_orbs_tot[atom_i] += n_orb[orbital]
    # Summarize recursion coefficients in a list
    for key in list(an_dic.keys()):
        atom = int(key.split()[0]) - 1
        orbital = key.split()[1]
        if an[atom] == []:
            for i_mom in range(len(an_dic[key])):
                an[atom].append(an_dic[key][i_mom]
                                * n_orb[orbital])
        else:
            for i_mom in range(len(an[atom])):
                an[atom][i_mom] += an_dic[key][i_mom] * \
                                   n_orb[orbital]

    # average all an over the number of orbitals
    for atom in range(1, len(an) + 1):
        atom_i = np.argwhere(np.array(atom_numbers) == atom)[0][0]
        for i_mom in range(len(an[atom - 1])):
            an[atom - 1][i_mom] = an[atom - 1][i_mom] \
                                  / n_orbs_tot[atom_i]

    atom_numbers = []
    # Collect all atom indices
    for key in list(bn_dic.keys()):
        atom = int(key.split()[0])
        if not (atom in atom_numbers):
            atom_numbers.append(atom)
    orbitals = [[] for i in range(len(atom_numbers))]
    # Collect orbitals for each atom
    for key in list(bn_dic.keys()):
        for atom_i, atom in enumerate(atom_numbers):
            temp_atom = int(key.split()[0])
            if atom == temp_atom:
                orbital = key.split()[1]
                if not (orbital in orbitals[atom_i]):
                    orbitals[atom_i].append(orbital)
    # Calculate total number of orbitals for each atom
    n_orbs_tot = [0 for i in range(n_atoms)]
    for atom_i in range(len(atom_numbers)):
        for orbital in orbitals[atom_i]:
            n_orbs_tot[atom_i] += n_orb[orbital]
    # Summarize recursion coefficients in a list
    for key in list(bn_dic.keys()):
        atom = int(key.split()[0]) - 1
        orbital = key.split()[1]
        if bn[atom] == []:
            for i_mom in range(len(bn_dic[key])):
                bn[atom].append(bn_dic[key][i_mom]
                                * n_orb[orbital])
        else:
            for i_mom in range(len(bn[atom])):
                bn[atom][i_mom] += bn_dic[key][i_mom] * \
                                   n_orb[orbital]

    # average all bn over the number of orbitals
    for atom in range(1, len(bn) + 1):
        atom_i = np.argwhere(np.array(atom_numbers) == atom)[0][0]
        for i_mom in range(len(bn[atom - 1])):
            bn[atom - 1][i_mom] = bn[atom - 1][i_mom] \
                                  / n_orbs_tot[atom_i]

    return an, bn


def get_a_inf_b_inf(filename="log.bx"):
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    lines = log.readlines()
    log.close()
    inf_dics = []
    for l in lines:
        if "aInf/bInf" in l:
            dic = {}
            # Replaces the arrow from version < 198
            l = l.replace("-------->", "")
            # Removes excess symbols
            if not "global" in l:
                l = l.replace(")", "").replace("=", "").split()
                spin = l[-3]
                orb = l[-5]
                atom = l[-6]
                dic["orb"] = orb
                dic["spin"] = spin
                dic["atom"] = atom
                dic["a_inf"] = l[-2]
                dic["b_inf"] = l[-1]
            else:
                l = l.split()
                dic["orb"] = "global"
                dic["spin"] = "global"
                dic["atom"] = "global"
                dic["a_inf"] = l[-2]
                dic["b_inf"] = l[-1]
            inf_dics.append(dic)
    return inf_dics


def get_charges(filename='log.bx'):
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    charges = []
    for i in range(len(l)):
        if "Charge (" in l[i]:
            s = l[i].split()
            charges.append(float(s[7]))
    charge_transferred = 0.
    for i in range(len(charges)):
        if charges[i] > 0.:
            charge_transferred += charges[i]
    charges.insert(0, charge_transferred / len(charges))
    return np.array(charges)


def get_eigenvalues(filename='bands.dat'):
    f = open(filename)
    l = f.readlines()
    f.close()
    eigenvalues = []
    for i in range(len(l)):
        s = l[i].split()
        for j in range(len(s)):
            try:
                s[j] = float(s[j])
            except:
                # sometimes during fitting eigenvalues go haywire
                raise ValueError("BOPfox eigenvalues are not readable.")
        eigenvalues.append(s)
    eigenvalues = np.array(eigenvalues)
    return eigenvalues


def get_orbital_character(filename='bands_character.dat'):
    f = open(filename)
    l = f.readlines()
    f.close()
    orb_char = []
    temp = []
    for i in range(len(l)):
        if 'band\m' in l[i]:
            if len(temp) > 1:
                orb_char.append(list(temp))
                temp = []
        elif 'k-point' in l[i]:
            pass
        else:
            s = l[i].split()[1:-1]
            s = [float(j) for j in s]
            temp.append(s)
    if len(temp) > 1:
        orb_char.append(temp)
    orb_char = np.array(orb_char)
    return orb_char


def read_fermi(filename='log.bx'):
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    fermi_energy = 0
    for i in range(len(l)):
        if 'E_Fermi' in l[i]:
            fermi_energy = float(l[i].split()[-1])
            break
    return fermi_energy


"""
def get_dos():
    dirs = os.listdir(os.getcwd()) 
    ef = read_fermi()
    for d in dirs:
        if "DOS" in d:
            fn = d
            break
    l = open(fn).readlines()
    dos = np.zeros((len(l)-1,len(l[1].split()))) 
    for i in range(1,len(l)-1):
        dos[i] = l[i].split()
    dos = dos.T
    dos[0] -= ef
    return dos
"""


def get_dos(get_dict=False):
    dirs = os.listdir(os.getcwd())
    ef = read_fermi()
    for d in dirs:
        if "DOS" in d:
            fn = d
            break
    l = open(fn).readlines()
    dos = np.zeros((len(l) - 1, len(l[1].split())))
    if not get_dict:
        for i in range(1, len(l) - 1):
            dos[i] = l[i].split()
        dos = dos.T
        dos[0] -= ef
        return dos
    else:
        for i in range(0, len(l) - 1):
            dos[i] = l[i + 1].split()
        dos_dict = {}
        dos = dos.T
        for index, key in enumerate(l[0].split("\t")[0:-1]):
            f = key.strip("#")
            dos_dict[f] = dos[index]
        dos_dict["efermi"] = ef
        return dos_dict


def get_relaxed_atom(filename='log.bx'):
    """
    Returns the relaxed atoms structures from log.bx ?
    """
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    pos = []
    sym = []
    cell = np.zeros((3, 3))
    for i in range(len(l)):
        if 'init: N(total)' in l[i]:
            N = int(float(l[i].split()[4]))
            pos = np.zeros((N, 3))
            sym = np.empty(N, dtype=str)
        elif 'init: cell(:' in l[i]:
            for j in range(3):
                cell[j] = l[i + j].split()[-3:]
        elif 'init: atom/type/pos/fix:' in l[i]:
            j = int(float(l[i].split()[2]))
            sym[j - 1] = l[i].split()[3]
        elif 'FBOP (total   )' in l[i]:
            j = int(float(l[i].split()[3]))
            pos[j - 1] = [float(x) for x in l[i].split()[7:10]]
    atom = Atoms(cell=cell, positions=pos, symbols=sym, pbc=True)
    return atom


def get_initial_atom(filename='log.bx'):
    """ Obtains the initial atoms structure ?
    """
    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()
    pos = []
    sym = []
    cell = np.zeros((3, 3))
    for i in range(len(l)):
        if 'init: N(total)' in l[i]:
            N = int(float(l[i].split()[4]))
            pos = np.zeros((N, 3))
            sym = np.empty(N, dtype=str)
        elif 'init: cell(:' in l[i]:
            for j in range(3):
                cell[j] = l[i + j].split()[-3:]
        elif 'init: atom/type/pos/fix:' in l[i]:
            j = int(float(l[i].split()[2]))
            sym[j - 1] = l[i].split()[3]
            pos[j - 1] = [float(x) for x in l[i].split()[4:7]]
    atom = Atoms(cell=cell, positions=pos, symbols=sym, pbc=True)
    return atom


def get_bonds(filename='log.bx'):
    bonds = []
    dists = []
    neighbors = []

    finish = False
    i_atom = 1

    while (finish == False):
        new_bonds, new_dists, new_neighbors = get_bonds_i(filename, i_atom)
        if (new_bonds != None):
            bonds.append(new_bonds)
            dists.append(new_dists)
            neighbors.append(new_neighbors)
            i_atom = i_atom + 1
        else:
            finish = True

    return bonds, dists, neighbors


def get_bonds_i(filename, i_atom):
    key_word = '<- %10d)' % i_atom

    if filename.split('.')[-1] == 'gz':
        import gzip
        # more robust way of checking zipped file?
        log = gzip.open(filename, 'r')
    else:
        log = open(filename, 'r')
    l = log.readlines()
    log.close()

    bonds = []
    dists = []
    neighbors = []

    key_word_not_found = True

    try:
        for line in l:
            if key_word in line:
                key_word_not_found = False
                bond_arr = np.array(line.split()[4:], dtype=float)
                bond = bond_arr[:3]
                dist = bond_arr[3]
                neighbor = int(line.split()[1])
                if (dist > 0):
                    bonds.append(bond)
                    dists.append(dist)
                    neighbors.append(neighbor)
        if (key_word_not_found == True):
            bonds = None
            dists = dists
            neighbors = neighbors
        return bonds, dists, neighbors
    except:
        return None, None, None
