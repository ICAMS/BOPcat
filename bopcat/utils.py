#!/usr/bin/env python
import itertools

import numpy as np
from ase import Atoms
from ase.dft.kpoints import ibz_points

"""
BOPcat utilities.

Part of the BOPcat package.

author: Alvin Noe Ladines
e-mail: ladinesalvinnoe@gmail.com
"""


###########################################################################
def gen_par_mod(key, constraint, param=None, atomsbx=None, bondsbx=None, infox=None):
    out = []
    count = 0
    if atomsbx is not None:
        if key not in list(atomsbx.get_atomspar().keys()):
            raise ValueError("Unrecognized key: %s" % key)
        par = atomsbx.get_atomspar()[key]
        if par is None:
            raise ValueError("No key %s in atomsbx." % key)
            if param is None:
                return out
            else:
                return atomsbx
        if len(par) != len(constraint):
            raise ValueError("Number of {} constraint is {:d} but needs to be {} (number of parameters).".\
                             format(key, len(constraint), len(par)))
        if param is None:
            for i in range(len(par)):
                if constraint[i]:
                    out.append(par[i])
        else:
            for i in range(len(par)):
                if constraint[i]:
                    par[i] = param[count]
                    count += 1
            atomsbx.set_atomspar({key: par})
    elif bondsbx is not None:
        if key in list(bondsbx.get_bondspar().keys()):
            par = bondsbx.get_bondspar()
        elif key in list(bondsbx.get_overlappar().keys()):
            par = bondsbx.get_overlappar()
        elif key in list(bondsbx.get_repetal().keys()):
            par = bondsbx.get_repetal()
        elif key in list(bondsbx.get_cutpar().keys()):
            par = bondsbx.get_cutpar()
        else:
            raise ValueError("Unrecognized key: %s" % key)
        par = par[key]
        func = None
        if par is None:
            raise ValueError("No key %s in bondsbx." % key)
            if param is None:
                return out
            else:
                return bondsbx
        if not isinstance(par, list):
            # expecting here a function
            func = par
            par = par.get_numbers()
            func.set(constraints=list(constraint))
        if len(par) != len(constraint):
            raise ValueError("Number of {} constraint is {:d} but needs to be {} (number of parameters).".\
                             format(key, len(constraint), len(par)))
        if param is None:
            for i in range(len(par)):
                if constraint[i]:
                    out.append(par[i])
        else:
            for i in range(len(par)):
                if constraint[i]:
                    par[i] = param[count]
                    count += 1
            if func is not None:
                func.set_parameters(param)
                par = func
            bondsbx.set_bondspar({key: par})
    elif infox is not None:
        if key not in infox:
            raise ValueError("Unrecognized key: %s" % key)
        par = infox[key]
        if param is None:
            for i in range(len(par)):
                if constraint[i]:
                    out.append(par[i])
        else:
            for i in range(len(par)):
                if constraint[i]:
                    par[i] = param[count]
                    count += 1
            infox[key] = par
    if param is None:
        return out
    else:
        if atomsbx is not None:
            return atomsbx
        elif bondsbx is not None:
            return bondsbx
        elif infox is not None:
            return infox


def gen_param_model(model, variables, newparam=None):
    """
    Returns parameters that need to be optimized subject to the
    constraints specified in variables. 
    
    Generates new model by changing the parameters with those specified in
    newparam subject to given constraints.

    model is a modelsbx object defined in module bopmodel.
    variables is a list of dictionaries each containing the keywords and
    their corresponding constraint conditions corresponding to each bond/atom.
    
    Example:
    variables = [{'bond': ['Fe', 'Fe'], 'pairrepulsion' : [True True True]},
                {'bond':['Fe','Nb'],'atom':'Nb','ddsigma':[True,False,False],
                 'valenceelectrons':[True]}]
    
    The number of parameters for a certain keyword should much
    the number of free variables (True) in constraints.
    
    """
    if variables is None:
        variables = []
    # check if length of newparam matches contraints in variables
    count = 0
    for i in range(len(variables)):
        for key, val in list(variables[i].items()):
            if key not in ['atom', 'bond', 'fit']:
                count += val.count(True)
    if newparam is not None:
        if count != len(newparam):
            raise ValueError("Expecting %d parameters" % count)
    model = model.copy()
    bond_keys = model.bondsbx[0].get_keys()
    atom_keys = model.atomsbx[0].get_keys()
    infox_keys = list(model.infox_parameters.keys())
    bbx = model.bondsbx
    abx = model.atomsbx
    infox = model.infox_parameters
    out = []
    count = 0
    for i in range(len(variables)):
        for key, con in list(variables[i].items()):
            key = key.lower()
            if key in bond_keys:
                if 'bond' not in variables[i]:
                    raise ValueError('Indicate bond in variables')
                bond = variables[i]['bond']
                if isinstance(bond, str):
                    bond = bond.split('-')
                bond.sort()
                for j in range(len(bbx)):
                    bbxj_bond = bbx[j].get_bond()
                    bbxj_bond.sort()
                    if bond == bbxj_bond:
                        if newparam is None:
                            out += gen_par_mod(key, con, bondsbx=bbx[j])
                            break
                        else:
                            temp = newparam[count:count + con.count(True)]
                            bbx[j] = gen_par_mod(key, con, bondsbx=bbx[j], param=temp)
                            count += con.count(True)
            elif key in atom_keys:
                if 'atom' not in variables[i]:
                    raise ValueError('Indicate atom in variables')
                atom = variables[i]['atom']
                if isinstance(atom, str):
                    atom = [atom]
                for j in range(len(abx)):
                    abxj_atom = abx[j].get_atom()
                    if isinstance(abxj_atom, str):
                        abxj_atom = [abxj_atom]
                    if atom == abxj_atom:
                        if newparam is None:
                            out += gen_par_mod(key, con, atomsbx=abx[j])
                        else:
                            temp = newparam[count:count + con.count(True)]
                            abx[j] = gen_par_mod(key, con, atomsbx=abx[j], param=temp)
                            count += con.count(True)
            elif key in infox_keys:
                if newparam is None:
                    out += gen_par_mod(key, con, infox=infox)
                else:
                    temp = newparam[count:count + con.count(True)]
                    infox = gen_par_mod(key, con, infox=infox, param=temp)
                    count += con.count(True)
            elif key in ['atom', 'bond', 'fit']:
                pass
            else:
                raise ValueError('Unrecognized key %s' % key)
    if newparam is None:
        return out
    else:
        model.infox_parameters = infox
        model.atomsbx = abx
        model.bondsbx = bbx
        return model


def gen_penalty_coeffs_list(model, variables, penalty_coeffs):
    """
    Returns parameters that need to be optimized subject to the
    constraints specified in variables.

    Generates new model by changing the parameters with those specified in
    newparam subject to given constraints.

    model is a modelsbx object defined in module bopmodel.
    variables is a list of dictionaries each containing the keywords and
    their corresponding constraint conditions corresponding to each bond/atom.

    Example:
    variables = [{'bond': ['Fe', 'Fe'], 'pairrepulsion' : [True True True]},
                {'bond':['Fe','Nb'],'atom':'Nb','ddsigma':[True,False,False],
                 'valenceelectrons':[True]}]

    The number of parameters for a certain keyword should much
    the number of free variables (True) in constraints.

    """
    if variables is None:
        variables = []

    if penalty_coeffs is None:
        penalty_coeffs = []
        for i in range(len(variables)):
            if 'bond' not in variables[i]:
                raise ValueError('Indicate bond in variables')
            else:
                penalty_coeffs.append({'bond': variables[i]['bond']})
    else:
        if len(variables) != len(penalty_coeffs):
            raise ValueError('Penatly coefficients and variables have'
                             ' to be of same length')
    model = model.copy()
    bond_keys = model.bondsbx[0].get_keys()
    atom_keys = model.atomsbx[0].get_keys()
    infox_keys = list(model.infox_parameters.keys())
    bbx = model.bondsbx
    abx = model.atomsbx
    out = []
    for i in range(len(variables)):
        for key, con in list(variables[i].items()):
            if key in bond_keys:
                if 'bond' not in variables[i]:
                    raise ValueError('Indicate bond in variables')
                if 'bond' not in penalty_coeffs[i]:
                    raise ValueError('Indicate bond in penalty_coeffs')
                if variables[i]['bond'] != penalty_coeffs[i]['bond']:
                    raise ValueError('Penatly coefficients and variables '
                                     'have to be of same order')
                bond = variables[i]['bond']
                if isinstance(bond, str):
                    bond = bond.split('-')
                bond.sort()
                for j in range(len(bbx)):
                    bbxj_bond = bbx[j].get_bond()
                    bbxj_bond.sort()
                    if bond == bbxj_bond:
                        coeffs = []
                        for k in range(len(variables[i][key])):
                            if not penalty_coeffs is None:
                                if variables[i][key][k] == True:
                                    if key in list(penalty_coeffs[i].keys()):
                                        if penalty_coeffs[i][key][k] == \
                                                None:
                                            new_coef = [0., 2., 1., None]
                                        else:
                                            new_coef = \
                                                penalty_coeffs[i][key][k]
                                    else:
                                        new_coef = [0., 2., 1., None]
                                    coeffs.append(new_coef)
                        out += coeffs
                        break
            elif key in atom_keys:
                if 'atom' not in variables[i]:
                    raise ValueError('Indicate atom in variables')
                atom = variables[i]['atom']
                if isinstance(atom, str):
                    atom = [atom]
                for j in range(len(abx)):
                    abxj_atom = abx[j].get_atom()
                    if isinstance(abxj_atom, str):
                        abxj_atom = [abxj_atom]
                    if atom == abxj_atom:
                        coeffs = []
                        for k in range(len(variables[i][key])):
                            if not penalty_coeffs is None:
                                if variables[i][key][k] == True:
                                    if key in list(penalty_coeffs[i].keys()):
                                        if penalty_coeffs[i][key][k] == \
                                                None:
                                            new_coef = [0., 2., 1., None]
                                        else:
                                            new_coef = \
                                                penalty_coeffs[i][key][k]
                                    else:
                                        new_coef = [0., 2., 1., None]
                                    coeffs.append(new_coef)
                        out += coeffs
            elif key in infox_keys:
                coeffs = []
                for k in range(len(variables[i][key])):
                    if variables[i][key][k] == True:
                        if not penalty_coeffs is None:
                            if key in list(penalty_coeffs[i].keys()):
                                if penalty_coeffs[i][key][k] == \
                                        None:
                                    new_coef = [0., 2., 1., None]
                                else:
                                    new_coef = \
                                        penalty_coeffs[i][key][k]
                            else:
                                new_coef = [0., 2., 1., None]
                            coeffs.append(new_coef)
                out += coeffs
            elif key in ['atom', 'bond', 'fit']:
                pass
            else:
                raise ValueError('Unrecognized key %s' % key)
    return out


def get_par_keys(var):
    if var is None:
        var = []
    par_keys = []
    atbo = ''
    for i in range(len(var)):
        if 'bond' in var[i]:
            val = var[i]['bond']
            if isinstance(val, str):
                val = val.split('-')
            atbo = '%s-%s' % (val[0], val[1])
        elif 'atom' in var[i]:
            val = var[i]['atom']
            if isinstance(val, str):
                val = [val]
            atbo = '%s' % val[0]
        for key, val in list(var[i].items()):
            if key.lower() in ['atom', 'bond', 'fit']:
                continue
            if True in val or False in val:
                for j in range(len(val)):
                    if val[j]:
                        par_keys.append('%s-%s-%d' % (atbo, key, j))
    return par_keys


def kptdensity2monkhorstpack(atoms, kptdensity=7, even=True):
    """Convert k-point density to Monkhorst-Pack grid size.

    a: Atoms object
        Contains unit cell and information about boundary conditions.
    kptdensity: float
        K-point density.  Default value is 3.5 point per Ang^-1.
    even: bool
        Round to even numbers.
    """
    # copied from ase.calculators
    recipcell = atoms.get_reciprocal_cell()
    kpts = []
    for i in range(3):
        if atoms.pbc[i]:
            k = 2 * np.pi * np.sqrt((recipcell[i] ** 2).sum()) * kptdensity
            if even:
                kpts.append(max(1, 2 * int(round(k / 2))))
            else:
                kpts.append(max(1, int(round(k))))
        else:
            kpts.append(1)
    return np.array(kpts)


def kpts2mp(atoms, kpts, even=False):
    # copied from ase.calculators
    if kpts is None:
        return np.array([1, 1, 1])
    if isinstance(kpts, (float, int)):
        return kptdensity2monkhorstpack(atoms, kpts, even)
    else:
        return kpts


def get_lattice_type(atom, symprec=1e-3):
    """
    Returns the lattice type of an input ASE atoms object.
    """
    import spglib
    sg = spglib.get_spacegroup(atom, symprec)
    number = int(sg.rstrip(')').split()[1].lstrip('('))
    lat_dic = {
        'triclinic': list(range(1, 3)),
        'monoclinic': list(range(3, 16)),
        'orthorhombic': list(range(16, 75)),
        'tetragonal': list(range(75, 143)),
        'trigonal': list(range(143, 168)),
        'hexagonal': list(range(168, 195)),
        'cubic': list(range(195, 231))
    }
    for t, r in list(lat_dic.items()):
        if number in r:
            return t, number


def get_high_symmetry_points(atom, k_points):
    # High-symmetry points in the Brillouin zone
    lattyp, sgnum = get_lattice_type(atom)
    # temporary patch for fcc and bcc
    if sgnum == 229:
        lattyp = 'bcc'
    elif sgnum == 225:
        lattyp = 'fcc'
    else:
        pass
    names = []
    path_k = []
    if len(k_points) == 0:
        # Get names of high sym points and paths connectin them
        for name, point in list(ibz_points[lattyp].items()):
            names.append(name)
            path_k.append(point)
    # ibz_points just list down the points and the path is drawn to connect
    # these points in order that they appear but what if you want to go 
    # along a different path?
    # for example: bcc: H P G N
    # but you may be interested also in H G H P G N
    elif isinstance(k_points[0], str):
        for n in k_points:
            for name, point in list(ibz_points[lattyp].items()):
                if name == n:
                    names.append(n)
                    path_k.append(point)
                    break
    else:  # expecting array of k_points
        path_k = list(k_points)
        names = None
        Nk = 1
    return path_k, names


def make_pairs(elements):
    bonds = []
    # comb = itertools.combinations_with_replacement(elements,2)
    comb = itertools.product(elements, repeat=2)
    temp = next(comb)
    while temp != "done":
        bonds.append(list(temp))
        try:
            temp = next(comb)
        except:
            temp = "done"
    return bonds


def do_volume(atom, model, vmin=0.9, vmax=1.1, step=0.02):
    """
    Perform calculations around the equilibrium  
    """
    from .calc_bopfox import initialize_bopfox
    atoms = []
    volfracs = np.arange(0.8, 1.21, step)
    done = []
    new = atom.copy()
    calc = initialize_bopfox(model.infox_parameters, modelsbx=model)
    new.set_calculator(calc)
    e0 = new.get_potential_energy() / len(new)
    count = 0
    for i in range(int(len(volfracs) / 2), len(volfracs)):
        new = atom.copy()
        c = atom.get_cell()
        c *= volfracs[i] ** (1 / 3.)
        new.set_cell(c, scale_atoms=True)
        calc = initialize_bopfox(model.infox_parameters, modelsbx=model)
        new.set_calculator(calc)
        ene = new.get_potential_energy() / len(new)
        count += 1
        if ene > e0:
            break
        else:
            e0 = ene
            atoms.append(new)
            done.append(volfracs[i])
    if count > 1:
        v0 = volfracs[i]
    else:
        count = 0
        for i in range(len(volfracs) / 2, -1, -1):
            new = atom.copy()
            c = atom.get_cell()
            c *= volfracs[i] ** (1 / 3.)
            new.set_cell(c, scale_atoms=True)
            calc = initialize_bopfox(model.infox_parameters, modelsbx=model)
            new.set_calculator(calc)
            ene = new.get_potential_energy() / len(new)
            count += 1
            if ene > e0:
                break
            else:
                e0 = ene
                atoms.append(new)
                done.append(volfracs[i])
        v0 = volfracs[i]
    volfracs = np.arange(v0 - (1 - vmin), v0 + (vmax - 1), step)
    atoms = []
    for i in range(len(volfracs)):
        if volfracs[i] in done:
            continue
        new = atom.copy()
        c = atom.get_cell()
        c *= volfracs[i] ** (1 / 3.)
        new.set_cell(c, scale_atoms=True)
        new.set_calculator(calc)
        ene = new.get_potential_energy() / len(new)
        atoms.append(new)
        done.append(volfracs[i])
    return atoms


def transform_tetra(atom, p):
    new = atom.copy()
    # ab = np.exp(np.log(atom.get_volume()/p)/3.)
    # c = atom.get_volume()/ab**2.
    # new.set_cell([ab,ab,c],scale_atoms=True)
    a0 = abs(atom.get_cell()[0][0]) * 2.
    c = np.array([[a0, 0, 0], [0, a0, 0], [0, 0, p * a0]])
    new.set_cell(c, scale_atoms=True)
    c *= (atom.get_volume() / new.get_volume()) ** (1 / 3.)
    new.set_cell(c, scale_atoms=True)
    return new


def transform_trigo(atom, p):
    new = atom.copy()
    a0 = abs(atom.get_cell()[0][0]) * 2.
    a1 = a0 * np.array([1. / 6. * p - 2. / 3., 1. / 6. * p + 1. / 3., 1. / 6. * p + 1. / 3.])
    a2 = a0 * np.array([1. / 6. * p + 1. / 3., 1. / 6. * p - 2. / 3., 1. / 6. * p + 1. / 3.])
    a3 = a0 * np.array([1. / 6. * p + 1. / 3., 1. / 6. * p + 1. / 3., 1. / 6. * p - 2. / 3.])
    c = np.array([a1, a2, a3])
    new.set_cell(c, scale_atoms=True)
    c *= ((atom.get_volume() / new.get_volume()) ** (1 / 3.))
    new.set_cell(c, scale_atoms=True)
    return new


def transform_hex(atom, p):
    new = atom.copy()
    brac1 = p * (2 * np.sqrt(3) - 3 * np.sqrt(2)) / 6. + np.sqrt(2) / 2.
    brac2 = p * (2 * np.sqrt(2) - 3.) / 3. + 1
    vv0 = np.sqrt(2) * brac1 * brac2
    a0 = abs(atom.get_cell()[0][0]) * 2.
    a = a0 * np.sqrt(2) / (vv0 ** (1. / 3.))
    b = a * brac1
    c = a * brac2
    pos1 = (0.5 - p / 6., 0, 0.5)
    pos2 = (0, 0, 0)
    pos3 = (0.5, 0.5, 0)
    pos4 = (- p / 6., 0.5, 0.5)
    a1 = a * np.array([1, 0, 0])
    a2 = b * np.array([0, 1, 0])
    a3 = c * np.array([0, 0, 1])
    cell = np.array([a1, a2, a3])
    new = Atoms(atom.get_chemical_symbols() * 4, scaled_positions=[pos1, pos2
        , pos3, pos4], cell=cell)
    cell *= ((atom.get_volume() * 4 / new.get_volume()) ** (1 / 3.))
    new.set_cell(cell, scale_atoms=True)
    new.set_initial_magnetic_moments( \
        list(atom.get_initial_magnetic_moments()) * len(new))
    return new


def transform_ortho(atom, p):
    a0 = abs(atom.get_cell()[0][0]) * 2.
    a1 = a0 * np.array([-1. / 2., 1. / 2., p / 2.])
    a2 = a0 * np.array([1. / 2., -1. / 2., p / 2.])
    a3 = a0 * np.array([1. / (2. * p), 1. / (2. * p), -p / 2.])
    cell = np.array([a1, a2, a3])
    new = atom.copy()
    new.set_cell(cell, scale_atoms=True)
    # cell *= (atom.get_volume()/new.get_volume())**(1/3.)
    # new.set_cell(cell,scale_atoms=True)
    return new


def do_transformation(atom, model, ttype='tetragonal', amin=0.8, amax=2.00
                      , step=0.01):
    """
    Perform deformation along special transformation paths
    """
    from .calc_bopfox import initialize_bopfox
    atoms = []
    p = np.arange(amin, amax, step)
    for i in range(len(p)):
        if 'tetra' in ttype.lower():
            new = transform_tetra(atom, p[i])
        elif 'trigo' in ttype.lower():
            new = transform_trigo(atom, p[i])
        elif 'hex' in ttype.lower():
            new = transform_hex(atom, p[i])
            # import os
            # from ase.io import bopfox
            # cwd = os.getcwd()
            # os.chdir('hex')
            # new = bopfox.read_strucbx('bopfox_hex_%d'%i)
            # new.set_initial_magnetic_moments([2.5]*len(new))
            # os.chdir(cwd)
        elif 'ortho' in ttype.lower():
            new = transform_ortho(atom, p[i])
            # import os
            # from ase.io import bopfox
            # cwd = os.getcwd()
            # os.chdir('ortho')
            # dirs = os.listdir(os.getcwd())
            # new = bopfox.read_strucbx(dirs[i])
            # new.set_initial_magnetic_moments([2.5]*len(new))
            # os.chdir(cwd)
        else:
            return NotImplementedError
        calc = initialize_bopfox(model.infox_parameters, modelsbx=model)
        new.set_calculator(calc)
        ene = new.get_potential_energy()
        atoms.append(new)
    return atoms


def neigh_dist(atom, origin=0, N=3):
    """
    Returns the distance from origin up to Nth neearest neighbor 
    """
    eps = 1E-5
    atom = atom.copy()
    c = atom.get_cell()
    atom.translate(-atom.get_positions()[origin])
    cluster = atom.repeat(N + 1)
    t = (c[0] + c[1] + c[2]) * (N + 1) / 2
    cluster.translate(-t)
    for i in range(len(cluster)):
        if np.linalg.norm(cluster.get_positions()[i]) < eps:
            cp = i
            break
    out = []
    for i in range(len(cluster)):
        d = cluster.get_distance(i, cp)
        d = round(d, 4)
        if d not in out:
            out.append(d)
    out.sort()
    out = out[1:N + 1]
    return out


def error_vector(ref, model, weights=None):
    """
    This function returns now only the error as an array. This means 
    that the values 'array' and 'mode' will not be used in the future.
    """

    assert (len(ref) == len(model))
    if weights is None:
        weights = np.ones(len(ref))
    out = []
    for i in range(len(ref)):
        diff = (model[i] - ref[i]) * weights[i]
        if isinstance(diff, float):
            out.append(diff)
        else:
            # diff = np.average(diff)
            diff = np.array(diff)
            diff = diff.reshape(np.size(diff))
            out += list(diff)
    out = np.array(out)
    out = out.reshape(np.size(out))

    # The lines below have been commented out for version 27

    # if not array:
    #    if mode=='rms':
    #        out = np.average(out**2)**0.5
    #    elif mode=='max':
    #       out = max(abs(out))
    #   elif mode=='max_square':
    #        out = max(abs(out))**2
    #    else:
    #        raise NotImplementedError('No options for %s'%mode)
    return out


def penalty_term(x0, p0, pc, array=True):
    """
    x0: [1, 1, 1, ...] at the beginning of the optimization
    p0: initial numbers of parameters which are optimized
    """
    assert (len(x0) == len(pc))
    penalty = []
    for i in range(len(x0)):
        if pc[i][3] == None:
            ref_val = 1.
        else:
            ref_val = pc[i][3] / p0[i]
        penalty.append(pc[i][0] * (abs(x0[i] - ref_val) / pc[i][2]) ** pc[i][1])
    if not array:
        penalty = np.average(penalty ** 2) ** 0.5
    return penalty


def calc_diff(data):
    """
    Returns difference between data[:size/2] and data[size/2:]
    """
    # length of data should be even
    assert (len(data) % 2 == 0)
    out = []
    for i in range(int(len(data) / 2)):
        out.append(data[i] - data[i + int(len(data) / 2)])
    return out


def stringtolist(s, separator='/'):
    l = s.split(separator)
    for i in range(len(l)):
        t = l[i]
        try:
            t = int(t)
        except:
            try:
                t = float(t)
            except:
                if t in ['True']:
                    t = True
                elif t in ['False']:
                    t = False
                else:
                    pass
        l[i] = t
    return l


def listtostring(l, separator='/'):
    s = ''
    for i in l[:-1]:
        s += '%s%s' % (str(i), separator)
    s += str(l[-1])
    return s


if __name__ == "__main__":
    from .bopmodel import read_modelsbx

    model = read_modelsbx(filename='models.bx')[0]
    var = [{'bond': ['Fe', 'Fe'], 'ddsigma': [True, True, False] * 3
               , 'rep2': [True, True, False, False, False] * 2
               , 'atom': 'Fe'
               , 'valenceelectrons': [True]
            }, {'atom': 'Nb', 'onsitelevels': [True], 'valenceelectrons': [True]
               , 'bond': ['Nb', 'Nb'], 'eamattraction': [False, False, True]}]
    par = gen_param_model(model, var, newparam=None)
    new_par = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    new_model = gen_param_model(model, var, newparam=new_par)
    new_model.write('deleteme.bx')
