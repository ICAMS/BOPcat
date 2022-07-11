#!/usr/bin/env python
import numpy as np
from ase.calculators import bopfox as bopcal
import subprocess
from .utils import kptdensity2monkhorstpack
from .utils import get_high_symmetry_points
from ase.dft.kpoints import kpoint_convert, get_bandpath
from ase import Atoms
from .eigs import arrange_bands, get_relevant_orbs
from .variables import is_boplib

"""
BOPfox utilities.

Part of the BOPcat package.

"""


###########################################################################
def _initialize_bopfox(bopfox_para, atomsbx=None, bondsbx=None, modelsbx=None):
    """
    External function to initialize the BOPfox calculator. Use this instead 
    of initializing the bopcal.BOPfox object directly.
    """
    bopfox_para = bopfox_para.copy()
    if modelsbx is None:
        if (atomsbx is None) or (bondsbx is None):
            raise ValueError("Provide atomsbx and bondsbx.")
        bopfox_calc = bopcal.BOPfox(atomsbx=atomsbx, bondsbx=bondsbx)
    else:
        if 'atomsversion' in bopfox_para:
            bopfox_para.pop('atomsversion')
        if 'bondsversion' in bopfox_para:
            bopfox_para.pop('bondsversion')
        modelsbx = modelsbx.copy()
        bopfox_calc = bopcal.BOPfox(modelsbx=modelsbx)
    bopfox_calc.set(version='bop')
    for key, val in list(bopfox_para.items()):
        if isinstance(val, str):
            exec("bopfox_calc.set(%s='%s')" % (key.lower(), val))
        else:
            exec("bopfox_calc.set(%s=%s)" % (key.lower(), val))
    return bopfox_calc


def initialize_bopfox(bopfox_para, modelsbx, atoms):
    """
    External function to initialize the BOPfox calculator. Use this instead 
    of initializing the bopcal.BOPfox object directly.
    """
    bopfox_para = bopfox_para.copy()
    if 'atomsversion' in bopfox_para:
        bopfox_para.pop('atomsversion')
    if 'bondsversion' in bopfox_para:
        bopfox_para.pop('bondsversion')
    modelsbx = modelsbx.copy()
    # bopfox_calc = bopcal.BOPfox(modelsbx=modelsbx,atoms=atoms,**bopfox_para)
    if 'tbkpointfolding' not in bopfox_para:
        try:
            kmesh = atoms.info['kmesh']
        except:
            kmesh = None
        if kmesh is None:
            kmesh = kptdensity2monkhorstpack(atoms)
        kprod = kmesh[0] * kmesh[1] * kmesh[2]
        if kprod * len(atoms) < 100:
            print(('Warning: Running TB calculation with %d kpoints' % kprod))
        bopfox_para['tbkpointfolding'] = kmesh
    magconfig = 'nm'
    if atoms.has('magmoms') or atoms.has('initial_magmoms'):
        magconfig = 'cm'
    task = 'energy'
    if 'required_property' in atoms.info:
        task = atoms.info['required_property']
        if task in ['energy', 'energies']:
            task = 'energy'
        if task in ['force', 'forces', 'stress', 'stresses']:
            task = 'force'
    bopfox_para['magconfig'] = magconfig
    bopfox_para['task'] = task
    bopfox_calc = bopcal.BOPfox(modelsbx=modelsbx, atoms=atoms
                                , **bopfox_para)
    atoms.set_calculator(bopfox_calc)
    # necessary for library
    if is_boplib():
        bopfox_calc.calculate(atoms, [task], ['modelsbx'])
    # bopfox_calc.set(version='bop')
    # for key, val in bopfox_para.items():
    #    if isinstance(val,str):
    #        exec("bopfox_calc.set(%s='%s')"%(key.lower(),val))
    #    else:
    #        exec("bopfox_calc.set(%s=%s)"%(key.lower(),val))
    return bopfox_calc


def add_initial_moments(atom, bopfox_para):
    bopfox_para['magconfig'] = 'cm'
    if np.count_nonzero(atom.get_initial_magnetic_moments()) > 0:
        return atom
    mom = []
    sym = atom.get_chemical_symbols()
    try:
        ini_mom = bopfox_para['ini_magmoms']
    except:
        # print "ini_magmoms not found in bopfox parameters."
        # print "setting magnetic moments to 2."
        ini_mom = {}
        for i in sym:
            ini_mom[i] = 2
    for i in sym:
        mom.append(ini_mom[i])
    atom.set_initial_magnetic_moments(mom)
    return atom


def calc_ebs_bopfox(atom, calc, k_points=[]
                    , cartesian=False, Nk=100, shift_Fermi=True):
    """
    Calculates the electronic bandstructure and density of states
    using BOPfox
    input k_points in cartesian should be in units of 2pi
    """
    atom = atom.copy()
    modelsbx = calc.modelsbx
    old_params = modelsbx.infox_parameters.copy()
    atomsbx = [abx.copy() for abx in modelsbx.atomsbx]
    bondsbx = [bbx.copy() for bbx in modelsbx.bondsbx]
    modelsbx.infox_parameters.update({'task': 'bandstructure'
                                         , 'scfsteps': 0, 'tbkpointmesh': 'path'
                                         , 'tbkpointfile': 'kpoints.dat', 'version': 'tight-binding'
                                         , 'repversion': 'None'})
    old = False
    # if old:
    #    cuts = ['rcut','dcut','r2cut','d2cut']
    #    for cut in cuts:
    #        maxcut = max([bbx.get_cutpar()[cut] for bbx in modelsbx.bondsbx])
    #        modelsbx.infox_parameters[cut] = maxcut
    #    if modelsbx.infox_parameters.has_key('model'):
    #        # if model is in infox it will activate new reading routine
    #        modelsbx.infox_parameters.pop('model')
    #    aversion = modelsbx.atomsbx[0].version
    #    bversion = modelsbx.bondsbx[0].version
    #    modelsbx.infox_parameters['atomsversion'] = aversion
    #    modelsbx.infox_parameters['bondsversion'] = bversion
    #    calc = initialize_bopfox(modelsbx.infox_parameters\
    #                                 ,atomsbx=atomsbx,bondsbx=bondsbx)
    # else:
    #    calc = initialize_bopfox(modelsbx.infox_parameters
    #                                 ,modelsbx=modelsbx)
    calc.set_modelsbx(modelsbx)
    atom = atom.copy()
    mag = 1
    if 'spin' in atom.info:
        mag = atom.info['spin']
    # if mag == 2:
    #    atom = add_initial_moments(atom,modelsbx.infox_parameters)
    atom.set_calculator(calc)
    if len(k_points) > 0:
        if not isinstance(k_points[0], str):
            Nk = 1
    path_k, names = get_high_symmetry_points(atom, k_points)
    # convert k_points if not in cartesian
    if not (cartesian):
        path_k = kpoint_convert(atom.get_cell(), skpts_kc=path_k) / (2 * np.pi)
    calc.set(kpoints=(path_k, Nk))
    ene = atom.get_potential_energy() / len(atom)
    eigenvalues = calc.get_eigenvalues()
    eigenvalues = eigenvalues.T
    eigenvalues = eigenvalues[4:]
    eigenvalues = eigenvalues.T
    if shift_Fermi:
        kmesh = kptdensity2monkhorstpack(atom)
        modelsbx.infox_parameters.update({'task': 'energy'
                                             , 'scfsteps': 500, 'tbkpointmesh': 'gamma-centered'
                                             , 'tbkpointfile': None, 'version': 'tight-binding'
                                             , 'tbkpointfolding': list(kmesh)})
        if old:
            cuts = ['rcut', 'dcut', 'r2cut', 'd2cut']
            for cut in cuts:
                maxcut = max([bbx.get_cutpar()[cut] for bbx in modelsbx.bondsbx])
                modelsbx.infox_parameters[cut] = maxcut
            if 'model' in modelsbx.infox_parameters:
                # if model is in infox it will activate new reading routine
                modelsbx.infox_parameters.pop('model')
            aversion = modelsbx.atomsbx[0].version
            bversion = modelsbx.bondsbx[0].version
            modelsbx.infox_parameters['atomsversion'] = aversion
            modelsbx.infox_parameters['bondsversion'] = bversion
            calc = initialize_bopfox(modelsbx.infox_parameters \
                                     , atomsbx=atomsbx, bondsbx=bondsbx)
        else:
            calc = initialize_bopfox(modelsbx.infox_parameters
                                     , modelsbx=modelsbx)
        atom.set_calculator(calc)
        ene = atom.get_potential_energy() / len(atom)
        eigenvalues -= calc.get_fermi_energy()
    modelsbx.infox_parameters = dict(old_params)
    eigenvalues = eigenvalues.reshape((mag, int(len(eigenvalues) / mag)
                                       , len(eigenvalues[0])))
    if not shift_Fermi:

        # arrange eigenvalues as s p-1 p0 p1 d-2 d-1 d0 d1 d2
        orb_char = calc.get_orbital_character()
        # in bopfox, orb char is arranged as s p0 p1 p-1 d0 d1 d-1 d2 d-2
        for i in range(len(orb_char)):
            for j in range(len(orb_char[i])):
                orb_char[i][j] = [orb_char[i][j][0], orb_char[i][j][3]
                    , orb_char[i][j][1], orb_char[i][j][2]
                    , orb_char[i][j][8], orb_char[i][j][6]
                    , orb_char[i][j][4], orb_char[i][j][5]
                    , orb_char[i][j][7]]
        if mag == 2:
            temp1 = []
            temp2 = []
            for i in range(len(orb_char)):
                if i % 2 == 0:
                    temp1.append(orb_char[i])
                else:
                    temp2.append(orb_char[i])
            orb_char = np.array(temp1 + temp2)
        orb_char = orb_char.reshape((mag, int(len(orb_char) / mag)
                                     , len(orb_char[0]), 9))
        tempo = np.zeros(np.shape(eigenvalues))
        required = get_relevant_orbs(atom, modelsbx)
        done = False
        for s in range(len(eigenvalues)):
            for k in range(len(eigenvalues[s])):
                eig, oc = arrange_bands(eigenvalues[s][k], orb_char[s][k]
                                        , required=required)
                # for b in range(len(eigenvalues[s][k])):
                #  if not done:
                #    print eigenvalues[s][k][b]
                # for b in range(len(eigenvalues[s][k])):
                #  if not done:
                #        print eig[b]
                # done = True
                eigenvalues[s][k] = eig
                tempo[s][k] = oc
        orb_char = np.array(tempo)
        atom.info['orbital_character'] = orb_char
    atom.info['eigenvalues'] = eigenvalues
    return atom


def nullify(modelsbx, keys):
    modelsbx = modelsbx.copy()
    bbx = modelsbx.bondsbx
    for i in range(len(bbx)):
        for key in keys:
            if key in list(bbx[i].get_bondspar().keys()):
                bbx[i].set_bondspar({key: None})
            elif key in list(bbx[i].get_overlappar().keys()):
                bbx[i].set_bondspar({key: None})
            elif key in list(bbx[i].get_repetal().keys()):
                bbx[i].set_bondspar({key: None})
            else:
                raise ValueError('No keyword %s in modelsbx.')
    modelsbx.bondsbx = bbx
    return modelsbx


def calc_efs_bopfox(atom, calc, required_property='energy'
                    , contribution='binding', atom_index='total'):
    """
    Calculates energy in bopfox.
    required_property can be energy, force, stress
    contibution can be bond, prom, exchange, ionic, coulomb, 
    rep, env, pair, binding
    """
    atom = atom.copy()
    modelsbx = calc.modelsbx
    old_params = modelsbx.infox_parameters.copy()
    # atomsbx = [abx.copy() for abx in modelsbx.atomsbx]
    # bondsbx = [bbx.copy() for bbx in modelsbx.bondsbx]
    if isinstance(required_property, str):
        required_property = [required_property]
    for i in range(len(required_property)):
        if required_property[i] in ['force', 'forces']:
            modelsbx.infox_parameters.update({'task': 'force'})
        if required_property[i] in ['stress', 'stresses']:
            modelsbx.infox_parameters.update({'task': 'force', 'printstress': True})
    if contribution in ['bond', 'electronic']:
        modelsbx.infox_parameters.update({'repversion': 'None'
                                             , 'bondsversion': modelsbx.bondsbx[0].version})
        modelsbx = nullify(modelsbx, ['rep%d' % i for i in range(1, 21)] +
                           ['pairrepulsion', 'eamattraction'])

    elif contribution in ['rep', 'empirical']:
        # repversion = modelsbx.bondsbx[0].version
        # if repversion is None:
        repversion = 'v0'  # not relevant but should be set initially
        modelsbx.infox_parameters.update({'bondsversion': 'None'
                                             , 'repversion': repversion})
        bondkeys = ['sssigma', 'spsigma', 'pssigma', 'sdsigma', 'dssigma', 'ppsigma'
            , 'pppi', 'pdsigma', 'dpsigma', 'pdpi', 'dppi', 'ddsigma', 'ddpi'
            , 'dddelta']
        overlapkeys = [b + 'overlap' for b in bondkeys]
        modelsbx = nullify(modelsbx, bondkeys + overlapkeys)

    # elif contribution == 'exchange':
    #    mag = atom.info['spin']
    #    if mag != 2:
    #        print 'Attempting to calculate exchange energy '
    #        print 'for a non-magnetic structure.'
    #        atom = add_initial_moments(atom,modelsbx.infox_parameters)
    # mag = 1
    # if atom.info.has_key('spin'):
    #    mag = atom.info['spin']
    # if mag == 2:
    #    atom = add_initial_moments(atom,modelsbx.infox_parameters)
    calc.set_modelsbx(modelsbx)
    # calc = initialize_bopfox(modelsbx.infox_parameters
    #                                 ,modelsbx=modelsbx)
    if calc.get_mode().lower() == 'tight-binding':
        if calc.lis_params['tbkpointfolding'] is None:
            try:
                kmesh = atom.info['kmesh']
            except:
                kmesh = None
            if kmesh is None:
                kmesh = kptdensity2monkhorstpack(atom)
            kprod = kmesh[0] * kmesh[1] * kmesh[2]
            if kprod * len(atom) < 100:
                print(('Warning: Running TB calculation with %d kpoints' % kprod))
            calc.set(tbkpointfolding=kmesh)
    # calc.set(task=required_property)
    atom.set_calculator(calc)
    atom.get_potential_energy()

    Nrepmax = 20
    rep_keys = ['rep%d' % i for i in range(1, Nrepmax + 1)]
    empirical_keys = ['env', 'pair', 'rep'] + rep_keys
    electronic_keys = ['analytic', 'bond', 'prom', 'exchange', 'ionic', 'coulomb']

    for k in range(len(required_property)):
        if atom_index == 'total':
            # sum of atoms
            at_i = [0]
        elif atom_index == 'all':
            # return for all atoms
            at_i = list(range(1, len(atom) + 1))
        elif isinstance(atom_index, int):
            at_i = [atom_index + 1]
        elif atom_index is None:
            if required_property[k] in ['energy', 'stress']:
                at_i = [0]
            elif required_property[k] in ['forces', 'stresses']:
                at_i = list(range(1, len(atom) + 1))
        else:
            # expecting list of atom indices
            at_i = [atom_index[i] + 1 for i in range(len(atom_index))]

        if required_property[k] in ['force', 'forces']:
            out = np.zeros((len(at_i), 3))
            for i in range(len(at_i)):
                if contribution in ['binding', 'total']:
                    out[i] = calc.get_contributions_forces('total')[at_i[i]]
                elif contribution == 'electronic':
                    for contrib in electronic_keys:
                        try:
                            out[i] += calc.get_contributions_forces(contrib) \
                                [at_i[i]]
                        except:
                            pass
                elif contribution == 'empirical':
                    for contrib in empirical_keys:
                        try:
                            out[i] += calc.get_contributions_forces(contrib) \
                                [at_i[i]]
                        except:
                            pass
                else:
                    out[i] = calc.get_contributions_forces(contribution)[at_i[i]]

            if required_property[k] == 'force':
                out = np.sum(out, axis=0)

        elif required_property[k] in ['energies', 'energy']:
            out = np.zeros(len(at_i))
            for i in range(len(at_i)):
                if contribution in ['binding', 'total']:
                    out[i] = calc.get_contributions_energy('binding')[at_i[i]]
                elif contribution == 'electronic':
                    for contrib in electronic_keys:
                        try:
                            out[i] += calc.get_contributions_energy(contrib) \
                                [at_i[i]]
                        except:
                            pass
                elif contribution == 'empirical':
                    for contrib in empirical_keys:
                        try:
                            out[i] += calc.get_contributions_energy(contrib) \
                                [at_i[i]]
                        except:
                            pass
                else:
                    out[i] = calc.get_contributions_energy(contribution)[at_i[i]]
            # return scalar for total energy
            if required_property[k] == 'energy':
                out = np.sum(out)

        elif required_property[k] in ['stress', 'stresses']:
            out = np.zeros((len(at_i), 6))
            for i in range(len(at_i)):
                if contribution in ['binding', 'total']:
                    out[i] = calc.get_contributions_stresses('total')[at_i[i]]
                elif contribution == 'electronic':
                    for contrib in electronic_keys:
                        try:
                            out[i] += calc.get_contributions_stresses(contrib) \
                                [at_i[i]]
                        except:
                            pass
                elif contribution == 'empirical':
                    for contrib in empirical_keys:
                        try:
                            out[i] += calc.get_contributions_stresses(contrib) \
                                [at_i[i]]
                        except:
                            pass
                else:
                    out[i] = calc.get_contributions_stresses(contribution)[at_i[i]]
            # normalize for total stress
            if required_property[k] == 'stress':
                out = np.sum(out, axis=0) / atom.get_volume()
        else:
            raise NotImplementedError('Cannot calculate %s' % required_property[k])
        atom.info[required_property[k]] = out

    modelsbx.infox_parameters = dict(old_params)
    # update onsite
    if calc.get_onsite_levels() is not None:
        atom.info['onsites'] = calc.get_onsite_levels()
    return atom


def get_phonopy_version():
    from phonopy import version as phonopy_version
    try:
        vers = phonopy_version.__version__
    except:
        try:
            vers = phonopy_version.phonopy_version
        except:
            vers = None
    return vers


def run_phon(atom, modelsbx, path_kc, cell_size):
    """ Uses the ASE-phonon code to calculate the phonons"""
    from ase.phonons import Phonons
    h = 4.135667516e-15
    old_params = modelsbx.infox_parameters.copy()
    atomsbx = [abx.copy() for abx in modelsbx.atomsbx]
    bondsbx = [bbx.copy() for bbx in modelsbx.bondsbx]
    modelsbx.infox_parameters.update({'tbnsmear': 1
                                         , 'tbdsmear': 0.1, 'tbintegration': 'methfessel-paxton'
                                      })
    old = False
    if old:
        cuts = ['rcut', 'dcut', 'r2cut', 'd2cut']
        for cut in cuts:
            maxcut = max([bbx.get_cutpar()[cut] for bbx in modelsbx.bondsbx])
            modelsbx.infox_parameters[cut] = maxcut
        if 'model' in modelsbx.infox_parameters:
            # if model is in infox it will activate new reading routine
            modelsbx.infox_parameters.pop('model')
        aversion = modelsbx.atomsbx[0].version
        bversion = modelsbx.bondsbx[0].version
        modelsbx.infox_parameters['atomsversion'] = aversion
        modelsbx.infox_parameters['bondsversion'] = bversion
        calc = initialize_bopfox(modelsbx.infox_parameters \
                                 , atomsbx=atomsbx, bondsbx=bondsbx)
    else:
        calc = initialize_bopfox(modelsbx.infox_parameters
                                 , modelsbx=modelsbx)
    calc.set(task='force')
    if calc.get_mode() == 'tight-binding':
        kmesh = kptdensity2monkhorstpack(atom)
        calc.set(tbkpointfolding=kmesh)
    atom.set_calculator(calc)
    ph = Phonons(atom, calc, supercell=cell_size)
    ph.run()
    # Read forces and assemble the dynamical matrix
    ph.read(acoustic=True)
    # Band structure in meV
    omega_kn = ph.band_structure(path_kc)
    omega_kn = omega_kn / h
    # Calculate phonon DOS
    # set here as twice the regular k-mesh in calculating DOS
    kmesh = np.array(kptdensity2monkhorstpack(atom)) * 2
    omega_e, dos_e = ph.dos(kpts=kmesh)  # , npts=1000, delta=5e-4)
    omega_e = omega_e / h
    omega_kn /= 1e12
    omega_e /= 1e12
    atom.info['phonon_bs_omega'] = omega_kn
    atom.info['phonon_dos_omega'] = omega_e
    atom.info['phonon_dos_dos'] = dos_e
    modelsbx.infox_parameters = dict(old_params)
    # clean up
    subprocess.call('rm phonon*.pckl', shell=True)
    return atom


def run_phonopy_backup(atom, modelsbx, path_kc, cell_size):
    # B A C K U P  for 1.9.0.1

    """ Runs the phonopy python module to perfom a phonon calculation.
    
    Path kc denotes 
    """

    from phonopy import Phonopy
    from phonopy import version as phonopy_version
    from phonopy.structure.atoms import Atoms as PhonopyAtoms

    if phonopy_version.phonopy_version != '1.9.0.1':
        print('Warning: BOPcat compatible only with phonopy version 1.9.0.1')
        print('         Will try to run but errors may occur.')
    old_params = modelsbx.infox_parameters.copy()
    atomsbx = [abx.copy() for abx in modelsbx.atomsbx]
    bondsbx = [bbx.copy() for bbx in modelsbx.bondsbx]
    # modelsbx.infox_parameters.update({'tbnsmear':1
    #    ,'tbdsmear':0.1,'tbintegration':'methfessel-paxton'
    #    })
    old = False
    if old:
        cuts = ['rcut', 'dcut', 'r2cut', 'd2cut']
        for cut in cuts:
            maxcut = max([bbx.get_cutpar()[cut] for bbx in modelsbx.bondsbx])
            modelsbx.infox_parameters[cut] = maxcut
        if 'model' in modelsbx.infox_parameters:
            # if model is in infox it will activate new reading routine
            modelsbx.infox_parameters.pop('model')
        aversion = modelsbx.atomsbx[0].version
        bversion = modelsbx.bondsbx[0].version
        modelsbx.infox_parameters['atomsversion'] = aversion
        modelsbx.infox_parameters['bondsversion'] = bversion
    else:
        calc = initialize_bopfox(modelsbx.infox_parameters
                                 , modelsbx=modelsbx)
    # mag = 1
    # if atom.info.has_key('spin'):
    #    mag = atom.info['spin']
    bulk = PhonopyAtoms(symbols=atom.get_chemical_symbols(),
                        scaled_positions=atom.get_scaled_positions(),
                        cell=atom.get_cell())
    phonon = Phonopy(bulk, [[cell_size[0], 0, 0], [0, cell_size[1], 0]
        , [0, 0, cell_size[2]]])
    # print phonon
    # phonon.generate_displacements()
    supercells = phonon.get_supercells_with_displacements()
    force_sets = []
    for sp in supercells:
        sp_atom = Atoms(symbols=sp.get_chemical_symbols()
                        , positions=sp.get_positions(), cell=sp.get_cell()
                        , pbc=True)
        if old:
            calc = initialize_bopfox(modelsbx.infox_parameters \
                                     , atomsbx=atomsbx, bondsbx=bondsbx)
        else:
            calc = initialize_bopfox(modelsbx.infox_parameters
                                     , modelsbx=modelsbx)
        calc.set(task='force')
        # if mag == 2:
        #    add_initial_moments(sp_atom,modelsbx.infox_parameters)
        if calc.get_mode() == 'tight-binding':
            kmesh = kptdensity2monkhorstpack(sp_atom)
            calc.set(tbkpointfolding=kmesh)
        sp_atom.set_calculator(calc)
        ene = sp_atom.get_potential_energy()
        force_sets.append(sp_atom.get_forces())
    phonon.set_forces(force_sets)
    phonon.produce_force_constants()
    phonon.set_band_structure([path_kc])
    qpoints, dist, omega_kn, eigenvectors = phonon.get_band_structure()
    omega_kn = np.array(omega_kn)[0]
    # Calculate phonon DOS
    # set here as twice the regular k-mesh in calculating DOS
    kmesh = np.array(kptdensity2monkhorstpack(atom)) * 2
    phonon.set_mesh(kmesh)
    phonon.set_total_DOS()
    omega_e, dos_e = phonon.get_total_DOS()
    atom.info['phonon_bs_omega'] = omega_kn
    atom.info['phonon_dos_omega'] = omega_e
    atom.info['phonon_dos_dos'] = dos_e
    modelsbx.infox_parameters = dict(old_params)
    return atom


def calc_phonon_bopfox(atom, modelsbx, cell_size=(5, 5, 5), show_plot=False
                       , k_names=[], k_points=[], solver='phon'):
    """
    Calculates the phonon bandstructure and density of states
    using PHON or PHONOPY.
    """
    print(('Calculating phonon band structure with %s' % solver))
    # mag = 1
    # if atom.info.has_key('spin'):
    #    mag = atom.info['spin']
    # if mag == 2:
    #    atom = add_initial_moments(atom,modelsbx.infox_parameters)
    path_k, names = get_high_symmetry_points(atom, k_points)
    if names is None:
        names = k_names
    path_kc, q, Q = get_bandpath(path_k, atom.cell, 100 * len(path_k))
    atom.info['phonon_bs_path'] = path_kc
    atom.info['phonon_bs_names'] = names
    if solver == 'phon':
        atom = run_phon(atom, modelsbx, path_kc, cell_size)
    elif solver == 'phonopy':
        atom = run_phonopy(atom, modelsbx, path_kc, cell_size)
    else:
        raise Exception('Solver %s not yet implemented.' % solver)
    omega_kn = atom.info['phonon_bs_omega']
    omega_e = atom.info['phonon_dos_omega']
    dos_e = atom.info['phonon_dos_dos']
    # Write output to file
    f = open('phonon_bs.dat', 'w')
    for i in range(len(omega_kn)):
        if q[i] in Q:
            f.write('# %s\n' % names[list(Q).index(q[i])])
        f.write('%5.5f ' % q[i])
        for j in range(len(omega_kn[i])):
            f.write(' %1.5e ' % omega_kn[i][j])
        f.write('\n')
    f.close()
    f = open('phonon_dos.dat', 'w')
    for i in range(len(omega_e)):
        f.write('%5.5f  %5.5f\n' % (omega_e[i], dos_e[i]))
    if show_plot:
        # Plot the band structure and DOS
        # import pylab as pl
        import matplotlib.pyplot as pl
        pl.figure(1, (8, 6))
        pl.axes([.1, .07, .67, .85])
        for n in range(len(omega_kn[0])):
            omega_n = omega_kn[:, n]
            pl.plot(q, omega_n, 'k-', lw=2)
        for i in range(len(names)):
            if names[i] == 'Gamma':
                names[i] = r'$\%s$' % names[i]
            else:
                names[i] = r'$%s$' % names[i]
        pl.xticks(Q, names, fontsize=18)
        pl.yticks(fontsize=18)
        pl.xlim(q[0], q[-1])
        pl.ylim(0, np.max(omega_kn))
        pl.ylabel("Frequency (THz)", fontsize=22)
        pl.grid('on')

        pl.axes([.8, .07, .17, .85])
        pl.fill_between(dos_e, omega_e, y2=0, color='lightgrey'
                        , edgecolor='k', lw=1)
        # pl.ylim(0, 35)
        pl.ylim(0, np.max(omega_kn))
        pl.xticks([], [])
        pl.yticks([], [])
        pl.xlabel("DOS", fontsize=18)
        pl.savefig('phon_bs_dos.png')
        print('Figure dumped in phon_bs_dos.png')
    # pl.show()
    BS = np.vstack((q, omega_kn.T))
    DOS = np.vstack((omega_e, dos_e))
    return BS, DOS


######################## EXPERIMENTAL SECTION
def run_phonopy(atom, modelsbx, path_kc, cell_size):
    """ Runs the phonopy python module to perfom a phonon calculation."""

    from phonopy import Phonopy
    from phonopy import version as phonopy_version
    from phonopy.structure.atoms import Atoms as PhonopyAtoms
    if get_phonopy_version() != '1.11.6':
        print('''Warning: current BOPcat compatible only with 
                 phonopy version "1.11.6"'
                 Will try to run but errors may occur.''')
    old_params = modelsbx.infox_parameters.copy()
    atomsbx = [abx.copy() for abx in modelsbx.atomsbx]
    bondsbx = [bbx.copy() for bbx in modelsbx.bondsbx]
    # modelsbx.infox_parameters.update({'tbnsmear':1
    #    ,'tbdsmear':0.1,'tbintegration':'methfessel-paxton'
    #    })
    old = False
    if old:
        cuts = ['rcut', 'dcut', 'r2cut', 'd2cut']
        for cut in cuts:
            maxcut = max([bbx.get_cutpar()[cut] for bbx in modelsbx.bondsbx])
            modelsbx.infox_parameters[cut] = maxcut
        if 'model' in modelsbx.infox_parameters:
            # if model is in infox it will activate new reading routine
            modelsbx.infox_parameters.pop('model')
        aversion = modelsbx.atomsbx[0].version
        bversion = modelsbx.bondsbx[0].version
        modelsbx.infox_parameters['atomsversion'] = aversion
        modelsbx.infox_parameters['bondsversion'] = bversion
    else:
        calc = initialize_bopfox(modelsbx.infox_parameters
                                 , modelsbx=modelsbx)
    # mag = 1
    # if atom.info.has_key('spin'):
    #    mag = atom.info['spin']
    bulk = PhonopyAtoms(symbols=atom.get_chemical_symbols(),
                        scaled_positions=atom.get_scaled_positions(),
                        cell=atom.get_cell())
    phonon = Phonopy(bulk, [[cell_size[0], 0, 0], [0, cell_size[1], 0]
        , [0, 0, cell_size[2]]])
    # print phonon
    phonon.generate_displacements()
    supercells = phonon.get_supercells_with_displacements()
    force_sets = []
    for sp in supercells:
        sp_atom = Atoms(symbols=sp.get_chemical_symbols()
                        , positions=sp.get_positions(), cell=sp.get_cell()
                        , pbc=True)
        if old:
            calc = initialize_bopfox(modelsbx.infox_parameters \
                                     , atomsbx=atomsbx, bondsbx=bondsbx)
        else:
            calc = initialize_bopfox(modelsbx.infox_parameters
                                     , modelsbx=modelsbx)
        calc.set(task='force')
        # if mag == 2:
        #    add_initial_moments(sp_atom,modelsbx.infox_parameters)
        if calc.get_mode() == 'tight-binding':
            kmesh = kptdensity2monkhorstpack(sp_atom)
            calc.set(tbkpointfolding=kmesh)
        sp_atom.set_calculator(calc)
        ene = sp_atom.get_potential_energy()
        force_sets.append(sp_atom.get_forces())
    phonon.set_forces(force_sets)
    phonon.produce_force_constants()
    phonon.set_band_structure(path_kc)
    qpoints, dist, omega_kn, eigenvectors = phonon.get_band_structure()
    omega_kn = np.array(omega_kn)
    qpoints = np.array(qpoints)
    # Calculate phonon DOS
    # set here as twice the regular k-mesh in calculating DOS
    kmesh = np.array(kptdensity2monkhorstpack(atom)) * 2
    phonon.set_mesh(kmesh)
    phonon.set_total_DOS()
    omega_e, dos_e = phonon.get_total_DOS()
    atom.info['phonon_bs_total_kpts'] = qpoints
    # atom.info['phonon_bs_total_omega']= omega_kn
    # print np.shape(omega_kn)
    # print np.shape(omega_kn)
    atom.info['phonon_bs_omega'] = omega_kn[:, 0, :]
    atom.info['phonon_dos_omega'] = omega_e
    atom.info['phonon_dos_dos'] = dos_e
    modelsbx.infox_parameters = dict(old_params)
    return atom


##############################################################################

def calc_phonon_bopfox_EXP_BACKUP(atom, modelsbx, cell_size=(5, 5, 5)
                                  , show_plot=False, k_names=[], k_points=[], solver='phon'
                                  , res_band=10):
    """
    Calculates the phonon bandstructure and density of states
    using PHON or PHONOPY. Returns band structure and DOS.
    res_band is the resolution of the band_structure
    """
    print(('Calculating phonon band structure with %s' % solver))
    # mag = 1
    # if atom.info.has_key('spin'):
    #    mag = atom.info['spin']
    # if mag == 2:
    #    atom = add_initial_moments(atom,modelsbx.infox_parameters)
    # path_k, names = get_high_symmetry_points(atom,k_points)

    if len(k_names) != len(k_points):
        print("Length of pathnames is not same as path_k")
        # print np.shape(names)
        # print np.shape(path_k)
        # print np.shape(k_points)
        # k_names=np.chararray(len(k_points))
        # print k_names
        # k_names[:]=''
    k_points_long, q, Q = get_bandpath(k_points, atom.cell, res_band * len(k_points))
    k_names = np.chararray(len(k_points_long))
    k_names[:] = ''
    print((np.shape(k_points)))
    names = k_names
    atom.info['phonon_bs_path'] = k_points_long
    atom.info['phonon_bs_names'] = k_names
    if solver == 'phon':
        atom = run_phon(atom, modelsbx, k_points_long, cell_size)
    elif solver == 'phonopy':
        atom = run_phonopy(atom, modelsbx, k_points_long, cell_size)
    else:
        raise Exception('Solver %s not yet implemented.' % solver)
    omega_kn = atom.info['phonon_bs_omega']
    omega_e = atom.info['phonon_dos_omega']
    dos_e = atom.info['phonon_dos_dos']
    # Write output to file

    # print np.shape(omega_kn)
    # print q
    import csv
    # BS=[]
    # for i in range(len(omega_kn)):
    # Single BS point
    #     BS_q=[]
    #    BS_q.append(q[i])
    #    qpts_f=omega_kn[i]
    BS = []
    f = open('phonon_bs.dat', 'a')
    for i in range(len(omega_kn)):
        writer = csv.writer(f)
        line = []
        BS_q = []
        line.append(k_names[i])
        line.append(str(k_points_long[i][0]) + '|' + str(k_points_long[i][1])
                    + '|' + str(k_points_long[i][2]))
        print((k_points_long[i]))
        line.append(q[i])
        BS_q.append(q[i])
        qpts_f = omega_kn[i]
        # print qpts_f
        for j in qpts_f:
            line.append(j)
            BS_q.append(j)
        writer.writerow(line)
        print(line)
        BS.append(BS_q)
    f.close()
    f = open('phonon_dos.dat', 'w')
    for i in range(len(omega_e)):
        f.write('%5.5f  %5.5f\n' % (omega_e[i], dos_e[i]))
    if show_plot:
        # Plot the band structure and DOS
        # import pylab as pl
        import matplotlib.pyplot as pl
        pl.figure(1, (8, 6))
        pl.axes([.1, .07, .67, .85])
        for n in range(len(omega_kn[0])):
            omega_n = omega_kn[:, n]
            pl.plot(q, omega_n, 'k-', lw=2)
        for i in range(len(names)):
            if k_names[i] == 'Gamma':
                k_names[i] = r'$\%s$' % k_names[i]
            else:
                k_names[i] = r'$%s$' % k_names[i]
        pl.xticks(Q, k_names, fontsize=18)
        pl.yticks(fontsize=18)
        pl.xlim(q[0], q[-1])
        pl.ylim(0, np.max(omega_kn))
        pl.ylabel("Frequency (THz)", fontsize=22)
        pl.grid('on')

        pl.axes([.8, .07, .17, .85])
        pl.fill_between(dos_e, omega_e, y2=0, color='lightgrey', edgecolor='k'
                        , lw=1)
        # pl.ylim(0, 35)
        pl.ylim(0, np.max(omega_kn))
        pl.xticks([], [])
        pl.yticks([], [])
        pl.xlabel("DOS", fontsize=18)
        pl.savefig('phon_bs_dos.png')
        print('Figure dumped in phon_bs_dos.png')
    # pl.show()
    # print np.shape(q)
    # print np.shape(omega_kn)
    # S = np.array((q,omega_kn))
    DOS = np.vstack((omega_e, dos_e))
    return BS, DOS
