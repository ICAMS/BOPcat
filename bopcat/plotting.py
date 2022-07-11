# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

import numpy as np
from copy import deepcopy
from .utils import neigh_dist
from .output import print_format
from scipy import optimize
from ase import Atoms


def Birch_Murnaghan(param, vol):
    """
    Birch-Murnaghan EOS. external function for BMfit
    """
    E0 = param[0]
    B0 = param[1]
    BP = param[2]
    V0 = param[3]
    E = E0 + (9.0 * V0 * B0) / 16.0 * (((V0 / vol) ** (2.0 / 3.0) - 1.0) ** 3.0 * BP +
                                       ((V0 / vol) ** (2.0 / 3.0) - 1.0) ** 2.0 * (
                                                   6.0 - 4.0 * (V0 / vol) ** (2.0 / 3.0)))
    return E


def evfunc(param, e, v):
    """
    Residual function for BMfit
    """
    # alpha = e - Murn(param,v)
    alpha = e - Birch_Murnaghan(param, v)
    return alpha


def eosfit(atoms):
    if isinstance(atoms[0], Atoms):
        ve = []
        for atom in atoms:
            ene = atom.get_potential_energy() / len(atom)
            vol = atom.get_volume() / len(atom)
            ve.append((vol, ene))
        ve = np.array(ve).T
    else:
        ve = [list(atoms[i]) for i in range(len(atoms))]
        ve.sort()
        ve = np.array(ve)
        if np.shape(ve)[0] != 2:
            ve = ve.T
    poly2 = np.polyfit(ve[0], ve[1], 2)
    v0 = -poly2[1] / (2 * poly2[0])
    e0 = poly2[0] * v0 ** 2 + poly2[1] * v0 + poly2[2]
    b0 = 2 * poly2[0] * v0
    bP = 4.
    x0 = [e0, b0, bP, v0]
    GPa = 1.0 / 160.217733
    fit, err = optimize.leastsq(evfunc, x0, args=(ve[1], ve[0]))
    V0 = fit[3]
    B0 = fit[1] / GPa
    E0 = fit[0]
    BP = fit[2]
    return E0, B0, BP, V0


def sort_atoms(atoms):
    temp = []
    for i in range(len(atoms)):
        atom = deepcopy(atoms[i])
        a = atom.repeat(2).get_distance(0, 1)
        temp.append((a, atom))
    temp.sort()
    new_atoms = []
    for i in range(len(temp)):
        new_atoms.append(temp[i][1])
    return new_atoms


def plot_electronic_band_structure(atoms, filename='ebs.png'):
    import matplotlib.pyplot as pl
    marks = ['o', 'v', 'd', 's', 'p', 'h', '>', '<', '^']
    cols = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig = pl.figure()
    eigs = []
    calcs = []
    atoms = sort_atoms(atoms)
    if len(atoms) == 1:
        dtype = 'bandstructure'
    else:
        dtype = 'list'
    for i in range(len(atoms)):
        temp = atoms[i].info['eigenvalues']
        if temp is None:
            print(("Nothing to plot for %s" % atoms[i].info['strucname']))
            continue
        else:
            eigs.append(list(temp))
        calc = atoms[i].get_calculator().name
        if calc not in calcs:
            calcs.append(calc)
    fills = np.linspace(0.5, 1, len(calcs))
    temp = {}
    for i in range(len(fills)):
        temp[calcs[i]] = fills[i]
    fills = dict(temp)
    del temp
    natom = len(eigs)
    pl.clf()
    for iatom in range(len(eigs)):
        nspin = len(eigs[iatom])
        for ispin in range(nspin):
            pl.subplot(ispin + 1, 2, 1)
            nkpt = len(eigs[iatom][ispin])
            data = np.array(eigs[iatom][ispin]).T
            for iband in range(len(data)):
                if dtype == 'bandstructure':
                    ks = np.arange(nkpt) + iatom * nkpt
                else:
                    ks = np.arange(nkpt) + iatom * nkpt
                fill = fills[atoms[iatom].get_calculator().name]
                pl.plot(ks, data[iband], '%s%s' % (cols[iband % len(cols)]
                                                   , marks[iband % len(marks)]), alpha=fill)
        for ispin in range(nspin):
            pl.subplot(ispin + 1, 2, 2)
            if dtype == 'bandstructure':
                nkpt = len(eigs[iatom][ispin])
                data = np.array(eigs[iatom][ispin]).T
                flat_data = data.flatten()
                hist = pl.hist(flat_data, 50, orientation='horizontal', edgecolor='r')
            else:  # expecting dimers
                if 'orbital_character' in atoms[iatom].info:
                    if len(atoms[iatom].info['orbital_character']) > 0:
                        orbs = atoms[iatom].info['orbital_character']
                        nkpt = len(eigs[iatom][ispin])
                        data = np.array(orbs[ispin]).T
                        for iband in range(len(data)):
                            # ks = atom[iatom].get_distance(0,1)
                            ks = neigh_dist(atoms[iatom], origin=0, N=1) * len(data[iband])
                            pl.plot(ks, data[iband]
                                    , '%s%s' % (cols[iband % len(cols)], marks[iband % len(marks)]))
    for ispin in range(len(eigs[0])):
        pl.subplot(ispin + 1, 2, 1)
        pl.ylabel("Energy(eV)")
        if dtype == 'bandstructure':
            pl.xlabel("k-point index")
        else:
            pl.xlabel("distance")
        pl.subplot(ispin + 1, 2, 2)
        if dtype == 'bandstructure':
            pl.xlabel('Number of states')
            pl.ylabel('Energy(eV)')
        else:
            pl.xlabel('Nk')
            pl.ylabel('orbital character')
    pl.savefig(filename)
    pl.show()


def plot_energy_volume(atoms, reference_energies=None, filename='ev.png'
                       , show_plot=False):
    """
    Plots an E-V curve and saves the output graph to ev.png
    """
    import matplotlib.pyplot as pl
    marks = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    marks = ['%s-' % m for m in marks]
    cols = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig = pl.figure()
    strucs = []
    calcs = []
    if reference_energies is None:
        reference_energies = [{}] * len(atoms)
    if isinstance(reference_energies, dict):
        reference_energies = [reference_energies] * len(atoms)
    for i in range(len(atoms)):
        struc = atoms[i].info['strucname'].split('-')[0]
        calc = atoms[i].get_calculator().name
        if struc not in strucs:
            strucs.append(struc)
        if calc not in calcs:
            calcs.append(calc)
    fills = np.linspace(0.5, 1, len(calcs))
    for i in range(len(strucs)):
        for j in range(len(calcs)):
            data = []
            for k in range(len(atoms)):
                if strucs[i] in atoms[k].info['strucname'] and \
                        atoms[k].get_calculator().name == calcs[j]:
                    try:
                        ene = atoms[k].get_potential_energy()
                        for s in atoms[k].get_chemical_symbols():
                            if s in reference_energies[k]:
                                ene -= reference_energies[k][s]
                        ene /= len(atoms[k])
                        vol = atoms[k].get_volume() / len(atoms[k])
                    except:
                        continue
                    data.append((vol, ene))
            data.sort()
            data = np.array(data).T
            lcalc = calcs[j]
            if lcalc == 'unknown': lcalc = 'DFT'
            if j < 1:
                pl.plot(data[0], data[1]
                        , '%s%s' % (cols[i % len(cols)], marks[i % len(marks)])
                        , alpha=fills[j], label=strucs[i] + '-' + lcalc)
            else:
                if i < 1:
                    pl.plot(data[0], data[1]
                            , '%s%s' % (cols[i % len(cols)], marks[i % len(marks)])
                            , alpha=fills[j], label=strucs[i] + '-' + lcalc)
                else:
                    pl.plot(data[0], data[1]
                            , '%s%s' % (cols[i % len(cols)], marks[i % len(marks)])
                            , alpha=fills[j])

    pl.xlabel('Volume per atom ($\AA^3$)')
    pl.ylabel('Energy (eV)')
    pl.legend(loc='best')
    pl.savefig(filename, dpi=200)
    print_format('Plot dumped in %s' % filename)
    if show_plot:
        pl.show()
