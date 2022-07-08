from ase.calculators import bopfox as bopcalc
from ase import Atoms
import numpy as np
from .output import print_format


def norm_sec_mom(atom, modelsbx, mu_target=1., eps=1e-4, maxiter=200):
    """
    Normalizes the unit cell volume to achieve a desired second moment.
    Adapted from strucmap.py
    """
    calc = bopcalc.BOPfox(modelsbx=modelsbx, printmu=True
                          , task='energy', scfsteps=200, efermisteps=200)

    atom = atom.copy()
    scaled_pos = atom.get_scaled_positions()

    cell_ini = atom.get_cell()
    atom.set_cell(cell_ini)
    atom.set_scaled_positions(scaled_pos)

    calc.set_atoms(atom)
    calc.calculate()

    mu_ini = calc.get_moments(atom_index='total', moment=2)
    mu1 = calc.get_moments(atom_index='total', moment=1)
    mu_ini -= mu1 ** 2

    alpha = 0.1
    count = 0
    mu_old = mu_ini
    cell_old = cell_ini
    while count < maxiter:
        if mu_old > mu_target:
            # cell_new = cell_old * (mu_old**(1./10.))
            cell_new = cell_old * (1 + (mu_old - mu_target) * alpha / mu_target)
        else:
            cell_new = cell_old / (1 + (mu_target - mu_old) * alpha / mu_target)
        atom.set_cell(cell_new, scale_atoms=True)
        calc.set_atoms(atom)
        calc.calculate()
        mu_new = calc.get_moments(atom_index='total', moment=2)
        mu1 = calc.get_moments(atom_index='total', moment=1)
        mu_new -= mu1 ** 2
        count += 1
        if mu_new is None:
            mu_new = mu_old
            cell_new = cell_old
        else:
            cell_old = cell_new
            mu_old = mu_new
        if (abs(mu_target - mu_new) < eps):
            break
    if count == maxiter:
        print_format('Cannot reached desired tolerance for mu2', level=3)
    return Atoms(cell=cell_new, scaled_positions=scaled_pos
                 , symbols=atom.get_chemical_symbols())


def get_mu2_ref(atoms, modelsbx):
    out = []
    calc = bopcalc.BOPfox(modelsbx=modelsbx, printmu=True
                          , task='energy', scfsteps=200, debug=False)
    for i in range(len(atoms)):
        atom = atoms[i].copy()
        calc.set_atoms(atom)
        calc.calculate()
        mu2 = calc.get_moments(atom_index='total', moment=2)
        mu1 = calc.get_moments(atom_index='total', moment=1)
        out.append(mu2 - mu1 ** 2)
    return out


def energy_diff(atoms, calc):
    """
    Determines the energy difference between all structures in atoms
    according to the structural energy difference theorem.
    """
    # adjust cell volumes so that mom2 = 1
    new = []
    modelsbx = calc.get_model().copy()
    modelsbx.infox_parameters['moments'] = 4
    modelsbx.infox_parameters['version'] = 'bop'
    # modelsbx.infox_parameters['bandwidth'] = 'azero'
    modelsbx.infox_parameters['repversion'] = 'None'
    # print modelsbx.infox_parameters
    mu_target = np.average(get_mu2_ref(atoms, modelsbx))
    for i in range(len(atoms)):
        new.append(norm_sec_mom(atoms[i], modelsbx, mu_target=mu_target))
        # calculate bond-energies
    calc.set_atoms(new)
    # calcs = calc._calcs
    # for i in range(len(calcs)):
    #    calcs[i].debug = True
    # calc._calcs = calcs
    bondene = calc.get_property(required_property='energy', contribution='bond')
    return bondene


if __name__ == '__main__':
    from ase.lattice import bulk
    # atom = bulk('Fe')
    # modelsbx = 'models.bx'
    # new = norm_sec_mom(atom,modelsbx,mu_target=3,maxiter=10)
    # print 'old cell: ', atom.get_cell()
    # print 'new_cell:' , new.get_cell()

    from .catcalc import CATCalc
    from .bopmodel import read_modelsbx

    modelsbx = read_modelsbx(filename='models.bx')[0]
    calc = CATCalc(model=modelsbx)
    atoms = [bulk('Fe')]

    print((energy_diff(atoms, calc)))
