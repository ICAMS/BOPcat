# This module is part of the BOPcat package
# available at https://github.com/ICAMS/BOPcat
# distributed under GNU General Public License v3.0

from ase import Atoms


class NoCalcError(RuntimeError):
    def __str__(self):
        return ''' Atoms object has no calculator'''


class BOPAtomsInitError(Exception):
    def __str__(self):
        return '''This is an extension to the ASE Atoms object and should not
                  be used as standalone object
               '''


class __BOPAtoms:
    '''
    Extension of the standard ASE Atoms class to accommodate BOPfox and BOPcat 
    specific properties. It is not intended to be used as standalone.
    '''

    def __init__(self):
        raise BOPAtomsInitError

    def __fitatoms_init__(self, *args, **kwargs):
        Atoms.__atoms_init__(self, *args, **kwargs)
        self._eigenvalues = None
        self.contributions_energy = None
        self.contributions_forces = None
        self.strucname = None
        self._sytem_ID = None
        self._orbital_character = None

    def set(self, kwargs):
        calc = self.get_calculator()
        if calc is None:
            return
        if calc.get_name().lower() == 'bopfox':
            calc.update(self, 'energy')
            self._eigenvalues = calc.eigenvalues
            self._contributions_energy = calc.contributions_energy
            self._contributions_forces = calc.contributions_forces

    def _get_prop_from_bopfox(self, prop, **kwargs):
        # if prop.lower() == 'eigenvalues':
        #    return self._calc.get_eigenvalues()
        # elif prop.lower() == 'orbital_character':
        #    return self._calc.get_orbital_character()
        if prop.lower() == 'dos':
            return self._calc.get_dos()
        elif prop.lower() == 'fermi_energy':
            return self._calc.get_fermi_energy()
        elif prop.lower() == 'moments':
            atom_index = kwargs['atom_index']
            moment = kwargs['moment']
            return self._calc.get_moments(atom_index, moment)
        elif prop.lower() == 'anbn':
            atom_index = kwargs['atom_index']
            moment = kwargs['moment']
            return self._calc.get_anbn(atom_index, moment)
        elif prop.lower() == 'charges':
            atom_index = kwargs['atom_index']
            return self._calc.get_moments(atom_index)
        elif prop.lower() == 'contributions_energy':
            key = kwargs['key']
            return self._calc.get_contributions_energy(key)
        elif prop.lower() == 'contributions_forces':
            key = kwargs['key']
            return self._calc.get_contributions_forces(key)
        elif prop.lower() == 'spin':
            spinpol = self._calc.get_spin_polarized()
            if spinpol:
                return 2
            else:
                return 1
        else:
            return self._get_prop_from_info(prop, **kwargs)

    def _get_prop_from_info(self, prop, **kwargs):
        if prop in self.info:
            return self.info[prop]
        else:
            # print 'No property %s'%prop
            return None

    def get_property(self, prop, **kwargs):
        if self._calc is None:
            return self._get_prop_from_info(prop, **kwargs)
        try:
            name = self._calc.name.lower()
        except:
            name = 'unknown'
        if name == 'bopfox':
            return self._get_prop_from_bopfox(prop, **kwargs)
        else:
            return self._get_prop_from_info(prop, **kwargs)

    def get_eigenvalues(self):
        return self.get_property('eigenvalues')

    def get_orbital_character(self):
        return self.get_property('orbital_character')

    def get_kpoints(self):
        return self.get_property('kpoints')

    def get_coord_k(self):
        return self.get_property('coord_k')

    def get_strucname(self):
        return self.get_property('strucname')

    def get_data_type(self):
        return self.get_property('data_type')

    def get_system_type(self):
        return self.get_property('system_type')

    def get_calculation_type(self):
        return self.get_property('calculation_type')

    def get_calculation_order(self):
        return self.get_property('calculation_order')

    def get_space_group(self):
        return self.get_property('space_group')

    def get_data_ID(self):
        return self.get_property('data_ID')

    def get_system_ID(self):
        return self.get_property('system_ID')

    def get_reference_atoms(self):
        return self.get_property('reference_atoms')

    def get_required_property(self):
        return self.get_property('required_property')

    def get_spin(self):
        return self.get_property('spin')

    def get_code(self):
        return self.get_property('code')

    def get_basis_set(self):
        return self.get_property('basis_set')

    def get_pseudopotential(self):
        return self.get_property('pseudopotential')

    def get_xc_functional(self):
        return self.get_property('xc_functional')

    def get_encut(self):
        return self.get_property('encut')

    def get_deltak(self):
        return self.get_property('deltak')

    def get_encut_ok(self):
        return self.get_property('encut_ok')

    def get_deltak_ok(self):
        return self.get_property('deltak_ok')

    def get_stoichiometry(self):
        stoic = self.get_property('stoichiometry')
        if stoic is not None:
            assert (stoic == self.get_chemical_formula())
        return stoic

    def get_moments(self, atom_index=0, moment=2):
        return self.get_property('moments', atom_index=atom_index, moment=moment)

    def get_anbn(self, atom_index=0, moment=2):
        return self.get_property('anbn', atom_index=atom_index, moment=moment)

    def get_contributions_energy(self, key='bond'):
        return self.get_property('contributions_energy', key=key)

    def get_contributions_forces(self, key='analytic'):
        return self.get_property('contributions_forces', key=key)

    def get_eigenvalues(self):
        return self.get_property('eigenvalues')

    def get_orbital_character(self):
        return self.get_property('eigenvalues')

    def get_dos(self):
        return self.get_property('dos')

    def get_fermi_energy(self):
        return self.get_property('fermi_energy')


# extend ASE Atoms functionalities by adding FitAtoms
Atoms.__bases__ = Atoms.__bases__ + (__BOPAtoms,)
Atoms.__atoms_init__ = Atoms.__init__
Atoms.__init__ = __BOPAtoms.__fitatoms_init__
