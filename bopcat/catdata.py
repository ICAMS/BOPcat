#!/usr/bin/env python

# Definition of the CATData object

# This module is part of the BOPcat package

from copy import deepcopy
import os
from . import read_data
from .variables import minimum_distance_isolated_atom, atomic_properties
from .utils import stringtolist, listtostring
from .output import print_format
import numpy as np


###########################################################################
class CATData:
    """
    
    Defines a set of reference data for parametrization.
    
    It stores the structures and their properties as ASE Atoms objects. 

    Extended properties used in the parametrization such as eigenvalues,
    formation energies and elastic constants are stored in the info dictionary.

    .. todo:: extend ASE Atoms object

    The main functionality of this object is to query structures for 
    parameterization.

    :Parameters:
    
        - *controls*: instance of CATControls  
        
            CATControls object to initialize parameters     
                                  
        - *atoms*: list of ASE Atoms instances 
        
            None: will read from file.

        - *filename*: str    
            
            file containing the structures and their properties, can also be 
            path to file.

        - *dataparams*: dict
        
            dictionary of data specifications used to filter structures.
            
            sample keywords: `deltak`, `encut`

        - *sysparams*: dict    
          
            dictionary of system specifications used to filter structures.
            
            sample keywords: `calculation_type`, `spin`

        - *elements*: list   
            
            list of chemical symbols

        - *free_atom_energies*: dict
        
            dictionary of element:free atom energies.
            
            ``dimer``: will generate free_atom_energies from dimers.
            
            ``None``: will read from atomic_properties in :mod:`variables`.
            
        - *verbose*: int
        
            controls verbosity e.g.
            
            prints out structures if verbose > 1
                
    """

    def __init__(self, **kwargs):
        self._init_msg()
        self.controls = None
        self.filename = None
        self.dataparams = {}
        self.sysparams = {}
        self.elements = []
        self.free_atom_energies = None
        self.atoms = None
        self.free_atoms = None
        self.structures = None
        self.ref_data = None
        self.ref_atoms = None
        self.ref_weights = None
        self.orb_char = []
        self.quantities = None
        self.verbose = 1
        self.set(**kwargs)

    def _init_msg(self):
        print_format('Preparing reference data', level=1)

    def set(self, **kwargs):
        if 'controls' in kwargs:
            self.controls = kwargs['controls']
            if self.controls is not None:
                self._unfold_controls()
        for key, val in list(kwargs.items()):
            if key.lower() == 'atoms':
                self.atoms = val
            elif key.lower() == 'filename':
                self.filename = val.strip()
            elif key.lower() == 'dataparams':
                self.dataparams = val
            elif key.lower() == 'sysparams':
                self.sysparams = val
            elif key.lower() == 'elements':
                self.elements = val
            elif key.lower() == 'free_atom_energies':
                self.free_atom_energies = val
            elif key.lower() == 'controls':
                pass
            else:
                raise ValueError('Unrecognized key %s' % key)
        if self.atoms is None:
            self.atoms = self.get_atoms()
        if len(self.elements) < 1:
            raise ValueError('Provide list of elements.')
        self._normalize_smap()

    def _unfold_controls(self):
        self.filename = self.controls.data_filename
        self.dataparams = self.controls.data_parameters
        self.sysparams = self.controls.data_system_parameters
        self.elements = self.controls.elements
        self.free_atom_energies = self.controls.data_free_atom_energies

    def _read_from_file(self):
        filename = self.filename
        dataparams = self.dataparams
        sysparams = self.sysparams
        print_format("Reading data from %s" % filename, level=2)
        cwd = os.getcwd()
        if not os.path.isfile(filename):
            filename = os.path.expanduser(filename)
            path = os.path.dirname(filename)
            filename = os.path.basename(filename)
            if not os.path.isdir(path):
                raise ValueError("path: %s does not exist" % path)
            os.chdir(path)
        atoms = read_data.read(filename)
        os.chdir(cwd)
        atoms = self._refine_atoms(atoms, dataparams, sysparams)
        if len(atoms) < 1:
            raise ValueError('No atoms found that match specs.')
        return atoms

    def _include_atom(self, atom, dataparams, sysparams):
        out = True
        # struc = atom.info['strucname']
        for params in [dataparams, sysparams]:
            for key, val in list(params.items()):
                prop = atom.info[key.lower()]
                if prop is not None:
                    if isinstance(prop, str):
                        if key == 'stoichiometry':
                            if isinstance(val, list):
                                temp = [sorted(s) for s in val]
                                if sorted(prop) not in temp:
                                    out = False
                                    return out
                            else:
                                if sorted(prop.strip().lower()) != \
                                        sorted(val.strip().lower()):
                                    out = False
                                    return out
                        elif val.strip().lower() not in \
                                prop.lower():
                            out = False
                            return out
                    elif isinstance(prop, float):
                        if isinstance(val, list):
                            if prop not in val:
                                out = False
                        else:
                            if val != prop:
                                out = False
                                return out
                    elif isinstance(prop, int):
                        if isinstance(val, list):
                            if prop not in val:
                                out = False
                                return out
                        else:
                            if val != prop:
                                out = False
                                return out
                else:
                    out = False
                    return out
        return out

    def _refine_atoms(self, atoms, dataparams, sysparams):
        for i in range(len(atoms)):
            if atoms[i] is None:
                continue
            if not self._include_atom(atoms[i], dataparams, sysparams):
                atoms[i] = None
        atoms = [atom for atom in atoms if atom is not None]
        return atoms

    def _match(self, atom, struc):
        temp = struc.split('/')
        out = True
        if len(temp) > 1:
            if atom.info['data_type'] == 0:
                prop = [atom.get_chemical_formula(), atom.info['space_group']
                    , atom.info['system_type'], atom.info['calculation_type']
                    , atom.info['calculation_order']]
                ptype = [str, int, int, int, float]
                # struc is system_ID 
                for i in range(len(temp)):
                    if temp[i] == '*':
                        pass
                    elif '*' in temp[i]:
                        spl = temp[i].split('*')
                        spl = [s for s in spl if len(s) > 0]
                        for s in spl:
                            if s not in prop[i]:
                                out = False
                        if len(spl) != len(self.unique_species(atom)):
                            out = False
                    else:
                        if ptype[i](temp[i].strip()) != prop[i]:
                            out = False
                            break
            else:
                dt = atom.info['data_type']
                raise NotImplementedError('No option for %s' % dt)
        else:
            if '*' in struc:
                spl = struc.split('*')
                spl = [s for s in spl if len(s) > 0]
                for s in spl:
                    if s not in atom.info['strucname']:
                        out = False
            else:
                if struc != atom.info['strucname']:
                    out = False
        return out

    def _scan_ref_atoms(self, structures, q):
        included = []
        ref_weights = []
        ref_data = []
        ref_atoms = []
        atoms = self.atoms
        for i in range(len(structures)):
            for j in range(len(atoms)):
                if j in included:
                    continue
                if not self._match(atoms[j], structures[i]):
                    continue
                temp = None
                if q.lower() == 'energy':
                    try:
                        temp = atoms[j].get_potential_energy()
                    except:
                        continue
                elif q.lower() == 'eigenvalues':
                    try:
                        temp = atoms[j].info['eigenvalues']
                    except:
                        continue
                elif q.lower() == 'forces':
                    try:
                        temp = atoms[j].get_forces()
                    except:
                        continue
                elif q.lower() == 'stress':
                    try:
                        temp = atoms[j].get_stress()
                    except:
                        continue
                else:
                    try:
                        temp = atoms[j].info[q.lower()]
                    except:
                        continue
                if temp is None:
                    continue
                try:
                    temp_weight = atoms[j].info['weight']
                except:
                    temp_weight = None
                # is it necessary to do a deepycopy?
                temp_atoms = deepcopy(atoms[j])
                if q.lower() in ['vacancy_energy']:
                    add = []
                    # get reference atoms
                    uniques = self.unique_species(atoms[j])
                    sp = 1
                    if 'spin' in atoms[j].info:
                        sp = atoms[j].info['spin']
                    for us in uniques:
                        add.append(self.get_ground_state(elements=[us]
                                                         , spin=sp))
                    temp_atoms.info['reference_atoms'] = add
                if q.lower() in ['energy']:
                    # shift energies with respect to reference
                    sym = atoms[j].get_chemical_symbols()
                    for s in sym:
                        temp -= self.get_free_atom_energies()[s]
                if q.lower() in ['eigenvalues']:
                    # save also orbital character
                    if len(temp) < 1:
                        continue
                    # temp -= atom.info['fermi_level']
                    try:
                        self.orb_char.append(
                            atoms[j].info['orbital_character'])
                    except:
                        continue
                included.append(j)
                temp_atoms.info['required_property'] = q
                ref_atoms.append(temp_atoms)
                ref_data.append(temp)
                ref_weights.append(temp_weight)
        return ref_atoms, ref_data, ref_weights

    def _get_reference_energies(self, elements, spin):
        out = {}
        for e in elements:
            gs = self.get_ground_state(elements=[e], spin=spin)
            if gs is None:
                return None
            out[e] = gs.get_potential_energy() / len(gs)
        return out

    def _calc_form_ene(self, atom, refene=None):
        sym = atom.get_chemical_symbols()
        unique = self.unique_species(atom)
        spin = 0
        if 'spin' in atom.info:
            spin = atom.info['spin']
        if refene is None:
            refene = self._get_reference_energies(unique, spin)
        if refene is None:
            raise RuntimeError('Cannot get reference energies for elements.')
        ene = atom.get_potential_energy()
        for s in sym:
            ene -= refene[s]
        ene /= len(atom)
        return ene

    def _sort_atoms(self, sort_by='energy', refene=None):
        if len(self.ref_data) == 0 or len(self.ref_atoms) == 0:
            return
        sortd = []
        randN = np.random.permutation(list(range(len(self.ref_atoms))))
        for i in range(len(self.ref_atoms)):
            if sort_by == 'energy':
                sortd.append((self.ref_atoms[i].get_potential_energy() / len( \
                    self.ref_atoms[i]), i))
            elif sort_by == 'volume':
                sortd.append((self.ref_atoms[i].get_volume() / len( \
                    self.ref_atoms[i]), i))
            elif sort_by == 'formation_energy':
                sortd.append((self._calc_form_ene(self.ref_atoms[i], refene=refene), i))
            elif sort_by == 'random':
                sortd.append((randN[i], i))
            else:
                raise NotImplementedError("Cannot sort by %s" % sort_by)
        sortd.sort()
        tmp_atoms = []
        tmp_data = []
        for i in range(len(sortd)):
            tmp_atoms.append(self.ref_atoms[sortd[i][1]])
            tmp_data.append(self.ref_data[sortd[i][1]])
        self.ref_atoms = list(tmp_atoms)
        self.ref_data = list(tmp_data)

    def get_ref_atoms(self, structures=None, quantities=None, sort_by=None, refene=None):
        """
        Returns list of ASE atoms objects specified in structures and with
        property in quantities.

        :Parameters:
        
            - *structures*: list
        
                list of strings of strucname of part of strucname, e.g. 'bcc' 
                or system_ID, wildcards are permitted, e.g. ``Fe*/*/*/1/*``
                (see definition of system_ID)  
            
            - *quantities*: list
        
                list of desired properties that a structure must possess to be 
                included. If a structure has more than one property, it will
                be included multiple times. 
                
                if quantities is None will simply return what is stored in 
                memory.
                
        """
        if quantities is None:
            if self.ref_atoms is None:
                print_format('Warning: No ref_atoms stored in memory', level=2)
                self.ref_atoms = []
            return self.ref_atoms
        self.quantities = list(quantities)
        self.ref_atoms = []
        self.ref_data = []
        self.ref_weights = []
        if structures is None:
            # structures = self.get_structures()
            structures = ['*']
        if not isinstance(structures, list):
            print_format("Provide a list of structure names.", level=2)
            return
        for q in quantities:
            atoms, data, weights = self._scan_ref_atoms(structures, q)
            self.ref_atoms += atoms
            self.ref_data += data
            self.ref_weights += weights
        if self.verbose > 1:
            print_format("%d reference structures found." % len(self.ref_atoms)
                         , level=2)
        for i in range(len(self.ref_atoms)):
            info = self.ref_atoms[i].info
            if self.verbose > 1:
                print_format("strucname: %30s  system_ID: %20s  property: %20s"
                             % (info['strucname'], info['system_ID']
                                , info['required_property']), level=3)
        if sort_by is not None:
            self._sort_atoms(sort_by=sort_by, refene=refene)
        return self.ref_atoms

    def get_ref_data(self, structures=None, quantities=None):
        """
        Returns the properties of the reference atoms. 
        See :func:`get_ref_atoms` for description of parameters.
        
        If quantities is None, it will return what is stored in memory,
        otherwise, it will initialize ref_atoms if quantities is not
        the same as stored.
        """
        if quantities is not None:
            if quantities != self.quantities:
                self.get_ref_atoms(structures=structures, quantities=quantities)
        else:
            if self.ref_data is None:
                print_format('Warning: No ref_data stored in memory', level=2)
                self.ref_data = []
        return self.ref_data

    def get_ref_weights(self, structures=None, quantities=None):
        """
        Returns the weights of the reference atoms in the optimization. 
        See :func:`get_ref_atoms` for description of parameters.
        
        If quantities is None, it will return what is stored in memory,
        otherwise, it will initialize ref_atoms if quantities is not
        the same as stored.

        If any of the reference atoms has no weight then all the weights
        will be nullified. 
        """
        if quantities is not None:
            if quantities != self.quantities:
                self.get_ref_atoms(structures=structures, quantities=quantities)
        else:
            if self.ref_weights is None:
                print_format('Warning: No ref_data stored in memory', level=2)
                self.ref_weights = []
        if None in self.ref_weights:
            warn = '''Warning: No weight assigned for at least one 
                      structure/property. Will set weights to None!
                   '''
            print_format(warn, level=2)

        return self.ref_weights

    def get_atoms_info(self, key):
        """
        Returns the `info[key]` values of all atoms
        
        :Parameters:
        
            - *key*: str
        
                key of entry in info dictionary
        """
        out = []
        if self.atoms is None:
            atoms = self.get_atoms()
        for atom in self.atoms:
            if key in atom.info:
                out.append(atom.info[key])
            else:
                out.append(None)
        return out

    def get_atoms(self):
        """
        Returns list of all ASE atoms objects of all structures. 
        Builds them from file if not initialized.
        """
        if self.atoms is None:
            if self.filename != None:
                self.atoms = self._read_from_file()
            else:
                raise ValueError("No file to read.")
        return self.atoms

    def get_structures(self, name=False):
        """
        Returns the system_ID or strucname (name=True) of all structures.
        The definition of system_ID varies depending on data_type of structure.
        """
        if self.atoms is None:
            atoms = self.get_atoms()
        if name:
            return [atom.info['strucname'] for atom in self.atoms]
        else:
            return [atom.info['system_ID'] for atom in self.atoms]

    def _get_free_atoms(self):
        if self.atoms is None:
            atoms = self.get_atoms()
        if self.free_atoms is None:
            self.free_atoms = []
            min_d = minimum_distance_isolated_atom()
            free_atom = None
            for elem in self.elements:
                for atom in self.atoms:
                    sym = atom.get_chemical_symbols()
                    if len(atom) == 2 and sym == [elem, elem]:
                        # atom should be a dimer
                        if atom.get_distance(0, 1, mic=True) > min_d:
                            free_atom = deepcopy(atom)
                            self.free_atoms.append(free_atom)
                            break
                if free_atom is None:
                    print_format('Failed to calculate free energy', level=2)
                    print_format('from %s-%s dimer.' % (elem, elem), level=2)
                    print_format('Include dimer (d > %2.1f A)' % min_d, level=2)
                    raise ValueError('No free atoms found.')
        return self.free_atoms

    def get_free_atom_energies(self):
        """
        Returns a ditionary of free atom energies corresponding to each of 
        the elements.
        """
        if self.free_atom_energies == 'dimer':
            self.free_atom_energies = {}
            if self.free_atoms is None:
                atoms = self._get_free_atoms()
            for atom in self.free_atoms:
                ene = atom.get_potential_energy() / len(atom)
                elem = atom.get_chemical_symbols()[0]
                self.free_atom_energies[elem] = ene
        elif self.free_atom_energies is None:
            self.free_atom_energies = {}
            for elem in self.elements:
                try:
                    self.free_atom_energies[elem] = \
                        atomic_properties()[elem]['free_atom_energy']
                except:
                    print(('No free_atom_energy data found for %s' % elem))
                    print('Setting to 0.0')
                    self.free_atom_energies[elem] = 0.0
        else:
            pass
        return self.free_atom_energies

    def get_equilibrium_distance(self, structure):
        """
        Returns the smallest bond length of the lowest energy structure
        in the group
        
        :Parameters:
        
            - *structure*: str
        
                strucname,  e.g. `'dimer'`
            
                system_ID, e.g. `'Fe/229/0/1/*'` : all bcc-Fe E-V structures
        """
        if self.atoms is None:
            self.get_atoms()
        mine = 10000.
        minat = None
        for atom in self.atoms:
            if self._match(atom, structure):
                if atom.get_potential_energy() < mine:
                    mine = atom.get_potential_energy()
                    minat = atom
        if minat is None:
            print_format('Cannot find any %s structure.' % structure, level=3)
            return
        mind = 10000.
        # if len(minat) < 2:
        minat = minat.repeat(2)
        for i in range(len(minat)):
            for j in range(len(minat)):
                if j > i:
                    d = minat.get_distance(i, j, mic=True)
                    if d < mind:
                        mind = d
        return mind

    def unique_species(self, atom):
        sym = atom.get_chemical_symbols()
        unique = []
        for i in range(len(sym)):
            if sym[i] not in unique:
                unique.append(sym[i])
        unique.sort()
        return unique

    def _normalize_smap(self):
        key = 'structuremap_coordinates'
        coor = []
        has_smap = []
        for i in range(len(self.atoms)):
            try:
                sc = self.atoms[i].info[key]
            except:
                has_smap.append(False)
                continue
            if sc is not None:
                coor.append(sc)
                has_smap.append(True)
            else:
                has_smap.append(False)
        if len(coor) < 1:
            return
        # copied from structmap class should also be a global function
        # the global function normalize_coordinates do not seem
        # to work
        minx = 1E99
        miny = 1E99
        maxx = -1E99
        maxy = -1E99
        for i in range(len(coor)):
            xy = np.amin(coor[i], axis=0)
            if xy[0] < minx:
                minx = xy[0]
            if xy[1] < miny:
                miny = xy[1]
            xy = np.amax(coor[i], axis=0)
            if xy[0] > maxx:
                maxx = xy[0]
            if xy[1] > maxy:
                maxy = xy[1]
        for i in range(len(coor)):
            c = np.array(coor[i])
            dx = (c.T[0] - minx) / float(maxx - minx)
            dy = (c.T[1] - miny) / float(maxy - miny)
            coor[i] = list(np.array([dx, dy]).T)

        count = 0
        for i in range(len(self.atoms)):
            if has_smap[i]:
                self.atoms[i].info[key] = coor[count]
                count += 1

    def get_structuremap_distance(self, struc0, strucs, index='total'):
        """
        Returns the distance of the strucs from struc0 in the 
        structure map
        
        :Parameters:
        
            - *struc0*: str
        
                strucname,  e.g. `'dimer'`
            
                system_ID, e.g. `'Fe/229/0/1/*'` : all bcc-Fe E-V structures

                ASE Atoms object, coordinates is expected to be normalized
                globally

            - *strucs*: list
 
                list of structures (format similar to struc0)

            - *mode*: str
 
                average/total/maximum of the distances

            - *index*: list, int, str

                list of atom indices corresponding to each struc in strucs
                whose distance from struc0 is returned, if int will apply 
                same index to all strucs, if total will average the coordinates
        """
        coor = []
        key = 'structuremap_coordinates'

        atoms = []
        for s in [struc0] + strucs:
            if isinstance(s, str):
                for atom in self.atoms:
                    if self._match(atom, s):
                        atoms.append(atom)
            else:
                atoms.append(s)
        for at in atoms:
            try:
                coor.append(at.info[key])
            except:
                name = at.info['strucname']
                raise ValueError('%s does not have %s' % (name, key))
        # coor has shape (len(atoms),natoms/cell,2)
        temp = np.zeros((len(coor), 2))
        if isinstance(index, int):
            index = [index] * len(coor)
            for i in range(len(coor)):
                temp[i] = coor[index[i]]
        elif index == 'total':
            for i in range(len(coor)):
                temp[i] = np.average(coor[i], axis=0)
        # get distance from reference
        temp = temp - temp[0]
        temp = temp.T
        dist = np.sqrt(temp[0] ** 2 + temp[1] ** 2)
        dist = dist[1:]
        return dist

    def get_ground_state(self, elements=None, out='atoms', spin=0):
        """
        Returns the lowest energy structure containing all the elements
        in elements. 

        For alloys, energy of formation is calculated.              
        
        :Parameters:
              
              - *elements*: list
                  list of chemical symbols
              
              - *out*: str
                  directive for output, if 'atoms' will return the ASE Atoms
                  object, otherwise will return info[out]
              
              - *spin*: int
                  ``0`` : all   ``1`` : only non-mag  ``2`` : only mag

        """
        atoms = self.atoms
        lo = 1000.
        gs = None
        if elements is None:
            # assumes that all atoms are unaries and of the same element
            elements = atoms[-1].get_chemical_symbols()[:1]
        unique = []
        for e in elements:
            if e not in unique:
                unique.append(e)
        elements = list(unique)
        elements.sort()
        for i in range(len(atoms)):
            unique = self.unique_species(atoms[i])
            if unique == elements:
                # consider only bulk/relax calculations
                if 'system_ID' in atoms[i].info:
                    sID = stringtolist(atoms[i].info['system_ID'])
                else:
                    continue
                if sID[2] != 0:  # or sID[3] != 0:
                    continue
                mag = 1
                if 'spin' in atoms[i].info:
                    mag = atoms[i].info['spin']
                if spin == 0:
                    pass
                elif spin in [1, 2]:
                    if spin != mag:
                        continue
                if len(unique) == 1:
                    try:
                        ene = atoms[i].get_potential_energy() / len(atoms[i])
                    except:
                        continue
                    if ene < lo:
                        lo = ene
                        gs = atoms[i]
                else:
                    sym = atoms[i].get_chemical_symbols()
                    try:
                        ene = atoms[i].get_potential_energy()
                    except:
                        continue
                    refene = {}
                    for s in unique:
                        at = self.get_ground_state(elements=[s], spin=spin)
                        refene[s] = at.get_potential_energy() / len(at)
                    for s in sym:
                        ene -= refene[s]
                    ene /= len(atoms[i])
                    if ene < lo:
                        lo = ene
                        gs = atoms[i]
                    # raise NotImplementedError("Invalid elements.")
        if gs is None:
            print_format('Warning: No structure found!', level=3)
            return
        if out == 'atoms':
            return gs
        else:
            return gs.info[out]

    def get_parent(self, atom):
        """
        Returns the system_ID of the parent of atom.
        
        If parent is not found will return system_ID of atom.
        
        .. note:: 
            Only considers bulk atoms with different volumes from parent 
            but should be generalized for other calculation types
            However, for the structure map, even if two atoms are related
            but have different cells, e.g transformation path they will
            have different moments hence should be treated as separate
            structures.

        :Parameters:

            - *atom*: instance of `ase.Atoms`        
        
                child of the parent atom
        
        """
        sID0 = stringtolist(atom.info['system_ID'])
        atoms = self.atoms
        found = False
        for i in range(len(atoms)):
            if 'system_ID' in atoms[i].info:
                sID = stringtolist(atoms[i].info['system_ID'])
            else:
                continue
            if sID[2:4] != [0, 0]:
                # parents are only bulk, relaxed atoms
                continue
            if sID0[0:3] == sID[0:3]:
                found = True
                parent = listtostring(sID)
                break
        if not found:
            # does not have a parent
            parent = listtostring(sID0)
        return parent


if __name__ == "__main__":
    data = CATData(filename='../examples/Fe.fit', elements=['Fe'])
    data.verbose = 1
    structures = data.get_structures(name=True)
    print(('Structure names:', structures))
    ref_atoms = data.get_ref_atoms(structures=['dimer'], quantities=['energy'])
    print(('number of dimers with energy: ', len(ref_atoms)))
    gs = data.get_ground_state(out='strucname')
    print(('lowest energy structure: ', gs))
    r0 = data.get_equilibrium_distance(structure=gs)
    print(('GS equilibrium distance: ', r0))
    print(('dimer equilibrium distance', data.get_equilibrium_distance(
        structure='dimer')))
