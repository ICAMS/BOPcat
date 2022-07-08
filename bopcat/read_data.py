from ase import Atoms, Atom
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.dft.kpoints import kpoint_convert
from . import variables
from .output import print_format


def gen_ID(atom):
    info = atom.info
    dtype = info['data_type']
    if dtype == 0:
        d1 = info['code']
        d2 = info['basis_set']
        d3 = info['xc_functional']
        d4 = info['pseudopotential']
        d5 = info['spin']
        s1 = info['stoichiometry']
        s2 = info['space_group']
        s3 = info['system_type']
        s4 = info['calculation_type']
        s5 = info['calculation_order']
    else:
        raise NotImplementedError('No options for %s' % dtype)
    data_ID = '%s/%s/%s/%s/%s/%s' % (dtype, d1, d2, d3, d4, d5)
    system_ID = '%s/%s/%s/%s/%s' % (s1, s2, s3, s4, s5)
    info['data_ID'] = data_ID
    info['system_ID'] = system_ID
    atom.info = info
    return atom


def check_info(atom):
    info = atom.info
    dtype = info['data_type']
    required = variables.required_data_fields(dtype=dtype)
    for key, val in list(required.items()):
        if key not in info:
            info[key] = val
    atom = gen_ID(atom)
    return atom


def reshape_eigenvalues(eigs, spin):
    nkpts = len(eigs)
    nbands = int(len(eigs[0]) / spin)
    temp = np.zeros((spin, nkpts, nbands))
    for i in range(spin):
        for nk in range(nkpts):
            temp[i][nk] = eigs[nk][i * nbands:(i + 1) * nbands]
    return temp


def reshape_k_points(kpts, spin, coord_k, cell):
    temp = np.zeros((spin, len(kpts), 3))
    for i in range(spin):
        for nk in range(len(kpts)):
            temp[i][nk] = kpts[nk][i * 3:(i + 1) * 3]
            if coord_k.lower()[0] == 'd':
                temp[i][nk] = kpoint_convert(cell, skpts_kc=temp[i][nk])
    kpts = np.array(temp)
    return kpts


def reshape_orbital_character(orb_char, spin):
    nkpts = len(orb_char)
    nbands = int(len(orb_char[0]) / (9 * spin))
    temp = np.zeros((spin, nkpts, nbands, 9))
    for i in range(spin):
        for nk in range(nkpts):
            temp2 = orb_char[nk][i * nbands * 9:(i + 1) * nbands * 9]
            temp[i][nk] = temp2.reshape(nbands, 9)
    orb_char = temp
    return orb_char


def reshape_arrays(info, cell):
    for key, val in list(info.items()):
        if isinstance(val, list):
            info[key] = np.array(val)
    if 'eigenvalues' in info:
        eigs = info['eigenvalues']
        if len(eigs) > 0:
            spin = info['spin']
            info['eigenvalues'] = reshape_eigenvalues(eigs, spin)
    # k_points
    if 'k_points' in info:
        kpts = info['k_points']
        if len(kpts) > 0:
            coord_k = info['coord_k']
            spin = info['spin']
            info['k_points'] = reshape_k_points(kpts, spin, coord_k, cell)
    # orbital_character
    if 'orbital_character' in info:
        orb_char = info['orbital_character']
        if len(orb_char) > 0:
            spin = info['spin']
            info['orbital_character'] = reshape_orbital_character(orb_char
                                                                  , spin)
    # structuremap_coordinates
    if 'structuremap_coordinates' in info:
        smap_coor = np.array(info['structuremap_coordinates'])
        if len(smap_coor) > 0:
            info['structuremap_coordinates'] = \
                smap_coor.reshape((len(smap_coor), 2))
    return info


def read_line(line, start, end, dtype):
    out = line.split()[start:end]
    if dtype == dict:
        temp = {}
        for i in np.arange(0, len(out) - 1, 2):
            try:
                temp[out[i]] = float(out[i + 1])
            except:
                temp[out[i]] = out[i + 1]
        out = dict(temp)
    elif dtype == list:
        temp = []
        for i in range(len(out)):
            try:
                temp.append(float(out[i]))
            except:
                temp.append(out[i])
        out = list(temp)
    else:
        out = dtype(out[0].strip())
    return out


def create_atom(**kwargs):
    """ 
    Creates an ASE-Atoms Object
    """
    pos = kwargs['positions']
    cell = kwargs['cell']
    sym = kwargs['symbols']
    energy = kwargs['energy']
    forces = kwargs['forces']
    stress = kwargs['stress']
    magmoms = kwargs['magmoms']
    info = kwargs['info']
    # info = group_info(info)
    assert (len(pos) == len(sym))
    try:
        pbc = [True, True, True]
        atom = Atoms(symbols=sym, positions=pos, cell=cell, pbc=pbc, info=info)
        atom.set_calculator(SinglePointCalculator(atom, energy=energy
                                                  , forces=forces
                                                  , stress=stress
                                                  , magmoms=magmoms))
        atom = check_info(atom)
    except:
        print_format('Cannot create Atoms for %s.' % info['strucname'], level=2)
        atom = None
    return atom


def read(filename):
    """
    Reads structures and its corresponding energy and forces 
    from an input file. Input format should be BOPfox-type
   
    Returns a list of Atoms object
    """
    elements = variables.periodic_table()
    comments = variables.data_comments()
    try:
        f = open(filename, 'r')
    except:
        raise
    atoms = []
    cell = np.zeros((3, 3))
    data_type = None
    for line in f:
        if len(line) < 2:
            continue
        if line[0] in ['#', '/']:
            continue
        if '=' in line:
            key = line.split('=')[0].lower().strip()
            start = 2
        else:
            key = line.split()[0].lower().strip()
            start = 1
        if key == 'data_type':
            if np.count_nonzero(cell) == 0:
                data_type = read_line(line, start, start + 1, int)
                info = {}
                all_keys = variables.data_keys()
                info['data_type'] = data_type
                positions = []
                symbols = []
                energy = None
                forces = []
                stress = []
                magmoms = []
            else:
                if coord.lower()[0] == 'd':
                    positions = np.dot(positions, cell)
                info['data_type'] = data_type
                info = reshape_arrays(info, cell)
                atom = create_atom(positions=positions, symbols=symbols
                                   , cell=cell * alat
                                   , energy=energy, forces=forces, stress=stress
                                   , magmoms=magmoms, info=info)
                if atom is not None:
                    atoms.append(atom)
                struc = read_line(line, start, start + 1, str)
                atom = None
                cell = np.zeros((3, 3))
                positions = []
                symbols = []
                energy = None
                forces = []
                stress = []
                magmoms = []
                info = {}
                all_keys = variables.data_keys()
        elif key == 'a1':
            cell[0] = read_line(line, start, start + 3, list)
        elif key == 'a2':
            cell[1] = read_line(line, start, start + 3, list)
        elif key == 'a3':
            cell[2] = read_line(line, start, start + 3, list)
        elif key == 'alat':
            alat = read_line(line, start, start + 1, float)
        elif key == 'coord':
            coord = read_line(line, start, start + 1, str)
        elif key[0].upper() + key[1:] in elements:
            symbols.append(key[0].upper() + key[1:])
            pos = read_line(line, start, start + 3, list)
            positions.append(pos)
        elif key.lower() == 'energy':
            energy = read_line(line, start, start + 1, float)
        elif key.lower() == 'forces':
            forces.append(read_line(line, start, start + 3, list))
        elif key.lower() == 'magmoms':
            magmoms.append(read_line(line, start, len(line), list))
        elif key.lower() == 'stress':
            stress = read_line(line, start, start + 6, list)
        elif key in list(all_keys.keys()):
            info[key] = all_keys[key]
            if isinstance(all_keys[key], list):
                info[key].append(read_line(line, start
                                           , len(line), type(all_keys[key])))
            else:
                info[key] = read_line(line, start, len(line), type(all_keys[key]))
    if np.count_nonzero(cell) > 0:
        if coord.lower()[0] == 'd':
            positions = np.dot(positions, cell)
        if data_type is None:
            raise ValueError('You must supply key data_type.')
        info['data_type'] = data_type
        info = reshape_arrays(info, cell)
        atom = create_atom(positions=positions, symbols=symbols
                           , cell=cell * alat
                           , energy=energy, forces=forces, stress=stress
                           , magmoms=magmoms, info=info)
        if atom is not None:
            atoms.append(atom)
    f.close()
    return atoms


if __name__ == "__main__":
    from copy import deepcopy

    atoms = read("temp.fit")
    atom = deepcopy(atoms[1])
    print((list(atom.info.keys())))
    print((atom.info['strucname']))
    print((atom.info['data_ID']))
    print((atom.info['system_ID']))
    print((atom.info['xc_functional']))
    print((atom.info['deltak']))
    print((atom.info['encut']))
    print((atom.info['spin']))
    print((atom.get_potential_energy()))
    print((atom.get_forces()))
    # print atom.info['data_details'][0]['xc_functional']
    # print atom.info['data_details'][1]['deltak']
    # print atom.info['data_details'][1]['encut']
    # print atom.info['data_details'][1]['spin']
