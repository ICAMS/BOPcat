import numpy as np
from .utils import get_lattice_type

""" 
Serves as an interface module for the use in strucscan example
"""


def get_deltak(atom):
    nkpts = atom.info['ibz_k_points_weights_rec'].T[-1]
    nkpts = np.sum(nkpts)
    vol = atom.get_volume()
    dk = (nkpts / vol) ** (1 / 3.)
    dk = round(dk, 3)
    return dk


def get_space_group(atom):
    lt, sg = get_lattice_type(atom)
    return sg


def get_sub(atom, calc_type):
    if calc_type in ['volume_relaxed', 'volume_allrelaxed', 'volume_unrelaxed'
        , 'relax', 'relax_all', 'band_structure']:
        sub = atom.get_volume() / len(atom)
        sub = round(sub, 4)
    elif calc_type in ['elastic_relaxed', 'elastic_unrelaxed']:
        name = atom.info['strucname'].split('-')
        con = {'e1': 0, 'e2': 1, 'e3': 2, 'e23': 3, 'e31': 4, 'e12': 5}
        strain = con[name[-2]]
        strp = ''
        for n in name[-1]:
            if n not in '0123456789':
                strp += n
        name[-1] = name[-1].strip(strp)
        dis = 1.0 - float(name[-1])
        sub = strain + dis
    elif calc_type in ['unrelax', 'relax_ion', 'relax_volume', 'relax_all']:
        con = {'unrelax': 0, 'relax_ion': 1, 'relax_volume': 2, 'relax_all': 3}
        sub = con[calc_type]
    elif calc_type in ['unrelaxed']:
        small = 100.
        for i in range(len(atom)):
            for j in range(len(atom)):
                if i < j:
                    d = atom.get_distance(i, j)
                    if d < small:
                        small = d
        sub = small
    return sub


def write_atom(atom):
    """
    Writes the BOPcat.Atom object (NOT ASE) to an outstring (not yet a file)
    """
    out = ''
    info = atom.info
    try:
        info['deltak'] = get_deltak(atom)
    except:
        pass
    cell = atom.get_cell()
    sym = atom.get_chemical_symbols()
    pos = atom.get_positions()
    strucname = info['strucname']
    out += "data_type = 0\n"
    out += "code = 7\n"
    out += "basis_set = 0\n"
    out += "pseudopotential = 30\n"
    out += "encut_ok = 1\n"
    out += "deltak_ok = 1\n"
    out += "strucname = %s\n" % info['strucname']
    out += "alat = 1\n"
    out += "a1 = %20.10f  %20.10f  %20.10f\n" % (cell[0][0], cell[0][1], cell[0][2])
    out += "a2 = %20.10f  %20.10f  %20.10f\n" % (cell[1][0], cell[1][1], cell[1][2])
    out += "a3 = %20.10f  %20.10f  %20.10f\n" % (cell[2][0], cell[2][1], cell[2][2])
    out += "coord = cartesian\n"
    for i in range(len(sym)):
        out += "%s %20.10f  %20.10f  %20.10f\n" % (sym[i]
                                                   , pos[i][0], pos[i][1], pos[i][2])
    try:
        ene = atom.get_potential_energy()
        out += "energy = %20.10f\n" % atom.get_potential_energy()
    except:
        pass
    try:
        forces = atom.get_forces()
        for i in range(len(forces)):
            out += "forces %20.10f  %10.10f  %20.10f\n" % (forces[i][0]
                                                           , forces[i][1], forces[i][2])
    except:
        pass
    try:
        stress = atom.get_stress()
        l = "stress = "
        for i in range(len(stress)):
            l += "%20.10f  " % (stress[i])
        l += '\n'
        out += l
    except:
        pass

    quantities = ['encut', 'deltak'
        , 'eigenvalues', 'k_points', 'coord_k', 'orbital_character'
                  # ,'occupation_numbers','ibz_k_points_weights_rec'
                  # ,'ibz_k_points_weights_car','number_of_iterations'
                  # ,'electronic_temperature'
        , 'number_of_valence_electrons', 'stress', 'number_of_bands'
        , 'mass', 'number_of_k_points', 'spin', 'fermi_level'
        , 'xcf_version', 'magnetic_moments', 'is_converged'
        , 'code', 'basis_set', 'xcfunc', 'pseudopotential'
        , 'system_type', 'calculation_type', 'calculation_order'
        , 'space_group', 'stoichiometry', 'structuremap_coordinates'
        , 'onsites'
                  ]
    key_convert = {'xcfunc': 'xc_functional', 'xcf_version': 'valency'}
    val_code = {'PAW_PBE': 201}
    for key, val in list(info.items()):
        if val is not None and key.lower() in quantities:
            if key in key_convert:
                key = key_convert[key]
            if type(val) in [str, str]:
                if val in val_code:
                    val = val_code[val]
            if type(val) in [str, str]:
                out += "%s = %s\n" % (key, val)
            elif isinstance(val, bool):
                if val:
                    out += "%s = %s\n" % (key, 'True')
                else:
                    out += "%s = %s\n" % (key, 'False')
            elif isinstance(val, float):
                out += "%s = %12.5f\n" % (key, val)
            elif isinstance(val, int):
                out += "%s = %d\n" % (key, val)
            elif isinstance(val, dict):
                out += "%s = " % key
                for key2, val2 in list(val.items()):
                    if type(val2) in [str, str]:
                        out += "%s %s " % (key2, val2)
                    elif isinstance(val2, float):
                        out += "%s %12.5f " % (key2, val2)
                    elif isinstance(val2, int):
                        out += "%s %d " % (key2, val2)
                    else:
                        print(("Cannot add data for %s" % key))
                        break
                out += "\n"
            else:  # expecting array
                if key.lower() in ['eigenvalues', 'k_points', 'occupation_numbers']:
                    for nkpts in range(len(val[0])):
                        out += "%s = " % key
                        for nspin in range(len(val)):
                            for nbands in range(len(val[0][0])):
                                out += "%12.5f " % val[nspin][nkpts][nbands]
                        out += "\n"
                elif key.lower() in ['orbital_character']:
                    # orbital character should have the shape
                    # [nspin][nkpts][nbands][nion+1][9]
                    # last index for each orbital type
                    # only total for all ions will be written out
                    count = 0
                    for nkpts in range(len(val[0])):
                        out += "%s = " % key
                        for nspin in range(len(val)):
                            for nbands in range(len(val[nspin][nkpts])):
                                for norb in range(9):
                                    count += 1
                                    out += "%12.5f  " % val[nspin][nkpts][nbands][norb]
                        out += "\n"
                elif key.lower() in ['onsites']:
                    for i in range(len(val)):
                        out += '%s ' % key
                        for j in range(len(val[i])):
                            out += "%12.5f " % val[i][j]
                        out += "\n"
                else:
                    out += "%s = " % key
                    if len(np.shape(val)) == 1:
                        for i in range(len(val)):
                            out += "%12.5f " % val[i]
                        out += "\n"
                    elif len(np.shape(val)) == 2:
                        for i in range(len(val)):
                            for j in range(len(val[i])):
                                out += "%12.5f " % val[i][j]
                            out += "\n"
                    else:
                        print(("Cannot include data for %s" % key, np.shape(val), val))
                        out += "\n"
    out += "/=====================================\n"
    return out


def dict_to_atoms(data):
    atoms = []
    convert_key = {'bulk': 0, 'cluster': 1, 'defect': 2
        , 'volume_relaxed': 1, 'volume_allrelaxed': 1
        , 'volume_unrelaxed': 1
        , 'relax': 0, 'phonons': 2, 'elastic_relaxed': 3
        , 'elastic_unrelaxed': 3
        , 'band_structure': 4
        , 'unrelaxed': 0
        , 'unrelax': 0, 'relax_ion': 1, 'relax_volume': 2
        , 'relax_all': 3}
    for stoic in list(data.keys()):
        for struc in list(data[stoic].keys()):
            for params in list(data[stoic][struc].keys()):
                for calc_type in list(data[stoic][struc][params].keys()):
                    if data[stoic][struc][params][calc_type] != None:
                        for job in list(data[stoic][struc][params] \
                                                [calc_type].keys()):
                            for sub in list(data[stoic][struc][params][calc_type] \
                                                    [job].keys()):
                                atom = data[stoic][struc][params][calc_type] \
                                    [job][sub]
                                if atom is None:
                                    continue
                                xcfunc = params.split('$')[0]
                                encut = float(params.split('$')[1])
                                # deltak = float(params.split('$')[2])
                                deltak = None
                                spin = int(params.split('$')[3])
                                add_info = {'xcfunc': xcfunc, 'encut': encut
                                    , 'deltak': deltak, 'spin': spin}
                                name = '%s-%s-%s-%s' % (struc, calc_type, job, sub)
                                atom.info.update({'strucname': name})
                                atom.info.update({'system_type': \
                                                      convert_key[calc_type]})
                                atom.info.update({'calculation_type': \
                                                      convert_key[job]})
                                sub = get_sub(atom, job)
                                atom.info.update({'calculation_order': sub})
                                sg = get_space_group(atom)
                                atom.info.update({'space_group': sg})
                                atom.info.update({'stoichiometry': \
                                                      atom.get_chemical_formula()})
                                atom.info.update(add_info)
                                atoms.append(atom)
    return atoms


def write_to_file(atoms, filename):
    """ 
    Writes the set of atoms to a file, including its meta-data using the
    write_atom procedure used above
    """
    f = open(filename, 'w')
    for atom in atoms:
        try:
            lines = write_atom(atom)
            f.write(lines)
        except:
            print(('cannot write atom for %s' % atom.info['strucname']))
    f.close()


def import_data(elements, params, path, filename):
    """
    Allows for loading existing data
    """
    from .get_strucscan_data import extract
    import pickle
    deltak = []
    xcfunc = []
    encut = []
    spin = []
    extended = []
    structures = []
    verbose = True
    calc_type = []
    cmpd_type = list(range(1, len(elements) + 1))
    for key, val in list(params.items()):
        if key.lower() == 'deltak':
            deltak = [val]
        if key.lower() == 'xc_functional':
            xcfunc = [val]
        if key.lower() == 'encut':
            encut = [val]
        if key.lower() == 'spin':
            spin = [val]
    data = extract(cmpd_type=cmpd_type
                   , elements=elements
                   , calc_type=calc_type
                   , structures=structures
                   , xcfunc=xcfunc, encut=encut
                   , deltak=deltak, spin=spin
                   , path=path
                   , verbose=verbose, extended=extended)
    # data = pickle.load(open('dft_data.pckl'))
    atoms = dict_to_atoms(data)
    write_to_file(atoms, filename)
    return atoms


if __name__ == "__main__":
    dft_params = {'xc_functional': 'PBE', 'deltak': 0.02, 'encut': 450, 'spin': 2}
    elements = ['Fe']
    path = 'data'
    filename = 'temp.fit'
    import_data(elements, dft_params, path, filename)
